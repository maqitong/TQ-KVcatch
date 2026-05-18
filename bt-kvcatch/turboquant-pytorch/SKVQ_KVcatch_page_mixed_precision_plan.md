# SKVQ + KVcatch + Page-Level Mixed Precision 实现方案

本文档面向 `D:\KVcatch\bt-kvcatch\turboquant-pytorch`，目标是在现有 `turboquant.block_cache` 框架上实现：

- SKVQ 风格的 group-wise KV cache 量化。
- KVcatch 已有的 block/page 级 KV cache 管理。
- Page-level mixed precision，也就是每个 page/block 可以按重要性使用不同 bit-width。

结论先放前面：**主线应放在 `bt-kvcatch`，从 `SKVQ` 项目迁移量化器思想，而不是迁移整套 SKVQ model patch。**

## 1. 当前基础

### 1.1 `bt-kvcatch` 已经有的能力

当前实现主要在：

```text
turboquant/block_cache/
  blocks.py
  policies.py
  quantizer.py
  hf_cache.py
  test_block_cache.py
  demo.py
  benchmark.py
```

其中：

- `blocks.py` 已经实现 `KVBlock` / `BlockTable`，可以把 sequence 切成固定大小 page/block。
- `policies.py` 已经实现 `TokenBlockPolicy`、`WindowBlockPolicy`、`HybridPolicy`。
- `quantizer.py` 已经实现 `BlockMSECompressor`，可以把 sealed block 压缩成 TurboQuant 存储格式。
- `hf_cache.py` 已经实现 `BlockKVCache`，可以接入 HuggingFace `model.generate(past_key_values=...)`。

这说明 `bt-kvcatch` 已经有 page-level cache 管理骨架。接下来缺的是：

- page 级重要性评估。
- page 级 bit-width 分配。
- SKVQ 风格的 page compressor。
- mixed precision 的 memory report 和实验入口。

### 1.2 `SKVQ` 项目中值得迁移的能力

当前 `D:\KVcatch\SKVQ` 中最有价值的部分是：

```text
KV_process.py
KVcache_manager.py
turboquant_backend.py
quant_parser.py
calibration.py
experiments/modeling_llama_skvq.py
```

建议迁移的内容：

- `KV_process.py` 中的 group-wise min/max 量化逻辑。
- `KV_process.py` 中的 reorder / inverse reorder 逻辑。
- `KV_process.py` 中对 `1.5-bit` 的 3-level 表达。
- `turboquant_backend.py` 中的 fallback Lloyd-Max / rotation 经验。
- `calibration.py` 中生成 reorder metadata 的思路。
- `quant_parser.py` 的量化字符串格式经验。

不建议第一版迁移的内容：

- 不迁移 `SlidingKVCacheManager` 的整体结构。
- 不迁移 `experiments/modeling_llama_skvq.py` 的整份 model patch。
- 不在第一版实现 attention-score heavy-hitter，因为那需要改 attention 计算路径。

原因是：`SKVQ` 的 cache 结构是 `sink + quant + window` 三段式，而不是 page table。如果在 `SKVQ` 中做 page-level mixed precision，需要重构 `k_quant/v_quant` 的存储结构，改动会比在 `bt-kvcatch` 上新增 page compressor 更大。

## 2. 总体设计

### 2.1 数据流

目标数据流：

```text
new K/V: (B, H, S_new, D)
        |
        v
BlockKVCache.update(layer_idx)
        |
        v
BlockTable.append()
        |
        v
产生 sealed pages
        |
        v
GroupingPolicy 选择哪些 page 离开 fp16 window
        |
        v
PageImportanceScorer 计算 page 重要性
        |
        v
PageBitAllocator 给每个 page 分配 K/V bit-width
        |
        v
SKVQPageCompressor 按 page 自己的 bits 压缩
        |
        v
KVBlock 进入 COMPRESSED 状态
        |
        v
materialize 时解压回 dense K/V 给 HF attention 使用
```

第一版仍然保持当前 `bt-kvcatch` 的 MVP 原则：**压缩存储，attention 前解压回 dense FP16/BF16**。

也就是说，第一版主要证明：

- page-level mixed precision 存储格式可以工作。
- SKVQ group quant 可以按 page 接入。
- 不同 page 的 bit-width 可以由策略动态决定。
- HuggingFace generate 路径仍然可跑。

暂时不承诺：

- 压缩态直接参与 attention。
- Quest 式 Top-K page selection。
- CUDA/Triton page attention kernel。

## 3. 新增模块规划

### 3.1 `page_importance.py`

新增文件：

```text
turboquant/block_cache/page_importance.py
```

建议定义：

```python
class PageImportanceScorer:
    def score(self, block, table, layer_idx: int) -> float:
        ...

class NormPageImportanceScorer(PageImportanceScorer):
    def score(self, block, table, layer_idx: int) -> float:
        ...
```

第一版使用 K norm：

```python
score = block.fp16_k.float().norm(dim=-1).mean().item()
```

可选扩展：

- `kv_norm`: 同时考虑 K/V norm。
- `k_norm`: 只考虑 K norm，默认推荐。
- `random`: 随机 baseline。
- `position`: 越旧越低精度，作为 ablation。
- `attention_acc`: 后续版本接入 attention score。

第一版推荐：

```text
importance = "k_norm"
```

原因：

- 不需要改 attention。
- 不需要额外存储运行时 attention weight。
- 可以直接在 sealed block 上计算。
- 和 page-level bit allocation 的接口天然匹配。

### 3.2 `bit_allocator.py`

新增文件：

```text
turboquant/block_cache/bit_allocator.py
```

建议定义：

```python
class PageBitAllocator:
    def assign(self, block, table, layer_idx: int) -> tuple[float | int, float | int]:
        ...

class ThresholdPageBitAllocator(PageBitAllocator):
    ...

class TopRatioPageBitAllocator(PageBitAllocator):
    ...
```

第一版推荐 `TopRatioPageBitAllocator`：

```text
important_ratio = 0.2
important page: K4/V2
normal page:    K2/V1.5
```

注意：真正实现时可以先支持整数 bit，`1.5-bit` 作为第二小步接入。原因是现有 `BlockMSECompressor` 只支持整数 bit，而 SKVQ 的 `1.5-bit` 需要 3-level quantization 和特殊 packing。

推荐配置：

```python
PageMixedPrecisionConfig(
    importance="k_norm",
    important_ratio=0.2,
    high_key_bits=4,
    high_value_bits=2,
    low_key_bits=2,
    low_value_bits=1.5,
)
```

### 3.3 `skvq_quantizer.py`

新增文件：

```text
turboquant/block_cache/skvq_quantizer.py
```

核心类：

```python
class SKVQPageCompressor:
    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        group_size: int = 128,
        reorder_idx: dict[str, torch.Tensor] | None = None,
        group_st_idx: dict[str, torch.Tensor] | None = None,
        clipping: float = 0.92,
        fp8_scale: bool = False,
    ):
        ...

    def compress(
        self,
        states: torch.Tensor,
        *,
        bits: int | float,
        ttype: str,
        layer_idx: int,
    ) -> dict:
        ...

    def decompress(self, compressed: dict) -> torch.Tensor:
        ...
```

输入输出约定：

```text
states: (B, H, S, D)
compressed:
  qdata / packed indices
  scale
  zero point
  bits
  shape
  group metadata
  reorder flag
  dtype
```

第一版实现建议：

- 支持 `bits in {2, 3, 4, 8}`。
- 先用 PyTorch 实现，保持简单。
- 支持 fake-dequant roundtrip 验证，但存储格式要保存 packed qdata。
- `1.5-bit` 第二步实现，用 3-level quantization，存储上可先用 2-bit container。

量化流程：

```text
(B, H, S, D)
-> transpose/reshape -> (B, S, H*D)
-> optional smooth scale
-> optional reorder
-> group-wise min/max
-> asymmetric quant
-> pack q indices
-> store scale/zp/metadata
```

解压流程：

```text
packed qdata
-> unpack
-> dequant by scale/zp
-> inverse reorder
-> inverse smooth scale
-> reshape back to (B, H, S, D)
```

### 3.4 扩展 `KVBlock`

修改：

```text
turboquant/block_cache/blocks.py
```

给 `KVBlock` 增加字段：

```python
importance: float = 0.0
key_bits: float | int | None = None
value_bits: float | int | None = None
page_meta: dict | None = None
```

用途：

- `importance`: page 级重要性分数。
- `key_bits/value_bits`: 当前 page 实际使用的 K/V bit-width。
- `page_meta`: 存放调试信息，例如 scorer 名称、rank、threshold、norm 等。

`memory_bytes()` 不需要大改，因为 compressed dict 里已有 tensor 统计。后续可以在 `memory_report()` 中额外统计不同 bit page 数量。

### 3.5 扩展 `BlockCacheConfig`

修改：

```text
turboquant/block_cache/hf_cache.py
```

建议把 config 扩成：

```python
@dataclass
class BlockCacheConfig:
    block_size: int = 16
    key_bits: int = 6
    value_bits: int = 4
    granularity: str = "per-vector"
    seed: int = 42
    policy: GroupingPolicy = field(default_factory=TokenBlockPolicy)

    quant_backend: str = "turboquant"  # "turboquant" | "skvq"
    mixed_precision: bool = False
    important_ratio: float = 0.2
    high_key_bits: int | float = 4
    high_value_bits: int | float = 2
    low_key_bits: int | float = 2
    low_value_bits: int | float = 1.5
    importance_metric: str = "k_norm"
    group_size: int = 128
    clipping: float = 0.92
    reorder_file: str | None = None
```

第一版可以不一次性加完所有字段。最小可用集合：

```python
quant_backend: str = "turboquant"
mixed_precision: bool = False
important_ratio: float = 0.2
high_key_bits: int = 4
high_value_bits: int = 2
low_key_bits: int = 2
low_value_bits: int = 2
importance_metric: str = "k_norm"
group_size: int = 128
clipping: float = 0.92
```

`1.5-bit` 和 `reorder_file` 可以第二步加。

## 4. `BlockCacheLayer.update()` 改造点

当前逻辑大致是：

```python
sealed = self.table.append(key_states, value_states)
if sealed:
    to_compress = self.cfg.policy.on_seal(sealed, self.table)
    for blk in to_compress:
        ck = self.k_compressor.compress(blk.fp16_k)
        cv = self.v_compressor.compress(blk.fp16_v)
        blk.to_compressed(ck, cv)
return self._materialize(key_states.dtype)
```

目标逻辑：

```python
sealed = self.table.append(key_states, value_states)
if sealed:
    to_compress = self.cfg.policy.on_seal(sealed, self.table)
    for blk in to_compress:
        k_bits, v_bits = self.bit_allocator.assign(blk, self.table, self.layer_idx)
        blk.key_bits = k_bits
        blk.value_bits = v_bits

        if self.cfg.quant_backend == "skvq":
            ck = self.skvq_compressor.compress(
                blk.fp16_k,
                bits=k_bits,
                ttype="k",
                layer_idx=self.layer_idx,
            )
            cv = self.skvq_compressor.compress(
                blk.fp16_v,
                bits=v_bits,
                ttype="v",
                layer_idx=self.layer_idx,
            )
        else:
            ck = self.k_compressor_for_bits(k_bits).compress(blk.fp16_k)
            cv = self.v_compressor_for_bits(v_bits).compress(blk.fp16_v)

        blk.to_compressed(ck, cv)
return self._materialize(key_states.dtype)
```

为了避免大改，可以第一版只实现：

```text
quant_backend = "skvq"
mixed_precision = True
low bits / high bits 均为整数
```

TurboQuant mixed precision 可以以后再做 compressor cache。

## 5. Reorder Metadata 接入

SKVQ 的 reorder metadata 格式：

```python
{
    "reorder_indices": [
        (k_indices_layer0, v_indices_layer0),
        ...
    ],
    "cluster_st_inds": [
        (k_group_st_layer0, v_group_st_layer0),
        ...
    ],
}
```

建议在 `BlockKVCache` 初始化时加载一次：

```python
if cfg.reorder_file is not None:
    rod_meta = torch.load(cfg.reorder_file)
```

每个 `BlockCacheLayer` 取当前层：

```python
reorder_idx = {
    "k": rod_meta["reorder_indices"][layer_idx][0],
    "v": rod_meta["reorder_indices"][layer_idx][1],
}
group_st_idx = {
    "k": rod_meta["cluster_st_inds"][layer_idx][0],
    "v": rod_meta["cluster_st_inds"][layer_idx][1],
}
```

第一版可以先不接 reorder，只做 natural group。这样能先跑通 page-level mixed precision。

推荐阶段：

```text
v1: no reorder, natural group
v2: load SKVQ reorder metadata
v3: calibration script 从 SKVQ 迁移/复用
```

## 6. 实验配置

### 6.1 最小 smoke test

配置：

```python
cfg = BlockCacheConfig(
    block_size=16,
    policy=HybridPolicy(sink_size=5, window_size=128),
    quant_backend="skvq",
    mixed_precision=True,
    importance_metric="k_norm",
    important_ratio=0.2,
    high_key_bits=4,
    high_value_bits=2,
    low_key_bits=2,
    low_value_bits=2,
    group_size=128,
    clipping=0.92,
)
```

先用 synthetic tensor 验证：

- append 产生多个 block。
- window 内 block 保持 fp16。
- window 外 block 被压缩。
- compressed block 中存在不同 bit-width。
- `_materialize()` 输出 shape 正确。
- 相对误差在合理范围。

### 6.2 Needle / PPL 对比

建议对比：

```text
FP16 baseline
TurboQuant BlockCache K6/V4
SKVQ PageCache fixed K2/V2
SKVQ PageCache mixed K4/V2 + K2/V2
SKVQ PageCache mixed K4/V2 + K2/V1.5
```

第一轮不要引入太多变量。推荐先固定：

```text
block_size = 16
window = 128
sink = 5
importance = k_norm
important_ratio = 0.2
```

### 6.3 推荐报告指标

每次实验记录：

- PPL / Needle 是否找回。
- compression ratio。
- `n_compressed_blocks`。
- `n_fp16_blocks`。
- `n_high_precision_blocks`。
- `n_low_precision_blocks`。
- decode latency。
- peak GPU memory。

## 7. 实现顺序 Checklist

### Phase 0: 对齐当前工程

- [ ] 跑通当前 `python -m turboquant.block_cache.test_block_cache`。
- [ ] 如果缺依赖，先补 `scipy` 或绕开 `turboquant.__init__` 的导入问题。
- [ ] 确认当前 `BlockKVCache` 在本地模型上仍能跑 demo。

### Phase 1: Page metadata

- [ ] 修改 `KVBlock`，增加 `importance`、`key_bits`、`value_bits`、`page_meta`。
- [ ] 扩展 `memory_report()`，统计不同 bit-width page 数量。
- [ ] 给 `test_block_cache.py` 增加 metadata 基础测试。

### Phase 2: Importance + bit allocation

- [ ] 新增 `page_importance.py`。
- [ ] 实现 `NormPageImportanceScorer`。
- [ ] 新增 `bit_allocator.py`。
- [ ] 实现 `TopRatioPageBitAllocator`。
- [ ] 测试 sealed blocks 能被分成 high/low precision。

### Phase 3: SKVQ page compressor

- [ ] 新增 `skvq_quantizer.py`。
- [ ] 实现 natural group min/max quant。
- [ ] 实现 pack/unpack。
- [ ] 实现 decompress。
- [ ] 支持 `2/3/4/8-bit`。
- [ ] 增加 roundtrip 测试。

### Phase 4: 接入 BlockKVCache

- [ ] 扩展 `BlockCacheConfig`。
- [ ] 在 `BlockCacheLayer.lazy_initialization()` 中初始化 SKVQ compressor。
- [ ] 在 `BlockCacheLayer.update()` 中按 page 分配 bits 并压缩。
- [ ] 在 `_materialize()` 中按 compressed dict 的 backend 自动解压。
- [ ] 跑通现有 block cache 测试。

### Phase 5: 1.5-bit

- [ ] 实现 3-level quantization。
- [ ] 使用 2-bit container 存储 3-level index。
- [ ] 在 memory report 中区分 container bytes 和 effective bits。
- [ ] 增加 `K2/V1.5` 实验。

### Phase 6: SKVQ reorder

- [ ] 支持加载 `reorder_file`。
- [ ] 按 layer 注入 `reorder_idx` 和 `group_st_idx`。
- [ ] compressor 中实现 reorder 和 inverse reorder。
- [ ] 对比 no-reorder / reorder 的 PPL 和 Needle。

### Phase 7: Attention score page importance

- [ ] 评估是否改 HF attention 或接入自定义 model patch。
- [ ] 设计 `AttentionScorePageImportanceScorer`。
- [ ] decode 阶段维护 block-level accumulated attention score。
- [ ] 与 `k_norm` 做 ablation。

## 8. 风险与取舍

### 8.1 第一版不直接做 attention-score

attention score 更贴近 H2O/MiKV，但需要拿到每步 attention weights。当前 `bt-kvcatch` 是 HF Cache 路线，不改 attention 模型文件。强行接 attention score 会让第一版实现变重。

建议：

```text
第一版: k_norm page importance
第二版: attention_acc page importance
```

### 8.2 第一版不做压缩态 attention

当前 `BlockKVCache` 会在 attention 前 materialize 回 dense KV。这个限制已经存在。page-level mixed precision 第一版先解决存储格式和策略问题。

后续如果要真正降低 attention 峰值显存，需要：

- 自定义 attention kernel。
- 或参考 Quest 做 Top-K page selection。
- 或参考 SKVQ kernel 做 packed dequant + attention 融合。

这属于下一阶段，不建议混在第一版。

### 8.3 1.5-bit 的工程表达

SKVQ 的 `1.5-bit` 是 3-level quantization。实际 packing 不能简单用 `8 // 1.5`。第一版可以：

- 先不支持 `1.5-bit`，只做 `2-bit`。
- 或使用 2-bit container 保存 3-level index，并在 report 中按 effective bits 估算。

推荐第二种，但放在 Phase 5。

## 9. 推荐最终 API

理想使用方式：

```python
from turboquant.block_cache import (
    BlockKVCache,
    BlockCacheConfig,
    HybridPolicy,
)

cfg = BlockCacheConfig(
    block_size=16,
    policy=HybridPolicy(sink_size=5, window_size=128),
    quant_backend="skvq",
    mixed_precision=True,
    importance_metric="k_norm",
    important_ratio=0.2,
    high_key_bits=4,
    high_value_bits=2,
    low_key_bits=2,
    low_value_bits=1.5,
    group_size=128,
    clipping=0.92,
)

cache = BlockKVCache(cfg)
out = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=64,
    do_sample=False,
)

print(cache.memory_report())
```

推荐命令行 demo 形式：

```powershell
python -m turboquant.block_cache.demo `
  --policy hybrid `
  --sink 5 `
  --window 128 `
  --quant-backend skvq `
  --mixed-precision `
  --importance k-norm `
  --important-ratio 0.2 `
  --high-key-bits 4 `
  --high-value-bits 2 `
  --low-key-bits 2 `
  --low-value-bits 1.5
```

## 10. 最小改动原则

第一轮实现只改 `turboquant/block_cache/`：

```text
blocks.py
hf_cache.py
test_block_cache.py
__init__.py
page_importance.py       # new
bit_allocator.py         # new
skvq_quantizer.py        # new
```

暂不改：

```text
turboquant/compressors_v3.py
turboquant/turboquant.py
原 SKVQ 项目文件
HF model attention 文件
```

这样可以保持边界清晰：`bt-kvcatch` 负责 page cache 系统，`SKVQ` 作为算法参考和实验结果来源。

## 11. 预期阶段性成果

第一版完成后，应该能说明：

```text
我们在 KVcatch 的 page/block cache 上加入了 SKVQ 风格 group-wise quantization，
并且每个 page 根据 k_norm importance 动态选择不同 K/V bit-width。
相比固定 bit quantization，page-level mixed precision 可以在接近压缩率下提升质量，
或在接近质量下进一步降低 KV cache 存储开销。
```

如果要写成实验标题，可以叫：

```text
SKVQ-KVcatch: Page-Level Mixed-Precision KV Cache Quantization
```

## 12. Review

本方案基于两个现有项目的分工整理：

- `bt-kvcatch` 已有 page/block cache 管理和 HF Cache 接入，因此作为主实现仓库。
- `SKVQ` 已有 group-wise quant、reorder、clip、1.5-bit 和评测经验，因此作为量化算法参考。
- 第一版推荐使用 `k_norm` 做 page importance，避免一开始改 attention 路径。
- 第一版推荐先实现整数 bit mixed precision，再扩展 `1.5-bit` 和 reorder。
- 后续如果要真正降低 attention 峰值显存，需要进入压缩态 attention 或 page selection kernel 阶段。
