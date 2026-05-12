# Block-Structured KV Cache + TurboQuant 量化框架

本子模块在原 `turboquant-pytorch` 之上加了一层 **块（block）抽象 + 可插拔分组策略**，把汇报里讲的 **token block / window block** 两类分组思想以最小代价工程化，并和 TurboQuant 的逐坐标 Lloyd-Max 量化器接上。

不依赖 vLLM；可直接被 HuggingFace `model.generate(past_key_values=...)` 吃下。

---

## 一、设计概览

```
Layer 4  Demo / Benchmark             demo.py · benchmark.py
Layer 3  HF Cache 适配                BlockKVCache(transformers.Cache)
Layer 2  分组策略（Strategy 接口）    TokenBlockPolicy · WindowBlockPolicy · HybridPolicy
Layer 1  块抽象                       KVBlock · BlockTable · BlockMSECompressor
Layer 0  TurboQuant 原语（已有）      MSECompressor (random rotation + Lloyd-Max)
```

每一层之间是单向依赖。要扩展新策略只用动 Layer 2；要换量化算法只用动 Layer 1 的 `BlockMSECompressor`；上层不用改。

## 二、核心数据流

```
新 K, V (B, H, n, D)
        │
        ▼
BlockKVCache.update(layer_idx)
        │
        ├── 路由进当前层的 BlockTable.append()
        │       └── 切成多个 KVBlock，FILLING → SEALED 状态机
        │
        ├── policy.on_seal(sealed_blocks)  ←——— 这里决定哪些块要量化
        │       └── TokenBlockPolicy: 一律量化
        │           WindowBlockPolicy: 整块跌出窗口才量化
        │           HybridPolicy: sink 永久 fp16 + window + 量化
        │
        ├── 对要量化的块调 BlockMSECompressor.compress
        │       └── 复用 turboquant.compressors_v3.MSECompressor 的旋转 + Lloyd-Max
        │
        └── 把所有块解压拼回 (B, H, S_total, D) 给原生 attention 用
```

**关键设计折中**：MVP 只压缩**存储**，不改 attention 计算。每次 `update()` 会把压缩块解压回 FP16 再 cat。这样可以无侵入地用任何 HF 模型，缺点是运行期峰值显存收益有限 — 真正的收益是 **CPU offload / long-context** 场景下持久化存储的那部分。后续如果要 attention 阶段不解压，需要改 attention kernel（参考 KIVI / SKVQ 的做法），属于扩展工作。

## 三、和报告里论文的对应

| 报告里讲的方法 | 在本框架里的对应 |
|---|---|
| **TurboQuant** (ICLR 2026) | `Layer 0` — 现有 `MSECompressor`：random rotation + per-coordinate Lloyd-Max，复用不动 |
| **PagedAttention / PagedEviction** 的 block table | `BlockTable` + `KVBlock`：固定 `block_size` 分块，每块独立维护状态。**借鉴**了 block table 的索引思路，**但不做物理 page 复用**（不需要 vLLM 的内核） |
| **KIVI** 的 residual window | `WindowBlockPolicy` 在 `sink_size=0, granularity='per-vector'` 下和 KIVI residual buffer 等价 |
| **SKVQ** 的 sliding-window quantization + group-level dynamic quantization | `WindowBlockPolicy` + `granularity='per-block'`：窗口外整块量化，块内共享一个 mean-L2-norm 标量 |
| **PagedEviction** 的 attention-free importance proxy `Si = ‖V‖/‖K‖` | `KVBlock.importance` 字段已预留；MVP 不实现驱逐 |
| **Semantic block (IceCache / SemantiCache)** | 不在本次范围 |

## 四、API 一览

```python
from turboquant.block_cache import (
    BlockKVCache, BlockCacheConfig,
    TokenBlockPolicy, WindowBlockPolicy, HybridPolicy,
)

# 1) 选策略
policy = WindowBlockPolicy(window_size=128)
# 或
# policy = TokenBlockPolicy()
# policy = HybridPolicy(sink_size=4, window_size=128)

# 2) 配置 cache
cfg = BlockCacheConfig(
    block_size=16,        # 每块 16 个 token
    key_bits=6,           # K 量化位数
    value_bits=4,         # V 量化位数（K/V 不对称是 V3 的关键发现）
    granularity="per-vector",  # 或 'per-block'（SKVQ 风格）
    policy=policy,
)

# 3) 喂给 HF generate
cache = BlockKVCache(cfg)
out = model.generate(input_ids, past_key_values=cache, max_new_tokens=64)

# 4) 看效果
print(cache.memory_report())
# {'compressed_bytes': ..., 'fp16_baseline_bytes': ..., 'compression_ratio': 3.4, ...}
```

## 五、扩展点（怎么加新分组方式）

写一个 `GroupingPolicy` 子类，重载 `on_seal` 即可：

```python
from turboquant.block_cache import GroupingPolicy
from turboquant.block_cache.blocks import BlockState

class MyPolicy(GroupingPolicy):
    def on_seal(self, sealed, table):
        # 在这里读 table.blocks 自由决策
        # 返回要立刻被量化的 KVBlock 子集
        return [b for b in sealed if my_condition(b)]
```

要换量化算法（不用 TurboQuant，用别的）：写一个新的 `XxxBlockCompressor`，接口和 `BlockMSECompressor` 一样（`compress(states)->dict` / `decompress(d)->tensor`）即可。

## 六、运行步骤

### 安装

```bash
cd turboquant-pytorch
pip install -r requirements.txt          # 含 torch / transformers / accelerate
pip install -e .                          # 让 import 路径生效
```

GPU 版 PyTorch（按你的 CUDA 版本调）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 跑单元测试（不需要模型权重）

```bash
python -m turboquant.block_cache.test_block_cache
```

预期输出 10 个 `ok:` 行，最后 `All block_cache tests passed.`。

### 跑 demo（需要下载模型）

Qwen3-4B（推荐先跑这个，~8GB）：
```bash
python -m turboquant.block_cache.demo
```

切策略：
```bash
python -m turboquant.block_cache.demo --policy token --block-size 16
python -m turboquant.block_cache.demo --policy window --window 128
python -m turboquant.block_cache.demo --policy hybrid --sink 4 --window 64
```

Llama3-8B（FP16 ~16GB；12GB 卡需要 8-bit 加载）：
```bash
python -m turboquant.block_cache.demo \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --load-in-8bit \
    --policy hybrid --sink 4 --window 128
```

### 横向对比 benchmark

```bash
python -m turboquant.block_cache.benchmark --model Qwen/Qwen3-4B-Instruct
```

会同时跑 4 个配置（FP16 / TokenBlock / WindowBlock / Hybrid），打出针在干草堆（needle-in-haystack）能否找到 + 各自压缩比。

## 七、文件清单

```
turboquant/block_cache/
  __init__.py            # 公开 API
  blocks.py              # KVBlock, BlockTable, BlockState 状态机
  policies.py            # GroupingPolicy 基类 + 三种实现
  quantizer.py           # BlockMSECompressor（per-vector / per-block 双模）
  hf_cache.py            # BlockKVCache：HF Cache 子类，含 reorder/crop
  demo.py                # 单配置最小 demo
  benchmark.py           # 四配置横向对比 + needle-in-haystack
  test_block_cache.py    # 10 个单元测试
README_block_cache.md    # 本文件
```

总新增代码 ≈ 800 行 Python，**不改动**已有 `turboquant/` 任何文件。

## 八、已知限制（写在前面避免误解）

1. **运行期解压**：MVP 在 attention 之前解压回 FP16，所以并发 KV 存储省了，但 attention 那一刻的瞬时显存没省。要彻底省得改 attention kernel。
2. **`crop()` 切到压缩块中段**：会就地把那一块解压回 FP16（带 `warnings`）。HF 的某些路径会调它，已处理但有性能代价。
3. **GQA / MQA**：自动支持，因为我们存的是 `n_kv_heads` 维而不是 `n_attn_heads`。
4. **per-block 量化精度损失**：Lloyd-Max 的码本是按单位方差高斯校准的；per-block 模式只用一个块均值缩放，块内方差不严格归一，会比 per-vector 多一点失真 — 这是有意的对照实验，对应 SKVQ 文章里 group-level 的概念。

## 九、和现有 TurboQuant V3 的关系

`compressors_v3.py` 的 `TurboQuantV3` 已经做了 `residual_window` —— 它本质是 "整段量化 + 最近 N 个 fp16"。和本模块的关系：

| | TurboQuantV3.residual_window | block_cache.WindowBlockPolicy |
|---|---|---|
| 老 token 怎么存 | 一整段 `(B, H, S-rw, D)` 一次性量化 | 按 `block_size` 切分，每块独立量化 |
| 能不能逐块驱逐 | 不能 | 能（接口已留 `KVBlock.importance`） |
| HF generate 集成 | 没有 | 有 `BlockKVCache` |

简单说：本模块是 V3 的"块化重构 + 工程化集成"，老的 V3 仍然保留（合成测试和论文 baseline 还会用）。
