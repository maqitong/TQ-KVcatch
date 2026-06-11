# Old-Page Uniform Base + Residual PageMix 方案设计

## 1. 背景和问题定义

当前 `TurboQuant pure` 的结构可以概括为：

```text
recent window / sink / protected tokens: 保持高精度或少量特殊处理
old pages: 统一 K2/V2 TurboQuant 压缩
```

这个结构的速度表现是正常的，因为 old region 的主压缩路径是同 bit-width 的连续 run：

```text
old pages: K2/V2, K2/V2, K2/V2, K2/V2, ...
```

前面已经通过 batch compression、batch materialization、compressed run 存储，把这些 old pages 在物理上提升成 run 级处理，因此 `block_tq_pure` 可以接近 `v3_flat` 的 latency。

问题出现在 `TurboQuant pure PageMix k_norm`：

```text
old pages: K2/V2, K4/V4, K2/V2, K4/V4, ...
protected layer: K8/V8
```

也就是说，recent/protected 机制本身不是主要问题；真正的问题是 old compressed region 内部又做了 page-wise heterogeneous bit allocation，破坏了原本同 bit run 的结构，导致：

- 同 bit batch compression 被拆成多个小 group；
- compressed run 不再连续；
- materialize 时需要更多 slice / merge / cat；
- high/protected bit 本身计算更重；
- 当前 HF attention 仍要求 materialize 回完整 dense KV，这会放大碎片化开销。

因此需要设计一种新的 old-page mixed precision，使其既能保留 PageMix 的“重要 page 更高精度”思想，又不破坏 old region 的主 run。

## 2. 方案核心

采用：

```text
old-page uniform base + important-page residual
```

即：

```text
所有 old pages:
  统一保存 K2/V2 base

重要 old pages:
  额外保存 residual correction
```

逻辑上：

```text
low page:
  KV_hat = Dequant(Base2)

high page:
  KV_hat = Dequant(Base2) + Dequant(Residual)
```

其中 residual 定义为：

```text
Residual = Original_FP16_KV - Dequant(Base2)
```

这个方案不再让 high page 直接变成独立的 `K4/V4` page，而是让所有 page 的主干都走统一 `K2/V2` base。PageMix 只体现在额外 residual 上。

## 3. 与现有 PageMix 的区别

当前 direct PageMix：

```text
low page:  Q2(original)
high page: Q4(original)
```

base + residual PageMix：

```text
low page:  Q2(original)
high page: Q2(original) + Qr(original - Dequant(Q2(original)))
```

两者存储 bit budget 可以接近，但数学上不等价：

- `Q4(original)` 是一次 4-bit 直接量化；
- `Q2(base) + Q2(residual)` 是两级误差修正。

因此它不是原 direct PageMix 的无损替代，而是一个更适合 block/run 后端的新 mixed precision 设计。

## 4. 预期收益

### 4.1 保留 run-level TurboQuant 主路径

所有 old pages 都有统一的 `K2/V2 base`：

```text
base run: K2/V2, K2/V2, K2/V2, K2/V2, ...
```

因此 base 可以继续使用当前已经优化好的：

- batch compression；
- compressed run 存储；
- batch materialization；
- O(1) block metadata；
- 连续 run 解压。

这能保住 `TurboQuant pure` 为什么快的根本原因。

### 4.2 PageMix 只影响 residual side path

重要 page 的 residual 数量通常远少于全部 old pages，例如 `important_ratio=0.3`。因此额外计算只发生在少量 selected pages 上：

```text
base path: all old pages
residual path: important old pages only
```

即使 residual path 有额外开销，也不应该像 direct `K4/V4` PageMix 那样把主 run 打碎。

### 4.3 更利于后续 paged attention

如果未来接入 quantized paged attention，base + residual 的结构更自然：

```text
attention kernel reads:
  base page for all tokens
  optional residual page for important tokens
```

这比直接支持任意 `K2/K4/K8` 混合 page 更容易做成结构化 kernel。

## 5. 精度影响

这个方案会影响 PPL / NIAH / LongBench，不能默认认为等价于 `K4/V4` direct PageMix。

预期排序大概是：

```text
TurboQuant pure:
  latency 最好，PPL 相对差

direct PageMix K4/V4:
  PPL 最好，latency 最差

base + residual PageMix:
  PPL 介于两者之间，latency 更接近 pure
```

如果 residual 设计得好，可能接近 direct `K4/V4` 的 PPL，同时 latency 不会爆炸。

需要重点做三组消融：

```text
A. base K2/V2 + K residual only
B. base K2/V2 + K/V residual
C. direct PageMix K4/V4
```

NIAH 通常更依赖 key 的 attention score 保真，因此 `K residual only` 可能已经明显改善 NIAH；LongBench 和生成质量可能更依赖 value，所以还需要测试 `K/V residual`。

## 6. 推荐的具体设计

### 6.1 配置项

建议新增一个 mixed precision 模式，而不是覆盖现有 direct PageMix：

```python
mixed_precision_mode: str = "direct"
```

可选：

```text
direct:
  当前 PageMix，high page 直接 K4/V4

base_residual:
  所有 old pages K2/V2 base，重要 page 追加 residual
```

新增 residual 配置：

```python
residual_key_bits: float = 2
residual_value_bits: float = 0
residual_granularity: str = "per-vector"
residual_importance_metric: str = "k_norm"
residual_ratio: float = important_ratio
```

其中 `residual_value_bits=0` 表示只给 key 加 residual。

### 6.2 数据结构

在 `KVBlock` 或 compressed payload meta 中增加 residual 字段：

```python
compressed_k_base
compressed_v_base
compressed_k_residual: Optional[dict]
compressed_v_residual: Optional[dict]
page_meta["precision"] = "base" | "base_residual"
```

为了不破坏当前 compressed run 设计，更推荐 layer 级管理：

```python
_tq_base_runs: list[dict]
_tq_residual_runs: list[dict]
```

block 只保存轻量 proxy：

```python
base_run_id
base_run_start
residual_run_id
residual_run_start
```

### 6.3 压缩流程

当前 direct PageMix：

```text
assign high/low bits
compress each bit group
```

base + residual：

```text
1. 对所有 ready old pages 统一做 K2/V2 base batch compression
2. base 解压得到 base_hat
3. 对 important pages 计算 residual = original - base_hat
4. residual 按 residual_key_bits / residual_value_bits 压缩
5. block meta 标记是否有 residual
```

注意第 2 步会带来额外一次 base decompress，但它发生在压缩时，而不是每个 decode step 都重复发生。PPL/prefill 任务中需要评估这个成本。

### 6.4 materialize 流程

当前 direct PageMix：

```text
按原顺序遇到 K2/K4/K8 page
分组解压后再拼回 dense KV
```

base + residual：

```text
1. base run 一次性或少数几次解压，得到 dense base KV
2. residual run 解压 selected important pages
3. 对对应 page slice 做 base += residual
4. 拼接 recent/sink/fp16 blocks
```

这里的关键是：

```text
base materialize 仍然是连续同 bit run
residual 只修正少量 page
```

因此 materialize 的主体复杂度接近 `TurboQuant pure`。

## 7. 实现阶段建议

### Phase 1: 最小可运行原型

目标：验证 PPL 和 latency 是否有希望。

约束：

- 只支持 TurboQuant backend；
- 只支持 `granularity="per-vector"`；
- 只支持 `residual_key_bits=2, residual_value_bits=0`；
- 暂不做 residual run 存储优化，可以先逐 residual page 保存；
- 保持原 direct PageMix 不变。

验证：

```text
pure
direct PageMix K4/V4
base_residual K-residual-only
```

指标：

```text
PPL
Latency
Compression Ratio
Effective bpw
bit/residual histogram
```

### Phase 2: residual run 优化

目标：把 residual side path 也从 page 级提升到 run/bucket 级。

做法：

- important pages 如果连续，合成 residual run；
- 不连续时按 residual bit bucket 分组压缩；
- materialize 时批量解压 residual，再 scatter-add 回 base dense KV。

### Phase 3: K/V residual 扩展

目标：提升 LongBench 和生成质量。

消融：

```text
K residual only
V residual only
K/V residual
residual bits = 1 / 2
important_ratio = 0.1 / 0.2 / 0.3
```

## 8. 风险和注意点

### 8.1 PPL 不一定达到 direct K4/V4

base + residual 是新量化结构，不保证等价于 direct `K4/V4`。

需要通过实验确认：

```text
是否能保留 direct PageMix 的大部分 PPL 收益
```

### 8.2 压缩时需要额外计算 residual

为了得到 residual，需要：

```text
base compress -> base decompress -> original - base_hat
```

这会增加一次性压缩成本。方案是否划算取决于它是否显著减少后续 materialize / mixed-bit fragmentation 成本。

### 8.3 residual 的量化分布不同

residual 不是原始 KV，它的分布可能更接近零均值、小范数。直接复用原 TurboQuant MSE compressor 大概率可行，但不一定最优。

后续可以考虑：

- residual 专用 codebook；
- residual scale clipping；
- residual per-block norm；
- residual sign / QJL correction。

### 8.4 memory report 和 effective bpw 需要重新定义

effective bpw 应该统计：

```text
base bits for all old pages
+ residual bits for important old pages
+ residual metadata/norm overhead
+ fp16 recent/protected overhead
```

不能简单沿用 direct PageMix 的 high/low weighted average。

## 9. centroid search 优化是否需要恢复

不建议恢复。

`80f8ff1` 中的 TurboQuant centroid search 优化是底层等价优化：

```text
旧实现:
  diffs = rotated.unsqueeze(-1) - centroids
  indices = diffs.abs().argmin(dim=-1)

新实现:
  indices = searchsorted(boundaries, rotated)
```

Lloyd-Max centroids 是有序标量 codebook，decision boundary 就是相邻 centroid 的中点。因此 boundary search 与 nearest-centroid 是等价的。测试已经覆盖 2/4/8-bit index 完全一致。

虽然它没有解决 direct PageMix 的 72s 主瓶颈，但原因是 direct PageMix 的主要问题在 mixed-bit fragmentation 和 materialize 结构，而不是单独的 centroid lookup。这个优化本身仍然有价值：

- 不改变 PPL；
- 不改变 compression ratio；
- 不改变 PageMix 语义；
- 对 high bit / residual quantization 仍可能有帮助；
- base + residual 方案也会继续调用 TurboQuant compressor。

因此建议保留它。

如果后续为了做严格 ablation，可以单独加一个 debug flag 切回 brute-force lookup，但不建议在主分支恢复旧实现。

## 10. 推荐汇报表述

可以这样总结：

```text
当前 TurboQuant pure 的 recent/protected 高精度机制本身不是问题；
异常慢来自 old compressed pages 内部的 page-wise mixed precision。
它把原本连续同 bit 的 old-page TurboQuant run 打碎，导致 batch compression、
run-level materialization 和 dense KV 构造收益下降。

因此下一步不再采用简单 direct K2/K4 page 混排，而是设计
old-page uniform base + residual PageMix：
所有 old pages 保持统一 K2/V2 base run，重要 old pages 追加 residual correction。
这样主 KV cache 仍然保持 run-level TurboQuant 的速度，同时用 residual 保留
PageMix 对重要 page 提高精度的能力。
```

