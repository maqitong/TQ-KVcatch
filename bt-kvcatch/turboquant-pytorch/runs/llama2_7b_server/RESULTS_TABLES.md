# Llama-2-7B-Chat 实验结果汇总（含 K/V BPW）

> K bpw / V bpw：每个压缩权重上 K、V 的平均比特宽度。
> eff bpw = 32/compression_ratio（相对 FP16 KV 对的整体有效比特）。

## 1. WikiText-2 PPL（含 K/V BPW）

| 方法 | K bpw | V bpw | avg bpw | eff bpw | 压缩比 | PPL | 量化说明 |
|------|------:|------:|--------:|--------:|-------:|----:|----------|
| FP16 | 16.0 | 16.0 | 16.0 | 16.0 | - | 6.8022 | FP16 K16/V16 |
| SKVQ skvq_baseline (native) | 2.0 | 2.0 | 2.0 | - | - | 7.6819 | SKVQ sliding K2/V2, sink=5 |
| TurboQuant V2 paper (QJL) | 2.0 | 2.0 | 2.0 | 2.1875 | 14.629 | 53.1032 | Uniform K2/V2 |
| TurboQuant V3 flat (rw=0) | 2.0 | 2.0 | 2.0 | 4.25 | 7.529 | 7.9937 | Uniform K2/V2 |
| TurboQuant pure (tq_replace) | 2.0 | 2.0 | 2.0 | 7.8278 | 4.088 | 7.9132 | TurboQuant uniform K2/V2, no reorder |
| TurboQuant TokenBlock | 2.0 | 2.0 | 2.0 | 5.0412 | 6.348 | 18.8856 | Uniform K2/V2 |
| SKVQ TokenBlock | 2.0 | 2.0 | 2.0 | 5.5406 | 5.775 | 34.4052 | Uniform K2/V2 |
| Hybrid+TQ+Block | 2.0 | 2.0 | 2.0 | 8.9121 | 3.591 | 12.9063 | Uniform K2/V2 |
| Hybrid+TQ+Block+PageMix | 2.6 | 2.6 | 2.6 | 9.9026 | 3.231 | 9.7752 | PageMix top 30%: K/V high 4/4, low 2/2 (k_norm) |
| Hybrid+SKVQ+Block | 2.0 | 2.0 | 2.0 | 9.3402 | 3.426 | 24.3832 | Uniform K2/V2 |
| Hybrid+SKVQ+Block+PageMix | 2.6 | 2.6 | 2.6 | 10.3306 | 3.098 | 15.0935 | PageMix top 30%: K/V high 4/4, low 2/2 (k_norm) |

## 2. NIAH（含 K/V BPW）

| 方法 | K bpw | V bpw | avg bpw | eff bpw | Found率 | 条数 |
|------|------:|------:|--------:|--------:|--------:|-----:|
| FP16 | 16.0 | 16.0 | 16.0 | - | 100.0% | 18 |
| Hybrid+SKVQ+Block | 2.375 | 2.375 | 2.375 | 6.93 | 16.7% | 18 |
| Hybrid+TQ+Block | 2.375 | 2.375 | 2.375 | 6.53 | 22.2% | 18 |
| Hybrid+TQ+Block+PageMix | 2.9507 | 2.9507 | 2.9507 | 7.60 | 16.7% | 18 |
| Hybrid+TQ+RandomMix | 2.9507 | 2.9507 | 2.9507 | 7.59 | 61.1% | 18 |
| SKVQ Baseline | 2.375 | 2.375 | 2.375 | 5.53 | 0.0% | 18 |
| SKVQ skvq_baseline (native) | 2.0 | 2.0 | 2.0 | - | 22.2% | 18 |
| TurboQuant Baseline | 2.375 | 2.375 | 2.375 | 5.07 | 22.2% | 18 |
| TurboQuant pure (tq_replace) | 2.0 | 2.0 | 2.0 | 5.64 | 72.2% | 18 |

## 3. LongBench ROUGE-L（含 K/V BPW）

| Backend | K bpw | V bpw | avg bpw | eff bpw | 平均ROUGE-L | n | 量化说明 |
|---------|------:|------:|--------:|--------:|------------:|--:|----------|
| block_skvq | 2.375 | 2.375 | 2.375 | - | 0.0000 | 3 | Uniform K2/V2 |
| block_tq_mix | 2.9457 | 2.9457 | 2.9457 | - | 0.0330 | 46 | PageMix top 30%: K/V high 4/4, low 2/2 (k_norm) |
| block_tq_pure | 2.0 | 2.0 | 2.0 | - | 0.0667 | 48 | TurboQuant uniform K2/V2, no reorder |
| skvq_native | 2.0 | 2.0 | 2.0 | - | 0.0549 | 48 | SKVQ sliding K2/V2, sink=5 |

### 3.1 按 Subset

| Backend | subset | n | avg ROUGE-L | K bpw | V bpw |
|---------|--------|--:|------------:|------:|------:|
| block_skvq | narrativeqa | 3 | 0.0000 | 2.0 | 2.0 |
| block_tq_mix | multifieldqa_en | 16 | 0.0495 | 2.6 | 2.6 |
| block_tq_mix | narrativeqa | 14 | 0.0000 | 2.6 | 2.6 |
| block_tq_mix | qasper | 16 | 0.0454 | 2.6 | 2.6 |
| block_tq_pure | multifieldqa_en | 16 | 0.1367 | 2.0 | 2.0 |
| block_tq_pure | narrativeqa | 16 | 0.0000 | 2.0 | 2.0 |
| block_tq_pure | qasper | 16 | 0.0635 | 2.0 | 2.0 |
| skvq_native | multifieldqa_en | 16 | 0.0478 | 2.0 | 2.0 |
| skvq_native | narrativeqa | 16 | 0.0518 | 2.0 | 2.0 |
| skvq_native | qasper | 16 | 0.0651 | 2.0 | 2.0 |

## 4. 混合精度（PageMix）K/V 量化方案说明

本仓库 **Hybrid+TQ+Block+PageMix** / **Hybrid+SKVQ+Block+PageMix** 使用 `block_tq_mix` / `block_skvq_mix` + **页级混合精度**（`TopRatioPageBitAllocator`）。

### 4.1 服务器默认配置（`run_gpu0.sh` / `run_gpu1.sh`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `important_ratio` | **0.3** | 按重要性得分取 **Top 30%** 的 KV 页用高精度 |
| `high_key_bits` / `high_value_bits` | **4 / 4** | 重要页：K 4bit，V 4bit |
| `low_key_bits` / `low_value_bits` | **2 / 2** | 其余 70% 页：K 2bit，V 2bit |
| `importance_metric` | **k_norm** | 重要性 = 页内 K 向量范数（`NormPageImportanceScorer`） |
| `block_size` | 16 | 每页 16 个 token 的 KV 块 |
| `policy` | hybrid | sink=16 + window=128 保留 FP16；其余页压缩 |
| `protected_layers` | 1 | 第 0 层 KV 用 **8bit** 保护（非 PageMix 档位） |

### 4.2 理论平均 BPW（仅压缩页，不含 sink/window/FP16 页）

```
K_bpw = V_bpw = ratio × high + (1 − ratio) × low
        = 0.3 × 4 + 0.7 × 2 = 2.6 bit/weight
avg_bpw = (K_bpw + V_bpw) / 2 = 2.6
```

表中的 **effective_bpw** = `32 / compression_ratio`（相对 FP16 的 K16+V16 整对 KV），
会高于 2.6，因为还包含 sink、滑动窗口、未压缩页等 FP16 开销。

### 4.3 与非混合方法对比

| 类型 | Backend | K/V 方案 | 理论 K bpw = V bpw |
|------|---------|----------|-------------------|
| 均匀低比特 | `block_tq` / `block_skvq` | 全页 K2/V2 | 2.0 |
| 页级混合 | `block_tq_mix` / `block_skvq_mix` | 30% 页 K4/V4 + 70% 页 K2/V2 | **2.6** |
| Token 基线 | `block_tq`(token) | 按 token 块均匀 K2/V2 | 2.0 |
| 论文 SKVQ | `skvq_native` | 滑动窗口+sink5，全序列 K2/V2 | 2.0 |
| 论文 TQ | `block_tq_pure` | hybrid sink5，全页 K2/V2，无 reorder | 2.0 |

### 4.4 RandomMix 消融

**Hybrid+TQ+RandomMix** 与 PageMix 相同位宽（4/4 与 2/2），但 `importance_metric=random`，重要性随机分配，用于对照 k_norm。
