# 论文对齐基线（NIAH / LongBench）

## 新增方法

| 显示名 | NIAH backend | LongBench backend | 对齐配置 |
|--------|--------------|-------------------|----------|
| SKVQ skvq_baseline (native) | `skvq_native` | `skvq_native` | SKVQ `ModelKVCacheManager`，window=128，sink=5，reorder，clip=0.96 |
| TurboQuant V3 flat (rw=128, K2/V2) | `v3_flat` | `v3_flat` | 作者 V3 flat；`residual_window=128`，无 block/sink |
| TurboQuant pure (tq_replace) | `block_tq` + `paper_baseline=tq_pure` | `block_tq_pure` | Hybrid sink=5，window=128，无 reorder，protect=0，clip=0.96 |
| TurboQuant pure+PageMix | `block_tq_pure_mix` | `block_tq_pure_mix` | 同上 + PageMix 30% K4/V4 + **第 0 层 K8/V8**（`protected_layers=1`，同主实验） |

`experiment_main` / `eval_longbench --backend all` 已包含上述两项。

## 当前在跑进程（旧代码）

- **GPU0** `experiment_main`：约 83/144（8 方法），**不含**论文基线；结束后请跑补充脚本。
- **GPU1** `eval_longbench --backend all`：约 98/240（5 个 block backend），**不含** `block_tq_pure` / `skvq_native`；结束后请跑补充脚本。

## 补充脚本（带 resume，结果 append 到同一 jsonl）

```bash
# GPU0：仅 2 个论文方法 × 18 格点 = 36 条 NIAH
bash runs/llama2_7b_server/run_gpu0_paper_niah.sh

# GPU1：2 backend × 48 样本 = 96 条 LongBench
bash runs/llama2_7b_server/run_gpu1_paper_longbench.sh
```

主实验全部重跑时，新 `run_gpu0.sh` / `run_gpu1.sh` 会直接跑满 10 方法 / 7 backend（含论文基线）。
