# V3 补充基线

在现有主实验之上新增两条对照：

| 方法 | Backend | 说明 |
|------|---------|------|
| **TurboQuant V3 flat (rw=128, K2/V2)** | `v3_flat` | 作者 V3：`V3FlatCache`，最近 128 token FP16，其余 MSE 量化；**无** block/sink |
| **TurboQuant pure+PageMix** | `block_tq_pure_mix` | 与 `tq_replace` 相同（sink=5、window=128、无 reorder）+ PageMix + **`protected_layers=1`（第 0 层 K8/V8）**，与主实验 `Hybrid+TQ+Block+PageMix` 一致 |

对比关系：

- `block_tq_pure`：块级 TQ（V3 MSE 核）+ 均匀 K2/V2
- `block_tq_pure_mix`：同上块结构 + 页级混合精度
- `v3_flat`：无块；最近 128 token FP16 + 旧段 K2/V2 量化（需重跑 PPL 更新旧 rw=0 结果）

## 运行

```bash
bash runs/llama2_7b_server/run_v3_baselines.sh
```

仅 NIAH：

```bash
python -m turboquant.block_cache.experiment_main \
  --model /path/to/models --local-files-only \
  --reorder-file runs/llama2_7b_calib/reorder_meta.pt \
  --only-v3-baselines --append-results \
  --output-dir runs/llama2_7b_server/server_main_exp
```

## 参数

- `--only-v3-baselines`：只跑上述两条（不含 SKVQ native / 旧 pure）
- `--only-paper-baselines`：四条论文线（native + pure + pure+PageMix + V3 flat）
