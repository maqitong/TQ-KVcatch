# 实验断点记录（停服前）

记录时间：2026-05-20  
工作目录：`/root/autodl-tmp/bt-kvcatch/bt-kvcatch/turboquant-pytorch`  
模型：`/root/autodl-tmp/bt-kvcatch/models`（Llama-2-7B-Chat）  
校准：`runs/llama2_7b_calib/reorder_meta.pt`

---

## 重要说明（明天续跑前必读）

1. **`experiment_main` 与 `eval_longbench` 默认只在全部跑完后才写结果文件**（`main_exp_results.jsonl` / `longbench_results.jsonl`）。中途杀进程时，**已完成条目的结构化结果主要在 `gpu0.log` / `gpu1.log` 里**，目录 `server_main_exp/`、`server_longbench/` 目前可能尚不存在或为空。
2. 官方脚本 **没有 `--resume`**。续跑 = **从断点起重新执行对应命令**；已完成的 backend/方法会重复算（除非改脚本或手工解析日志恢复）。
3. **PPL 已完成**（且后续又补跑了 baseline / V2 / V3 等），见 `server_ppl/ppl.jsonl`、`ppl_summary.csv`。**不要**再跑 `run_gpu1.sh` 里第一段 `eval_ppl --backend all`，除非故意覆盖。

---

## GPU0（`run_gpu0.sh`）— 当前在跑

| 项目 | 状态 |
|------|------|
| 进程 | `experiment_main`（PID 见 `pgrep -f experiment_main`） |
| 日志 | `runs/llama2_7b_server/gpu0.log` |
| 任务 1/2 | NIAH 主实验 → `server_main_exp/`（**未落盘**，跑完才写） |
| 任务 2/2 | `profile_memory`（**未开始**） |

### NIAH 进度（`experiment_main`）

- **计划总量**：8 方法 × 2 上下文(2048,4096) × 3 位置 × 3 seed = **144**（`--include-random-mix`）
- **已完成**：**83 / 144**（约 58%）
- **2048**：**72 / 72**（已全部完成，含 RandomMix）
- **4096**：**20 / 72**（`pos=0.10` 下 8 方法×3 seed 中已完成 20 条，差 **RandomMix seed=2** 等）

**下一条应跑（断点）**：

```text
method=Hybrid+TQ+RandomMix  ctx=4096  pos=0.10  seed=2
```

（上一条已完成：`Hybrid+TQ+Block+PageMix ctx=4096 pos=0.10 seed=2`）

**4096 剩余概要**：

- 完成：`pos=0.10` 下 FP16 / SKVQ Baseline / TQ Baseline / Hybrid+SKVQ / Hybrid+TQ / PageMix / RandomMix(seed0,1)，以及各方法 seed2 中除 RandomMix 外
- 未完成：`'Hybrid+TQ+RandomMix', 4096, 0.1, 2` 起 → 随后 `pos=0.50`、`0.90` 全部方法

### 停 GPU0 后明天续跑 NIAH

```bash
cd /root/autodl-tmp/bt-kvcatch/bt-kvcatch/turboquant-pytorch
source /root/miniconda3/etc/profile.d/conda.sh && conda activate btkvcatch
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 仅 NIAH（与 run_gpu0.sh 前半相同）
python -m turboquant.block_cache.experiment_main \
  --model /root/autodl-tmp/bt-kvcatch/models --local-files-only \
  --reorder-file runs/llama2_7b_calib/reorder_meta.pt \
  --context-lengths 2048,4096 --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --block-size 16 --sink 16 --window 128 \
  --key-bits 2 --value-bits 2 --important-ratio 0.3 \
  --high-key-bits 4 --high-value-bits 4 --low-key-bits 2 --low-value-bits 2 \
  --importance-metric k_norm --protected-layers 1 \
  --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 \
  --max-cached-decompressed-blocks 128 --include-random-mix \
  --output-dir runs/llama2_7b_server/server_main_exp \
  2>&1 | tee -a runs/llama2_7b_server/gpu0.log
```

注意：会从 **144 条从头再跑**（除非先改代码支持 skip）。若要避免重复，需根据 `gpu0.log` 里 83 条 `=== method=...` 解析后改 `experiment_main` 加 resume。

NIAH 完成后再跑 profile：

```bash
python -m turboquant.block_cache.profile_memory \
  --model /root/autodl-tmp/bt-kvcatch/models --local-files-only --backend all \
  --reorder-file runs/llama2_7b_calib/reorder_meta.pt \
  --context-length 4096 --position 0.5 --seed 0 --max-new-tokens 32 \
  --block-size 16 --sink 16 --window 128 --key-bits 2 --value-bits 2 \
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --importance-metric k_norm \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 --max-cached-decompressed-blocks 128 \
  --output-dir runs/llama2_7b_server/server_profile
```

---

## GPU1（`run_gpu1.sh`）— LongBench 曾被中断

| 阶段 | 状态 |
|------|------|
| `eval_ppl --backend all` | **已完成**（见 `gpu1.log` 前 30 行；结果已合并到 `server_ppl/ppl.jsonl`） |
| `eval_longbench` | **中断**（`gpu1.log` 末尾 `Terminated`） |
| `ablation` ×2 | **未开始** |

### LongBench 进度（3 subsets × 16 samples × 5 backends = **240**）

| backend | 完成数 | 状态 |
|---------|--------|------|
| `dynamic` | 48/48 | 完成 |
| `block_tq` | 48/48 | 完成 |
| `block_tq_mix` | **2/48** | 仅 `narrativeqa` sample **0、1** |
| `block_skvq` | 0/48 | 未开始 |
| `block_skvq_mix` | 0/48 | 未开始 |

**合计已完成：98 / 240**（约 41%）

**断点（下一条）**：

```text
backend=block_tq_mix  subset=narrativeqa  sample=2
```

（上一条已完成：`block_tq_mix narrativeqa sample=1`）

**中断前日志特征**：`block_tq` 在 `multifieldqa_en` 部分 sample 41–46 出现 `score=0`、空 `prediction`（长上下文 8192 token），属已跑完但质量差，仍算“已执行”。

### 明天续跑 LongBench（推荐按 backend 拆开，避免重跑 dynamic/block_tq）

```bash
cd /root/autodl-tmp/bt-kvcatch/bt-kvcatch/turboquant-pytorch
source /root/miniconda3/etc/profile.d/conda.sh && conda activate btkvcatch
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) 补完 block_tq_mix（会从 sample 0 重跑该 backend 全部 48 条，除非改脚本）
python -m turboquant.block_cache.eval_longbench \
  --model /root/autodl-tmp/bt-kvcatch/models --local-files-only \
  --backend block_tq_mix \
  --subsets narrativeqa,qasper,multifieldqa_en --max-samples 16 --policy hybrid \
  --block-size 16 --sink 16 --window 128 --key-bits 2 --value-bits 2 \
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --importance-metric k_norm \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 \
  --reorder-file runs/llama2_7b_calib/reorder_meta.pt \
  --max-cached-decompressed-blocks 128 \
  --output-dir runs/llama2_7b_server/server_longbench_tq_mix \
  2>&1 | tee -a runs/llama2_7b_server/gpu1_longbench_tq_mix.log

# 2) 未开始的两个 backend
python -m turboquant.block_cache.eval_longbench \
  ... --backend block_skvq --output-dir runs/llama2_7b_server/server_longbench_skvq ...

python -m turboquant.block_cache.eval_longbench \
  ... --backend block_skvq_mix --output-dir runs/llama2_7b_server/server_longbench_skvq_mix ...
```

合并结果时需把各子目录的 `longbench_results.jsonl` 拼在一起；**不要**再对整个 `run_gpu1.sh` 无脑 `nohup`，否则会重做 PPL。

### GPU1 后续队列（未动）

- `server_ablation_scheme`（`ablation` important_ratio / block_size 等 sweep）
- `server_ablation_metric`（importance_metric sweep）

---

## 停服操作建议

```bash
# 查看进程
pgrep -af 'run_gpu0|experiment_main|run_gpu1|eval_longbench'

# 优雅结束（在 turboquant-pytorch 目录）
kill $(pgrep -f 'run_gpu0.sh') 2>/dev/null
kill $(pgrep -f 'turboquant.block_cache.experiment_main') 2>/dev/null
# GPU1 若已无 eval_longbench 可省略
```

**可以明天再跑**；数据与日志在 `/root/autodl-tmp/` 下，实例重启后路径不变即可续。

---

## 已完成且可保留的结果（无需重跑）

| 内容 | 路径 |
|------|------|
| PPL 总表 | `runs/llama2_7b_server/server_ppl/ppl.jsonl`、`ppl_summary.csv` |
| SKVQ native / TQ pure / V3 flat / V2 paper 等 | 同目录 `*_ppl.jsonl` |
| NIAH 过程日志 | `runs/llama2_7b_server/gpu0.log`（83 条） |
| LongBench 过程日志 | `runs/llama2_7b_server/gpu1.log`（98 条） |
