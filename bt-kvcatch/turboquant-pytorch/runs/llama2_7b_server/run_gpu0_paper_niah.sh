#!/usr/bin/env bash
# Supplement NIAH: SKVQ skvq_baseline (native) only (tq_pure already in jsonl).
# Skips completed rows via --append-results + resume logs.
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export SKVQ_ROOT=/root/autodl-tmp/bt-kvcatch/SKVQ

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
LOG="$BASE/gpu0_paper.log"

echo "=== GPU0 paper NIAH $(date -Iseconds) ===" | tee "$LOG"

python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --only-paper-baselines \
  --context-lengths 2048,4096 --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --block-size 16 --group-size 128 \
  --key-bits 2 --value-bits 2 \
  --output-dir "$BASE/server_main_exp" \
  --resume-log "$BASE/gpu0.log" --resume-log "$LOG" \
  --append-results \
  2>&1 | tee -a "$LOG"

echo "GPU0 paper NIAH finished $(date -Iseconds)" | tee -a "$LOG"
