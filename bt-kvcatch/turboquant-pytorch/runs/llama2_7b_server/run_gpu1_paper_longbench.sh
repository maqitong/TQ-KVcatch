#!/usr/bin/env bash
# Supplement LongBench with block_tq_pure + skvq_native (paper-aligned).
# Run after main eval_longbench --backend all finishes; supports resume.
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export SKVQ_ROOT=/root/autodl-tmp/bt-kvcatch/SKVQ

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
LOG="$BASE/gpu1_paper.log"

echo "=== GPU1 paper LongBench $(date -Iseconds) ===" | tee "$LOG"

python -m turboquant.block_cache.eval_longbench \
  --model "$MODEL" --local-files-only \
  --backend block_tq_pure,skvq_native \
  --subsets narrativeqa,qasper,multifieldqa_en --max-samples 16 \
  --block-size 16 --key-bits 2 --value-bits 2 \
  --group-size 128 --key-group-size 128 --value-group-size 64 \
  --max-input-tokens 8192 --max-new-tokens 64 \
  --reorder-file "$REORDER" \
  --output-dir "$BASE/server_longbench" \
  --resume-log "$LOG" --append-results \
  2>&1 | tee -a "$LOG"

echo "GPU1 paper LongBench finished $(date -Iseconds)" | tee -a "$LOG"
