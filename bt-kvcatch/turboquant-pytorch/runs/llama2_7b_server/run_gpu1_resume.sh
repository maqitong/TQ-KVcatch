#!/usr/bin/env bash
# Resume GPU1 pipeline: LongBench (skip 98 done in gpu1.log) + ablations. PPL already done.
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
CTX="2048,4096"
LOG="$BASE/gpu1.log"

echo "=== GPU1 resume: LongBench $(date -Iseconds) ===" | tee -a "$LOG"

python -m turboquant.block_cache.eval_longbench \
  --model "$MODEL" --local-files-only --backend all \
  --subsets narrativeqa,qasper,multifieldqa_en --max-samples 16 --policy hybrid \
  --block-size 16 --sink 16 --window 128 --key-bits 2 --value-bits 2 \
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --importance-metric k_norm \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 --reorder-file "$REORDER" \
  --max-cached-decompressed-blocks 128 \
  --output-dir "$BASE/server_longbench" \
  --resume-log "$LOG" --append-results \
  2>&1 | tee -a "$LOG"

echo "=== GPU1 resume: ablation scheme $(date -Iseconds) ===" | tee -a "$LOG"

python -m turboquant.block_cache.ablation \
  --model "$MODEL" --local-files-only --backend block_tq_mix \
  --reorder-file "$REORDER" --context-lengths "$CTX" \
  --positions 0.1,0.5,0.9 --seeds 0,1,2 --max-new-tokens 32 --policy hybrid \
  --sink 16 --window 128 --key-bits 2 --value-bits 2 \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 --max-cached-decompressed-blocks 128 \
  --sweep important_ratio=0.2,0.3,0.5 --sweep high_key_bits=4,6 \
  --sweep high_value_bits=4 --sweep low_key_bits=2 --sweep low_value_bits=2 \
  --sweep block_size=8,16,32 --output-dir "$BASE/server_ablation_scheme" \
  2>&1 | tee -a "$LOG"

echo "=== GPU1 resume: ablation metric $(date -Iseconds) ===" | tee -a "$LOG"

python -m turboquant.block_cache.ablation \
  --model "$MODEL" --local-files-only --backend block_tq_mix \
  --reorder-file "$REORDER" --context-lengths "$CTX" \
  --positions 0.1,0.5,0.9 --seeds 0,1,2 --max-new-tokens 32 --policy hybrid \
  --block-size 16 --sink 16 --window 128 --key-bits 2 --value-bits 2 \
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --protected-layers 1 \
  --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 --max-cached-decompressed-blocks 128 \
  --sweep importance_metric=k_norm,kv_norm,vk_ratio,random \
  --output-dir "$BASE/server_ablation_metric" \
  2>&1 | tee -a "$LOG"

echo "GPU1 pipeline finished $(date -Iseconds)" | tee -a "$LOG"
echo "论文 LongBench 可在 GPU0 并行跑: bash runs/llama2_7b_server/run_gpu0_paper_longbench.sh" | tee -a "$LOG"
