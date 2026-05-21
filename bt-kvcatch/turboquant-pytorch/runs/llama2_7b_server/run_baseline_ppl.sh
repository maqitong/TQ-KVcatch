#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
OUT="$BASE/server_ppl/baseline_ppl_temp.jsonl"
rm -f "$OUT"

COMMON=(
  --model "$MODEL" --local-files-only
  --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test
  --max-samples 64 --max-tokens 8192 --seq-len 1024 --stride 512
  --policy token
  --block-size 16 --sink 16 --window 128 --key-bits 2 --value-bits 2
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4
  --low-key-bits 2 --low-value-bits 2 --importance-metric k_norm
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8
  --key-group-size 128 --value-group-size 64
  --reorder-file "$REORDER" --max-cached-decompressed-blocks 128
)

echo "=== TurboQuant Baseline (block_tq, policy=token) ==="
python -m turboquant.block_cache.eval_ppl \
  "${COMMON[@]}" --backend block_tq --output "$OUT"

echo "=== SKVQ Baseline (block_skvq, policy=token) ==="
python -m turboquant.block_cache.eval_ppl \
  "${COMMON[@]}" --backend block_skvq --output "$OUT"

echo "baseline PPL finished"
