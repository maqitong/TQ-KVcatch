#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
CTX="2048,4096"

python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --context-lengths "$CTX" --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --block-size 16 --sink 16 --window 128 \
  --key-bits 2 --value-bits 2 --important-ratio 0.3 \
  --high-key-bits 4 --high-value-bits 4 --low-key-bits 2 --low-value-bits 2 \
  --importance-metric k_norm --protected-layers 1 \
  --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 \
  --max-cached-decompressed-blocks 128 --include-random-mix \
  --output-dir "$BASE/server_main_exp"

python -m turboquant.block_cache.profile_memory \
  --model "$MODEL" --local-files-only --backend all \
  --reorder-file "$REORDER" --context-length 4096 --position 0.5 --seed 0 \
  --max-new-tokens 32 --block-size 16 --sink 16 --window 128 \
  --key-bits 2 --value-bits 2 --important-ratio 0.3 \
  --high-key-bits 4 --high-value-bits 4 --low-key-bits 2 --low-value-bits 2 \
  --importance-metric k_norm --protected-layers 1 \
  --protected-key-bits 8 --protected-value-bits 8 \
  --key-group-size 128 --value-group-size 64 \
  --max-cached-decompressed-blocks 128 \
  --output-dir "$BASE/server_profile"

echo "GPU0 pipeline finished"
