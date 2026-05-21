#!/usr/bin/env bash
# Llama-2-7B server experiments (adapted from EXPERIMENTS_4090_README.md)
set -euo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
# Llama-2-7B max_position_embeddings=4096 — skip 8192
CTX="2048,4096"

_common() {
  echo "$@"
}

echo "=== GPU0: experiment_main ==="
CUDA_VISIBLE_DEVICES=0 python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" \
  --local-files-only \
  --reorder-file "$REORDER" \
  --context-lengths "$CTX" \
  --positions 0.1,0.5,0.9 \
  --seeds 0,1,2 \
  --max-new-tokens 32 \
  --block-size 16 \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --important-ratio 0.3 \
  --high-key-bits 4 \
  --high-value-bits 4 \
  --low-key-bits 2 \
  --low-value-bits 2 \
  --importance-metric k_norm \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --max-cached-decompressed-blocks 128 \
  --include-random-mix \
  --output-dir "$BASE/server_main_exp"

echo "=== GPU0: profile_memory ==="
CUDA_VISIBLE_DEVICES=0 python -m turboquant.block_cache.profile_memory \
  --model "$MODEL" \
  --local-files-only \
  --backend all \
  --reorder-file "$REORDER" \
  --context-length 4096 \
  --position 0.5 \
  --seed 0 \
  --max-new-tokens 32 \
  --block-size 16 \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --important-ratio 0.3 \
  --high-key-bits 4 \
  --high-value-bits 4 \
  --low-key-bits 2 \
  --low-value-bits 2 \
  --importance-metric k_norm \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --max-cached-decompressed-blocks 128 \
  --output-dir "$BASE/server_profile"

echo "=== GPU1: eval_ppl ==="
CUDA_VISIBLE_DEVICES=1 python -m turboquant.block_cache.eval_ppl \
  --model "$MODEL" \
  --local-files-only \
  --backend all \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 64 \
  --max-tokens 8192 \
  --seq-len 1024 \
  --stride 512 \
  --policy hybrid \
  --block-size 16 \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --important-ratio 0.3 \
  --high-key-bits 4 \
  --high-value-bits 4 \
  --low-key-bits 2 \
  --low-value-bits 2 \
  --importance-metric k_norm \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --reorder-file "$REORDER" \
  --max-cached-decompressed-blocks 128 \
  --output "$BASE/server_ppl/ppl.jsonl"

echo "=== GPU1: eval_longbench ==="
CUDA_VISIBLE_DEVICES=1 python -m turboquant.block_cache.eval_longbench \
  --model "$MODEL" \
  --local-files-only \
  --backend all \
  --subsets narrativeqa,qasper,multifieldqa_en \
  --max-samples 16 \
  --policy hybrid \
  --block-size 16 \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --important-ratio 0.3 \
  --high-key-bits 4 \
  --high-value-bits 4 \
  --low-key-bits 2 \
  --low-value-bits 2 \
  --importance-metric k_norm \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --reorder-file "$REORDER" \
  --max-cached-decompressed-blocks 128 \
  --output-dir "$BASE/server_longbench"

echo "=== GPU1: ablation scheme ==="
CUDA_VISIBLE_DEVICES=1 python -m turboquant.block_cache.ablation \
  --model "$MODEL" \
  --local-files-only \
  --backend block_tq_mix \
  --reorder-file "$REORDER" \
  --context-lengths "$CTX" \
  --positions 0.1,0.5,0.9 \
  --seeds 0,1,2 \
  --max-new-tokens 32 \
  --policy hybrid \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --max-cached-decompressed-blocks 128 \
  --sweep important_ratio=0.2,0.3,0.5 \
  --sweep high_key_bits=4,6 \
  --sweep high_value_bits=4 \
  --sweep low_key_bits=2 \
  --sweep low_value_bits=2 \
  --sweep block_size=8,16,32 \
  --output-dir "$BASE/server_ablation_scheme"

echo "=== GPU1: ablation metric ==="
CUDA_VISIBLE_DEVICES=1 python -m turboquant.block_cache.ablation \
  --model "$MODEL" \
  --local-files-only \
  --backend block_tq_mix \
  --reorder-file "$REORDER" \
  --context-lengths 2048,4096 \
  --positions 0.1,0.5,0.9 \
  --seeds 0,1,2 \
  --max-new-tokens 32 \
  --policy hybrid \
  --block-size 16 \
  --sink 16 \
  --window 128 \
  --key-bits 2 \
  --value-bits 2 \
  --important-ratio 0.3 \
  --high-key-bits 4 \
  --high-value-bits 4 \
  --low-key-bits 2 \
  --low-value-bits 2 \
  --protected-layers 1 \
  --protected-key-bits 8 \
  --protected-value-bits 8 \
  --key-group-size 128 \
  --value-group-size 64 \
  --max-cached-decompressed-blocks 128 \
  --sweep importance_metric=k_norm,kv_norm,vk_ratio,random \
  --output-dir "$BASE/server_ablation_metric"

echo "=== ALL DONE ==="
