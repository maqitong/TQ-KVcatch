#!/usr/bin/env bash
# Supplemental baselines:
#   1) TurboQuant V3 flat (no BlockKVCache)
#   2) TurboQuant pure (tq_replace) + PageMix on BlockKVCache
set -euo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
CTX="2048,4096"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

_common_ppl() {
  python -m turboquant.block_cache.eval_ppl \
    --model "$MODEL" --local-files-only \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test \
    --max-samples 64 --max-tokens 8192 --seq-len 1024 --stride 512 \
    --block-size 16 --key-bits 2 --value-bits 2 \
    --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
    --low-key-bits 2 --low-value-bits 2 --num-layers 32 \
    "$@"
}

echo "=== PPL: V3 flat + pure+PageMix ==="
_common_ppl --backend v3_flat --residual-window 128 \
  --output "$BASE/server_ppl/v3_flat_baseline.jsonl"
_common_ppl --backend block_tq_pure_mix \
  --output "$BASE/server_ppl/tq_pure_pagemix_baseline.jsonl"

echo "=== NIAH: --only-v3-baselines ==="
python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --context-lengths "$CTX" --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --only-v3-baselines --residual-window 128 \
  --append-results --output-dir "$BASE/server_main_exp" \
  2>&1 | tee -a "$BASE/v3_baselines_niah.log"

echo "=== LongBench: v3_flat + block_tq_pure_mix ==="
python -m turboquant.block_cache.eval_longbench \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --subsets narrativeqa,qasper,multifieldqa_en \
  --max-samples 16 --max-input-tokens 8192 --max-new-tokens 64 \
  --backend v3_flat,block_tq_pure_mix --residual-window 128 \
  --append-results --output-dir "$BASE/server_longbench" \
  2>&1 | tee -a "$BASE/v3_baselines_longbench.log"

echo "Done. Merge PPL: python runs/llama2_7b_server/merge_ppl.py"
