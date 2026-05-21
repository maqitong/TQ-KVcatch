#!/usr/bin/env bash
# PPL + NIAH only: TurboQuant V3 flat (rw=128, K2/V2) and TurboQuant pure+PageMix
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
LOG="$BASE/v3_ppl_niah_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== start $(date -Iseconds) GPU=$CUDA_VISIBLE_DEVICES ==="

_common_ppl() {
  python -m turboquant.block_cache.eval_ppl \
    --model "$MODEL" --local-files-only \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test \
    --max-samples 64 --max-tokens 8192 --seq-len 1024 --stride 512 \
    --block-size 16 --key-bits 2 --value-bits 2 \
    --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
    --low-key-bits 2 --low-value-bits 2 --num-layers 32 \
    --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
    "$@"
}

echo "=== [1/3] PPL v3_flat rw=128 K2/V2 ==="
_common_ppl --backend v3_flat --residual-window 128 \
  --output "$BASE/server_ppl/v3_flat_rw128_k2v2.jsonl"

echo "=== [2/3] PPL block_tq_pure_mix ==="
_common_ppl --backend block_tq_pure_mix \
  --output "$BASE/server_ppl/tq_pure_pagemix_rw128_sink5.jsonl"

echo "=== [3/3] NIAH --only-v3-baselines (2 methods x 18 = 36 runs) ==="
python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --context-lengths 2048,4096 --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --only-v3-baselines --residual-window 128 \
  --append-results --output-dir "$BASE/server_main_exp"

echo "=== merge PPL table ==="
python runs/llama2_7b_server/merge_ppl.py

echo "=== done $(date -Iseconds) log=$LOG ==="
