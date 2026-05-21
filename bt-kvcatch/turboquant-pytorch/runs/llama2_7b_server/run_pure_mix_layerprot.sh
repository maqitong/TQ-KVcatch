#!/usr/bin/env bash
# Rerun TurboQuant pure+PageMix with layer-0 K8/V8 (same as main PageMix).
set -eo pipefail
cd "$(dirname "$0")/../.."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL=/root/autodl-tmp/bt-kvcatch/models
REORDER=runs/llama2_7b_calib/reorder_meta.pt
BASE=runs/llama2_7b_server
NIAH_JSONL="$BASE/server_main_exp/main_exp_results.jsonl"
LOG="$BASE/pure_mix_layerprot_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== start $(date -Iseconds) GPU=$CUDA_VISIBLE_DEVICES ==="

echo "=== strip old pure+PageMix NIAH rows ==="
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/llama2_7b_server/server_main_exp/main_exp_results.jsonl")
if not path.exists():
    raise SystemExit(0)
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
keep = [r for r in rows if r.get("method") != "TurboQuant pure+PageMix"]
removed = len(rows) - len(keep)
path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + ("\n" if keep else ""))
print(f"removed {removed} rows, kept {len(keep)}")
PY

echo "=== [1/2] PPL block_tq_pure_mix (layer0 K8/V8) ==="
python -m turboquant.block_cache.eval_ppl \
  --model "$MODEL" --local-files-only \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test \
  --max-samples 64 --max-tokens 8192 --seq-len 1024 --stride 512 \
  --backend block_tq_pure_mix \
  --block-size 16 --key-bits 2 --value-bits 2 \
  --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --num-layers 32 \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --output "$BASE/server_ppl/tq_pure_pagemix_rw128_sink5.jsonl"

echo "=== [2/2] NIAH pure+PageMix only (18 runs) ==="
python -m turboquant.block_cache.experiment_main \
  --model "$MODEL" --local-files-only --reorder-file "$REORDER" \
  --context-lengths 2048,4096 --positions 0.1,0.5,0.9 --seeds 0,1,2 \
  --max-new-tokens 32 --filter-paper-baseline tq_pure_mix \
  --block-size 16 --sink 5 --window 128 --key-bits 2 --value-bits 2 \
  --clipping 0.96 --important-ratio 0.3 --high-key-bits 4 --high-value-bits 4 \
  --low-key-bits 2 --low-value-bits 2 --num-layers 32 \
  --protected-layers 1 --protected-key-bits 8 --protected-value-bits 8 \
  --append-results --output-dir "$BASE/server_main_exp"

echo "=== merge PPL + export tables ==="
python runs/llama2_7b_server/merge_ppl.py
python runs/llama2_7b_server/export_all_results.py

echo "=== done $(date -Iseconds) log=$LOG ==="
