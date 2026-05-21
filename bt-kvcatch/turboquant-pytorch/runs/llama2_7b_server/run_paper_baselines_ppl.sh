#!/usr/bin/env bash
# Paper-aligned PPL:
#   1) SKVQ repo skvq_baseline (sliding window, sink=5, window=128, reorder)
#   2) turboquant-pytorch pure TurboQuant (tq_replace: no reorder, no layer protect)
set -eo pipefail

MODEL=/root/autodl-tmp/bt-kvcatch/models
TQ_ROOT=/root/autodl-tmp/bt-kvcatch/bt-kvcatch/turboquant-pytorch
SKVQ_ROOT=/root/autodl-tmp/bt-kvcatch/SKVQ
BASE="$TQ_ROOT/runs/llama2_7b_server"
OUT="$BASE/server_ppl/paper_baselines_ppl.jsonl"
SKVQ_OUT="$BASE/server_ppl/skvq_native_baseline.jsonl"
TQ_OUT="$BASE/server_ppl/tq_pure_baseline.jsonl"
SKVQ_CSV="$SKVQ_ROOT/experiments/results/llama2_7b_skvq_native_baseline/exp1_ppl_ablation.csv"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate btkvcatch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -f "$OUT" "$SKVQ_OUT" "$TQ_OUT"

# --- 1) SKVQ paper skvq_baseline (native attention path) ---
echo "=== [1/2] SKVQ native skvq_baseline (wikitext2, k2-v2, window=128, sink=5) ==="
export CUDA_VISIBLE_DEVICES=1
cd "$SKVQ_ROOT"
python run_exp1_ppl_ablation.py \
  --model-path "$MODEL" \
  --model-family llama \
  --methods skvq_baseline \
  --bits 2+2 \
  --datasets wikitext2 \
  --seq-len 2048 \
  --window-size 128 \
  --sink 5 \
  --clip 0.96 \
  --group-size 128 \
  --results-dir experiments/results/llama2_7b_skvq_native_baseline

mkdir -p "$BASE/server_ppl"
python - <<'PY' "$SKVQ_CSV" "$SKVQ_OUT" "$MODEL"
import csv, json, sys
from pathlib import Path
csv_path, out_path, model = sys.argv[1:4]
row = next(r for r in csv.DictReader(open(csv_path)) if r["method"] == "skvq_baseline" and r["status"] == "ok")
out = {
    "method": "SKVQ skvq_baseline (native)",
    "backend": "skvq_native",
    "model": model,
    "tokens": "",
    "loss": "",
    "ppl": float(row["ppl"]),
    "seconds": float(row["elapsed_sec"]),
    "peak_memory_bytes": int(float(row["peak_allocated_gb"]) * 1024**3),
    "config": {
        "policy": "sliding_window",
        "window": int(row["window_size"]),
        "sink": int(row["sink"]),
        "key_bits": 2,
        "value_bits": 2,
        "group_size": int(row["group_size"]),
        "clip": float(row["clip"]),
        "seq_len": int(row["seq_len"]),
        "dataset": row["dataset"],
        "protected_layers": 0,
        "reorder": True,
        "integration": "SKVQ/ModelKVCacheManager",
    },
}
Path(out_path).write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"skvq_baseline ppl={out['ppl']:.6f}")
PY

# --- 2) Pure TurboQuant in turboquant-pytorch (tq_replace equivalent) ---
echo "=== [2/2] TurboQuant pure block_tq (window=128, sink=5, no reorder, protect=0) ==="
cd "$TQ_ROOT"
export CUDA_VISIBLE_DEVICES=1
python -m turboquant.block_cache.eval_ppl \
  --model "$MODEL" --local-files-only --backend block_tq \
  --dataset wikitext --dataset-config wikitext-2-raw-v1 --split test \
  --max-samples 64 --max-tokens 8192 --seq-len 1024 --stride 512 \
  --policy hybrid --sink 5 --window 128 \
  --block-size 16 --key-bits 2 --value-bits 2 \
  --granularity per-vector --group-size 128 \
  --key-group-size 128 --value-group-size 64 \
  --clipping 0.96 \
  --protected-layers 0 \
  --max-cached-decompressed-blocks 0 \
  --output "$TQ_OUT"

python - <<'PY' "$TQ_OUT" "$OUT"
import json, sys
from pathlib import Path
src, dst = sys.argv[1:3]
row = json.loads(Path(src).read_text().strip())
row["method"] = "TurboQuant pure (tq_replace)"
row["config"]["integration"] = "turboquant-pytorch/BlockKVCache"
row["config"]["reorder"] = False
Path(dst).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"tq_pure ppl={row['ppl']:.6f}")
PY

cat "$SKVQ_OUT" >> "$OUT"
python "$TQ_ROOT/runs/llama2_7b_server/merge_ppl.py"
echo "merged all PPL -> $TQ_ROOT/runs/llama2_7b_server/server_ppl/ppl.jsonl"
