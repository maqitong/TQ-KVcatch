#!/usr/bin/env python3
"""Merge all PPL runs into one ppl.jsonl + human-readable summary table."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Allow running as script from repo root or runs dir
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from turboquant.block_cache.bpw_metrics import attach_bpw_fields

METHOD_ORDER = [
    "FP16",
    "SKVQ skvq_baseline (native)",
    "TurboQuant V2 paper (QJL)",
    "TurboQuant V3 flat (rw=0)",
    "TurboQuant pure (tq_replace)",
    "TurboQuant TokenBlock",
    "SKVQ TokenBlock",
    "Hybrid+TQ+Block",
    "Hybrid+TQ+Block+PageMix",
    "Hybrid+SKVQ+Block",
    "Hybrid+SKVQ+Block+PageMix",
]

METHOD_BY_BACKEND_POLICY = {
    ("dynamic", "hybrid"): "FP16",
    ("dynamic", "token"): "FP16",
    ("block_tq", "hybrid"): "Hybrid+TQ+Block",
    ("block_tq", "token"): "TurboQuant TokenBlock",
    ("block_tq_mix", "hybrid"): "Hybrid+TQ+Block+PageMix",
    ("block_skvq", "hybrid"): "Hybrid+SKVQ+Block",
    ("block_skvq", "token"): "SKVQ TokenBlock",
    ("block_skvq_mix", "hybrid"): "Hybrid+SKVQ+Block+PageMix",
}

LEGACY_METHOD_RENAME = {
    "TurboQuant Baseline": "TurboQuant TokenBlock",
    "SKVQ Baseline": "SKVQ TokenBlock",
}


def infer_method(row: dict) -> str:
    policy = row.get("config", {}).get("policy", "hybrid")
    backend = row["backend"]
    if backend == "v2_paper":
        return "TurboQuant V2 paper (QJL)"
    if backend == "v3_flat":
        rw = row.get("config", {}).get("residual_window", 0)
        return f"TurboQuant V3 flat (rw={rw})"
    if backend == "skvq_native":
        return "SKVQ skvq_baseline (native)"
    if row.get("method") in LEGACY_METHOD_RENAME:
        return LEGACY_METHOD_RENAME[row["method"]]
    if row.get("method") and row.get("method") not in METHOD_BY_BACKEND_POLICY.values():
        if "native" in row["method"] or "tq_replace" in row["method"] or "pure" in row["method"]:
            return row["method"]
    return METHOD_BY_BACKEND_POLICY.get(
        (backend, policy),
        row.get("method") or f"{backend}({policy})",
    )


def row_key(row: dict) -> tuple:
    cfg = row.get("config", {})
    return (
        row["backend"],
        cfg.get("policy"),
        cfg.get("sink"),
        cfg.get("window"),
        cfg.get("residual_window"),
        cfg.get("protected_layers"),
        cfg.get("mixed"),
        cfg.get("reorder"),
        cfg.get("integration"),
        cfg.get("seq_len"),
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def policy_label(row: dict) -> str:
    cfg = row.get("config", {})
    p = cfg.get("policy", "")
    if p == "v2_paper" or row.get("backend") == "v2_paper":
        return "v2_paper,no_residual_window"
    if p == "v3_flat" or row.get("backend") == "v3_flat":
        return f"v3_flat,rw={cfg.get('residual_window', 0)}"
    if p == "sliding_window":
        return f"window={cfg.get('window')},sink={cfg.get('sink')}"
    if p:
        sink = cfg.get("sink", "")
        window = cfg.get("window", "")
        extra = f",sink={sink},win={window}" if p == "hybrid" else ""
        return f"{p}{extra}"
    return "-"


def _fmt_num(value, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def main() -> None:
    base = Path(__file__).resolve().parent / "server_ppl"
    sources = [
        base / "ppl.jsonl",
        base / "paper_baselines_ppl.jsonl",
        base / "baseline_ppl_temp.jsonl",
        base / "v3_flat_ppl.jsonl",
        base / "v2_paper_ppl.jsonl",
    ]
    out_jsonl = base / "ppl.jsonl"
    out_csv = base / "ppl_summary.csv"

    by_key: dict[tuple, dict] = {}
    for path in sources:
        for row in load_jsonl(path):
            row["method"] = infer_method(row)
            key = row_key(row)
            prev = by_key.get(key)
            if prev is None or row.get("config", {}).get("integration"):
                by_key[key] = row

    def sort_key(row: dict) -> tuple[int, str]:
        method = row["method"]
        try:
            return (METHOD_ORDER.index(method), method)
        except ValueError:
            return (len(METHOD_ORDER), method)

    merged = sorted(by_key.values(), key=sort_key)
    for row in merged:
        attach_bpw_fields(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "method",
        "backend",
        "policy",
        "ppl",
        "k_bpw",
        "v_bpw",
        "avg_bpw",
        "effective_bpw",
        "compression_ratio",
        "loss",
        "tokens",
        "protected_layers",
        "reorder",
        "mixed",
        "sink",
        "window",
        "integration",
        "seconds",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in merged:
            cfg = row.get("config", {})
            writer.writerow(
                {
                    "method": row["method"],
                    "backend": row.get("backend", ""),
                    "policy": policy_label(row),
                    "ppl": _fmt_num(row.get("ppl")),
                    "k_bpw": _fmt_num(row.get("k_bpw")),
                    "v_bpw": _fmt_num(row.get("v_bpw")),
                    "avg_bpw": _fmt_num(row.get("avg_bpw")),
                    "effective_bpw": _fmt_num(row.get("effective_bpw")),
                    "compression_ratio": _fmt_num(row.get("avg_compression_ratio"), 3),
                    "loss": _fmt_num(row.get("loss")),
                    "tokens": row.get("tokens", ""),
                    "protected_layers": cfg.get("protected_layers", ""),
                    "reorder": cfg.get("reorder", cfg.get("reorder_file", "")),
                    "mixed": cfg.get("mixed", ""),
                    "sink": cfg.get("sink", ""),
                    "window": cfg.get("window", ""),
                    "integration": cfg.get("integration", "eval_ppl/BlockKVCache"),
                    "seconds": _fmt_num(row.get("seconds"), 1),
                }
            )

    print(f"Wrote {len(merged)} rows -> {out_jsonl}")
    print(f"Summary table -> {out_csv}\n")
    print(
        f"{'method':<32} {'k':>5} {'v':>5} {'avg':>5} {'eff':>5} "
        f"{'ratio':>6} {'ppl':>8}"
    )
    print("-" * 88)
    for row in merged:
        ppl = row.get("ppl")
        ppl_s = f"{ppl:>8.4f}" if isinstance(ppl, (int, float)) else f"{'':>8}"
        ratio = row.get("avg_compression_ratio")
        ratio_s = f"{ratio:>6.3f}" if isinstance(ratio, (int, float)) else f"{'':>6}"
        eff = row.get("effective_bpw")
        eff_s = f"{eff:>5.2f}" if isinstance(eff, (int, float)) else f"{'':>5}"
        print(
            f"{row['method']:<32} "
            f"{row.get('k_bpw', ''):>5} "
            f"{row.get('v_bpw', ''):>5} "
            f"{row.get('avg_bpw', ''):>5} "
            f"{eff_s} {ratio_s} {ppl_s}"
        )


if __name__ == "__main__":
    main()
