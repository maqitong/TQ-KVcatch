#!/usr/bin/env python3
"""Export experiment tables with k_bpw / v_bpw / avg_bpw / effective_bpw."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from turboquant.block_cache.bpw_metrics import attach_bpw_fields

BASE = Path(__file__).resolve().parent
OUT_MD = BASE / "RESULTS_TABLES.md"
OUT_DIR = BASE


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def method_label(row: dict) -> str:
    return row.get("method") or row.get("backend", "?")


def row_with_bpw(row: dict) -> dict:
    payload = {
        "method": method_label(row),
        "backend": row.get("backend", ""),
        "config": row.get("config") or {},
        "compression_ratio": row.get("compression_ratio") or row.get("avg_compression_ratio"),
        "bit_histogram": row.get("bit_histogram"),
    }
    attach_bpw_fields(payload)
    return payload


def export_ppl() -> list[str]:
    lines = ["## 1. WikiText-2 PPL（含 K/V BPW）", ""]
    rows = load_jsonl(BASE / "server_ppl/ppl.jsonl")
    if not rows:
        rows = load_jsonl(BASE / "server_ppl/ppl_summary.csv")  # fallback skip
    for row in rows:
        attach_bpw_fields(row)
    if not rows:
        lines.append("（无 PPL 数据）")
        return lines

    header = (
        "| 方法 | K bpw | V bpw | avg bpw | eff bpw | 压缩比 | PPL | 量化说明 |"
    )
    sep = "|------|------:|------:|--------:|--------:|-------:|----:|----------|"
    lines.extend([header, sep])
    for row in rows:
        cfg = row.get("config", {})
        scheme = _quant_scheme_note(row)
        lines.append(
            f"| {row.get('method','')} "
            f"| {row.get('k_bpw','')} "
            f"| {row.get('v_bpw','')} "
            f"| {row.get('avg_bpw','')} "
            f"| {row.get('effective_bpw') or '-'} "
            f"| {_fmt(row.get('avg_compression_ratio'), 3)} "
            f"| {_fmt(row.get('ppl'), 4)} "
            f"| {scheme} |"
        )
    lines.append("")
    return lines


def _quant_scheme_note(row: dict) -> str:
    cfg = row.get("config") or {}
    backend = row.get("backend", "")
    if backend == "dynamic":
        return "FP16 K16/V16"
    if cfg.get("mixed") or cfg.get("mixed_precision"):
        r = cfg.get("important_ratio", 0.3)
        hk, hv = cfg.get("high_key_bits", 4), cfg.get("high_value_bits", 4)
        lk, lv = cfg.get("low_key_bits", 2), cfg.get("low_value_bits", 2)
        m = cfg.get("importance_metric", "k_norm")
        return f"PageMix top {r:.0%}: K/V high {hk}/{hv}, low {lk}/{lv} ({m})"
    kb = cfg.get("key_bits", 2)
    vb = cfg.get("value_bits", 2)
    if backend == "skvq_native" or cfg.get("policy") == "sliding_window":
        return f"SKVQ sliding K{kb}/V{vb}, sink={cfg.get('sink',5)}"
    if cfg.get("paper_baseline") == "tq_pure" or cfg.get("reorder") is False:
        return f"TurboQuant uniform K{kb}/V{vb}, no reorder"
    return f"Uniform K{kb}/V{vb}"


def export_niah() -> list[str]:
    lines = ["## 2. NIAH（含 K/V BPW）", ""]
    raw = load_jsonl(BASE / "server_main_exp/main_exp_results.jsonl")
    if not raw:
        lines.append("（无数据）")
        return lines

    by_method: dict[str, list] = defaultdict(list)
    for r in raw:
        by_method[r["method"]].append(r)

    header = "| 方法 | K bpw | V bpw | avg bpw | eff bpw | Found率 | 条数 |"
    sep = "|------|------:|------:|--------:|--------:|--------:|-----:|"
    lines.extend([header, sep])

    for method in sorted(by_method.keys()):
        group = by_method[method]
        sample = group[0]
        bpw = row_with_bpw(sample)
        found = sum(int(x["found"]) for x in group)
        ratios = [x.get("compression_ratio") for x in group if x.get("compression_ratio")]
        eff = None
        if ratios:
            avg_r = sum(ratios) / len(ratios)
            from turboquant.block_cache.bpw_metrics import effective_bpw_kv_pair

            eff = effective_bpw_kv_pair(avg_r)
        lines.append(
            f"| {method} "
            f"| {bpw['k_bpw']} "
            f"| {bpw['v_bpw']} "
            f"| {bpw['avg_bpw']} "
            f"| {_fmt(eff, 2) if eff else '-'} "
            f"| {100*found/len(group):.1f}% "
            f"| {len(group)} |"
        )
    lines.append("")
    return lines


def export_longbench() -> list[str]:
    lines = ["## 3. LongBench ROUGE-L（含 K/V BPW）", ""]
    raw = load_jsonl(BASE / "server_longbench/longbench_results.jsonl")
    if not raw:
        lines.append("（无数据）")
        return lines

    by_backend: dict[str, list] = defaultdict(list)
    for r in raw:
        by_backend[r["backend"]].append(r)

    header = (
        "| Backend | K bpw | V bpw | avg bpw | eff bpw | 平均ROUGE-L | n | 量化说明 |"
    )
    sep = "|---------|------:|------:|--------:|--------:|------------:|--:|----------|"
    lines.extend([header, sep])

    for backend in sorted(by_backend.keys()):
        group = by_backend[backend]
        sample = {"backend": backend, "config": group[0].get("config", {})}
        if group[0].get("bit_histogram"):
            sample["bit_histogram"] = group[0]["bit_histogram"]
        ratios = [x.get("compression_ratio") for x in group if x.get("compression_ratio")]
        if ratios:
            sample["compression_ratio"] = sum(ratios) / len(ratios)
        bpw = row_with_bpw(sample)
        avg_score = sum(x["score"] for x in group) / len(group)
        note_row = {"backend": backend, "config": group[0].get("config", {})}
        lines.append(
            f"| {backend} "
            f"| {bpw['k_bpw']} "
            f"| {bpw['v_bpw']} "
            f"| {bpw['avg_bpw']} "
            f"| {bpw.get('effective_bpw') or '-'} "
            f"| {avg_score:.4f} "
            f"| {len(group)} "
            f"| {_quant_scheme_note(note_row)} |"
        )

    lines.extend(["", "### 3.1 按 Subset", ""])
    lines.append("| Backend | subset | n | avg ROUGE-L | K bpw | V bpw |")
    lines.append("|---------|--------|--:|------------:|------:|------:|")
    key_groups: dict[tuple, list] = defaultdict(list)
    for r in raw:
        key_groups[(r["backend"], r["subset"])].append(r)
    for (backend, subset), group in sorted(key_groups.items()):
        bpw = row_with_bpw({"backend": backend, "config": group[0].get("config", {})})
        avg = sum(x["score"] for x in group) / len(group)
        lines.append(
            f"| {backend} | {subset} | {len(group)} | {avg:.4f} "
            f"| {bpw['k_bpw']} | {bpw['v_bpw']} |"
        )
    lines.append("")
    return lines


def mixed_precision_doc() -> list[str]:
    return [
        "## 4. 混合精度（PageMix）K/V 量化方案说明",
        "",
        "本仓库 **Hybrid+TQ+Block+PageMix** / **Hybrid+SKVQ+Block+PageMix** 使用 "
        "`block_tq_mix` / `block_skvq_mix` + **页级混合精度**（`TopRatioPageBitAllocator`）。",
        "",
        "### 4.1 服务器默认配置（`run_gpu0.sh` / `run_gpu1.sh`）",
        "",
        "| 参数 | 值 | 含义 |",
        "|------|-----|------|",
        "| `important_ratio` | **0.3** | 按重要性得分取 **Top 30%** 的 KV 页用高精度 |",
        "| `high_key_bits` / `high_value_bits` | **4 / 4** | 重要页：K 4bit，V 4bit |",
        "| `low_key_bits` / `low_value_bits` | **2 / 2** | 其余 70% 页：K 2bit，V 2bit |",
        "| `importance_metric` | **k_norm** | 重要性 = 页内 K 向量范数（`NormPageImportanceScorer`） |",
        "| `block_size` | 16 | 每页 16 个 token 的 KV 块 |",
        "| `policy` | hybrid | sink=16 + window=128 保留 FP16；其余页压缩 |",
        "| `protected_layers` | 1 | 第 0 层 KV 用 **8bit** 保护（非 PageMix 档位） |",
        "",
        "### 4.2 理论平均 BPW（仅压缩页，不含 sink/window/FP16 页）",
        "",
        "```",
        "K_bpw = V_bpw = ratio × high + (1 − ratio) × low",
        "        = 0.3 × 4 + 0.7 × 2 = 2.6 bit/weight",
        "avg_bpw = (K_bpw + V_bpw) / 2 = 2.6",
        "```",
        "",
        "表中的 **effective_bpw** = `32 / compression_ratio`（相对 FP16 的 K16+V16 整对 KV），",
        "会高于 2.6，因为还包含 sink、滑动窗口、未压缩页等 FP16 开销。",
        "",
        "### 4.3 与非混合方法对比",
        "",
        "| 类型 | Backend | K/V 方案 | 理论 K bpw = V bpw |",
        "|------|---------|----------|-------------------|",
        "| 均匀低比特 | `block_tq` / `block_skvq` | 全页 K2/V2 | 2.0 |",
        "| 页级混合 | `block_tq_mix` / `block_skvq_mix` | 30% 页 K4/V4 + 70% 页 K2/V2 | **2.6** |",
        "| Token 基线 | `block_tq`(token) | 按 token 块均匀 K2/V2 | 2.0 |",
        "| 论文 SKVQ | `skvq_native` | 滑动窗口+sink5，全序列 K2/V2 | 2.0 |",
        "| 论文 TQ | `block_tq_pure` | hybrid sink5，全页 K2/V2，无 reorder | 2.0 |",
        "",
        "### 4.4 RandomMix 消融",
        "",
        "**Hybrid+TQ+RandomMix** 与 PageMix 相同位宽（4/4 与 2/2），但 "
        "`importance_metric=random`，重要性随机分配，用于对照 k_norm。",
        "",
    ]


def _fmt(v, digits=2):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    lines = [
        "# Llama-2-7B-Chat 实验结果汇总（含 K/V BPW）",
        "",
        "> K bpw / V bpw：每个压缩权重上 K、V 的平均比特宽度。",
        "> eff bpw = 32/compression_ratio（相对 FP16 KV 对的整体有效比特）。",
        "",
    ]
    lines.extend(export_ppl())
    lines.extend(export_niah())
    lines.extend(export_longbench())
    lines.extend(mixed_precision_doc())

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")

    # CSV mirrors
    ppl_rows = load_jsonl(BASE / "server_ppl/ppl.jsonl")
    for r in ppl_rows:
        attach_bpw_fields(r)
    if ppl_rows:
        csv_path = BASE / "results_ppl_with_bpw.csv"
        fields = [
            "method", "backend", "ppl", "k_bpw", "v_bpw", "avg_bpw", "effective_bpw",
            "compression_ratio", "mixed", "high_key_bits", "low_key_bits",
            "high_value_bits", "low_value_bits", "important_ratio",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in ppl_rows:
                cfg = row.get("config", {})
                w.writerow({
                    "method": row.get("method"),
                    "backend": row.get("backend"),
                    "ppl": row.get("ppl"),
                    "k_bpw": row.get("k_bpw"),
                    "v_bpw": row.get("v_bpw"),
                    "avg_bpw": row.get("avg_bpw"),
                    "effective_bpw": row.get("effective_bpw"),
                    "compression_ratio": row.get("avg_compression_ratio"),
                    "mixed": cfg.get("mixed") or cfg.get("mixed_precision"),
                    "high_key_bits": cfg.get("high_key_bits"),
                    "low_key_bits": cfg.get("low_key_bits"),
                    "high_value_bits": cfg.get("high_value_bits"),
                    "low_value_bits": cfg.get("low_value_bits"),
                    "important_ratio": cfg.get("important_ratio"),
                })
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
