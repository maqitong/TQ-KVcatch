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
        "| 方法 | K bpw | V bpw | avg bpw | eff bpw | ratio | PPL | 量化说明 |"
    )
    sep = "|------|------:|------:|--------:|--------:|------:|----:|----------|"
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
    if cfg.get("paper_baseline") == "tq_pure_mix":
        r = cfg.get("important_ratio", 0.3)
        hk, hv = cfg.get("high_key_bits", 4), cfg.get("high_value_bits", 4)
        lk, lv = cfg.get("low_key_bits", 2), cfg.get("low_value_bits", 2)
        pl = cfg.get("protected_layers", 1)
        pk = cfg.get("protected_key_bits", 8)
        pv = cfg.get("protected_value_bits", 8)
        return (
            f"PageMix top {r:.0%}: K/V high {hk}/{hv}, low {lk}/{lv} (k_norm); "
            f"protect layer0 K{pk}/V{pv}"
        )
    if cfg.get("mixed") or cfg.get("mixed_precision"):
        r = cfg.get("important_ratio", 0.3)
        hk, hv = cfg.get("high_key_bits", 4), cfg.get("high_value_bits", 4)
        lk, lv = cfg.get("low_key_bits", 2), cfg.get("low_value_bits", 2)
        m = cfg.get("importance_metric", "k_norm")
        pl = cfg.get("protected_layers", 0)
        if pl and pl > 0:
            pk = cfg.get("protected_key_bits", 8)
            pv = cfg.get("protected_value_bits", 8)
            return (
                f"PageMix top {r:.0%}: K/V high {hk}/{hv}, low {lk}/{lv} ({m}); "
                f"protect layer0 K{pk}/V{pv}"
            )
        return f"PageMix top {r:.0%}: K/V high {hk}/{hv}, low {lk}/{lv} ({m})"
    kb = cfg.get("key_bits", 2)
    vb = cfg.get("value_bits", 2)
    if backend == "skvq_native" or cfg.get("policy") == "sliding_window":
        return f"SKVQ sliding K{kb}/V{vb}, sink={cfg.get('sink',5)}"
    if cfg.get("paper_baseline") == "tq_pure" or cfg.get("reorder") is False:
        return f"TurboQuant uniform K{kb}/V{vb}, no reorder"
    return f"Uniform K{kb}/V{vb}"


def _niah_stats(group: list[dict]) -> dict:
    sample = group[0]
    bpw = row_with_bpw(sample)
    found = sum(int(x["found"]) for x in group)
    ratios = [x.get("compression_ratio") for x in group if x.get("compression_ratio")]
    avg_r = sum(ratios) / len(ratios) if ratios else None
    eff = None
    if avg_r is not None:
        from turboquant.block_cache.bpw_metrics import effective_bpw_kv_pair

        eff = effective_bpw_kv_pair(avg_r)
    by_ctx: dict[int, list] = defaultdict(list)
    for r in group:
        by_ctx[int(r["context_length"])].append(r)
    ctx_cells = []
    for ctx in sorted(by_ctx):
        g = by_ctx[ctx]
        f = sum(int(x["found"]) for x in g)
        ctx_cells.append(f"{ctx}:{f}/{len(g)}")
    return {
        "bpw": bpw,
        "found": found,
        "n": len(group),
        "found_pct": 100 * found / len(group) if group else 0,
        "avg_r": avg_r,
        "eff": eff,
        "ctx_breakdown": ", ".join(ctx_cells),
    }


def export_niah() -> list[str]:
    lines = ["## 2. NIAH（含 K/V BPW）", ""]
    raw = load_jsonl(BASE / "server_main_exp/main_exp_results.jsonl")
    if not raw:
        lines.append("（无数据）")
        return lines

    by_method: dict[str, list] = defaultdict(list)
    for r in raw:
        by_method[r["method"]].append(r)

    header = "| 方法 | K bpw | V bpw | avg bpw | eff bpw | ratio | Found率 | 条数 |"
    sep = "|------|------:|------:|--------:|--------:|------:|--------:|-----:|"
    lines.extend([header, sep])

    for method in sorted(by_method.keys()):
        st = _niah_stats(by_method[method])
        lines.append(
            f"| {method} "
            f"| {st['bpw']['k_bpw']} "
            f"| {st['bpw']['v_bpw']} "
            f"| {st['bpw']['avg_bpw']} "
            f"| {_fmt(st['eff'], 2) if st['eff'] else '-'} "
            f"| {_fmt(st['avg_r'], 3) if st['avg_r'] else '-'} "
            f"| {st['found_pct']:.1f}% "
            f"| {st['n']} |"
        )
    lines.append("")
    return lines


def export_niah_tq_comparison() -> list[str]:
    """TurboQuant pure / V3 flat / pure+PageMix NIAH 对照（同 18 条：2048/4096 × 3 pos × 3 seed）。"""
    raw = load_jsonl(BASE / "server_main_exp/main_exp_results.jsonl")
    order = [
        "TurboQuant pure (tq_replace)",
        "TurboQuant pure+PageMix",
        "TurboQuant V3 flat (rw=128, K2/V2)",
    ]
    notes = {
        "TurboQuant pure (tq_replace)": "BlockKV，sink=5，window=128，均匀 K2/V2，无 reorder",
        "TurboQuant pure+PageMix": "同上 + PageMix + 第0层 K8/V8 保护（同主实验 PageMix）",
        "TurboQuant V3 flat (rw=128, K2/V2)": "V3FlatCache，无 block/sink，residual_window=128，K2/V2",
    }
    lines = [
        "## 2.1 TurboQuant 补充基线 NIAH 对比",
        "",
        "同一 NIAH 协议：`context_length` 2048/4096，`position` 0.1/0.5/0.9，`seed` 0/1/2，共 **18** 条/方法。",
        "",
        "| 方法 | Found | Found率 | ratio | eff bpw | 2048 | 4096 | 说明 |",
        "|------|------:|--------:|------:|--------:|-----:|-----:|------|",
    ]
    for method in order:
        group = [r for r in raw if r.get("method") == method]
        if not group:
            lines.append(f"| {method} | - | - | - | - | - | - | （无数据） |")
            continue
        st = _niah_stats(group)
        by_ctx = defaultdict(lambda: [0, 0])
        for r in group:
            by_ctx[int(r["context_length"])][0] += int(r["found"])
            by_ctx[int(r["context_length"])][1] += 1
        c2048 = by_ctx.get(2048, [0, 0])
        c4096 = by_ctx.get(4096, [0, 0])
        lines.append(
            f"| {method} "
            f"| {st['found']}/{st['n']} "
            f"| {st['found_pct']:.1f}% "
            f"| {_fmt(st['avg_r'], 3) if st['avg_r'] else '-'} "
            f"| {_fmt(st['eff'], 2) if st['eff'] else '-'} "
            f"| {c2048[0]}/{c2048[1]} "
            f"| {c4096[0]}/{c4096[1]} "
            f"| {notes.get(method, '')} |"
        )
    lines.extend(
        [
            "",
            "**解读（简要）**：",
            "- **pure (tq_replace)**：块级均匀 2bit，NIAH 最好（约 72%），与此前论文对齐配置一致。",
            "- **pure+PageMix（含第 0 层 K8/V8）**：PPL 最低；NIAH 约 77.8%，略高于无层保护的 pure（72.2%）；首层 8bit 使压缩比略低于无保护版。",
            "- **V3 flat (rw=128)**：无 block，作者式 residual window；NIAH 约 22%，低于块级 pure。",
            "",
        ]
    )
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
        "> ratio = avg_compression_ratio（KV 显存相对 FP16 的压缩倍数）。",
        "> eff bpw = 32/ratio（相对 FP16 KV 对的整体有效比特）。",
        "",
    ]
    lines.extend(export_ppl())
    lines.extend(export_niah())
    lines.extend(export_niah_tq_comparison())
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
            "ratio", "compression_ratio", "mixed", "high_key_bits", "low_key_bits",
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
                    "ratio": row.get("avg_compression_ratio"),
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
