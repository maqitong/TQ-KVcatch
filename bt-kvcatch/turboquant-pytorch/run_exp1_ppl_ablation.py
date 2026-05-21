"""
Experiment 1: PPL ablation for bt-kvcatch BlockKVCache (aligned with SKVQ/exp1).

Methods (bt-kvcatch / KVcatch page cache):
  - fp16: HuggingFace baseline (no BlockKVCache)
  - tq_block: TurboQuant block cache, fixed K/V bits (analogous to SKVQ tq_replace at page level)
  - tq_block_mixed: Hybrid + TurboQuant + block + page mixed precision (top-ratio allocator)
  - skvq_page_fixed: SKVQ page compressor, fixed bits per page
  - skvq_page_mixed: SKVQ page compressor + TopRatio mixed precision

Defaults mirror SKVQ/run_exp1_ppl_ablation.py:
  WikiText-2, seq_len 2048, bits 2+2 / 2+1.5 / 1.5+1.5,
  group_size=128, window_size=128, sink=5, clip=0.96, calib_samples=256.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    HybridPolicy,
)

DEFAULT_METHODS = "fp16,tq_block,skvq_page_fixed,skvq_page_mixed"
DEFAULT_BITS = "2+2,2+1.5,1.5+1.5"
DEFAULT_DATASETS = "wikitext2"
DEFAULT_SEQ_LENS = "2048"

SUMMARY_METHOD_ORDER = [
    "fp16",
    "tq_block",
    "tq_block_mixed",
    "skvq_page_fixed",
    "skvq_page_mixed",
]

CSV_FIELDS = [
    "dataset",
    "seq_len",
    "method",
    "kbits",
    "vbits",
    "status",
    "ppl",
    "elapsed_sec",
    "peak_allocated_gb",
    "n_eval",
    "calib_dataset",
    "calib_samples",
    "group_size",
    "window_size",
    "sink",
    "clip",
    "protect_layers",
    "protected_bits",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bt-kvcatch Exp1: PPL ablation (BlockKVCache)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--results-dir", default="experiments/results/exp1")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS)
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=0, help="0 = all WikiText-2 test chunks")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--sink", type=int, default=5)
    p.add_argument("--clip", type=float, default=0.96)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--bits", default=DEFAULT_BITS)
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--datasets", default=DEFAULT_DATASETS)
    p.add_argument("--calib-dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--important-ratio", type=float, default=0.2)
    p.add_argument("--importance-metric", default="k_norm")
    p.add_argument("--force-recalib", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_bit_value(value: str) -> int | float:
    value = value.strip()
    parsed = float(value) if "." in value else int(value)
    return int(parsed) if isinstance(parsed, float) and parsed.is_integer() else parsed


def bit_tag(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_bits(spec: str) -> list[tuple[int | float, int | float]]:
    result = []
    for pair in spec.split(","):
        if not pair.strip():
            continue
        k, v = pair.strip().split("+")
        result.append((parse_bit_value(k), parse_bit_value(v)))
    return result


def parse_csv_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_seq_lens(args: argparse.Namespace) -> list[int]:
    if args.seq_len is not None:
        return [args.seq_len]
    return [int(item) for item in parse_csv_list(args.seq_lens)]


def is_integer_or_half_bit(bits: int | float) -> bool:
    return bits == 1.5 or bits == round(bits)


def methods_for_summary(final_rows: list[dict]) -> list[str]:
    present = {r["method"] for r in final_rows}
    ordered = [m for m in SUMMARY_METHOD_ORDER if m in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


# ---------------------------------------------------------------------------
# Calibration (same reorder format as SKVQ)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_minmax_stats(model, chunks: list[torch.Tensor]) -> dict:
    num_layers = len(model.model.layers)
    stats = {
        "min": {"k": [None] * num_layers, "v": [None] * num_layers},
        "max": {"k": [None] * num_layers, "v": [None] * num_layers},
    }

    def hook(_module, _inputs, output, ttype: str, layer_idx: int):
        flat = output.detach().reshape(-1, output.shape[-1]).float()
        cur_min = flat.amin(dim=0).cpu()
        cur_max = flat.amax(dim=0).cpu()
        if stats["min"][ttype][layer_idx] is None:
            stats["min"][ttype][layer_idx] = cur_min
            stats["max"][ttype][layer_idx] = cur_max
        else:
            stats["min"][ttype][layer_idx] = torch.minimum(stats["min"][ttype][layer_idx], cur_min)
            stats["max"][ttype][layer_idx] = torch.maximum(stats["max"][ttype][layer_idx], cur_max)

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(
            layer.self_attn.k_proj.register_forward_hook(
                lambda m, x, y, i=layer_idx: hook(m, x, y, "k", i)
            )
        )
        hooks.append(
            layer.self_attn.v_proj.register_forward_hook(
                lambda m, x, y, i=layer_idx: hook(m, x, y, "v", i)
            )
        )

    device = next(model.parameters()).device
    for chunk in chunks:
        model.model(chunk.to(device), use_cache=False)

    for handle in hooks:
        handle.remove()
    return stats


def build_reorder_cache(stats: dict, n_clusters: int, save_path: Path) -> None:
    reorder_indices = []
    cluster_st_inds = []
    num_layers = len(stats["min"]["k"])
    for layer_idx in range(num_layers):
        layer_reorders = []
        layer_starts = []
        for ttype in ("k", "v"):
            features = torch.stack(
                (stats["min"][ttype][layer_idx], stats["max"][ttype][layer_idx]), dim=1
            )
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(features.numpy())
            labels = torch.from_numpy(kmeans.labels_)
            indices = labels.argsort()
            starts = torch.zeros(n_clusters + 1, dtype=torch.int64)
            starts[1:] = labels.bincount(minlength=n_clusters).cumsum(0).to(torch.int64)
            layer_reorders.append(indices)
            layer_starts.append(starts)
        reorder_indices.append(tuple(layer_reorders))
        cluster_st_inds.append(tuple(layer_starts))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reorder_indices": reorder_indices, "cluster_st_inds": cluster_st_inds}, save_path)


def load_wikitext_token_ids(tokenizer, split: str) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row for row in ds["text"] if row.strip())
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids


def make_chunks_from_ids(
    ids: torch.Tensor, seq_len: int, nsamples: int, *, repeat: bool = False
) -> list[torch.Tensor]:
    needed = nsamples * seq_len
    if ids.shape[1] < needed:
        if not repeat:
            raise ValueError(f"need {needed} tokens, got {ids.shape[1]}")
        ids = ids.repeat(1, needed // ids.shape[1] + 2)
    return [
        ids[:, i * seq_len : (i + 1) * seq_len].contiguous()
        for i in range(nsamples)
    ]


def ensure_reorder(model, tokenizer, args: argparse.Namespace, seq_len: int, reorder_path: Path) -> None:
    if reorder_path.exists() and not args.force_recalib:
        print(f"[calib] reuse {reorder_path}")
        return

    print(f"[calib] loading WikiText-2 train for seq_len={seq_len}...")
    ids = load_wikitext_token_ids(tokenizer, "train")
    chunks = make_chunks_from_ids(ids, seq_len, args.calib_samples, repeat=True)

    print(f"[calib] collecting min/max from {args.calib_samples} chunks...")
    stats = collect_minmax_stats(model, chunks)

    cfg = model.config
    kv_hidden = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    n_clusters = kv_hidden // args.group_size
    if n_clusters <= 0:
        raise ValueError(f"group_size={args.group_size} > kv_hidden={kv_hidden}")

    print(f"[calib] KMeans reorder: kv_hidden={kv_hidden}, n_clusters={n_clusters}")
    build_reorder_cache(stats, n_clusters, reorder_path)
    print(f"[calib] saved {reorder_path}")


def load_eval_chunks(
    tokenizer, args: argparse.Namespace, dataset: str, seq_len: int
) -> tuple[list[torch.Tensor], int]:
    if dataset != "wikitext2":
        raise ValueError(f"unsupported dataset: {dataset}")
    ids = load_wikitext_token_ids(tokenizer, "test")
    max_eval = ids.shape[1] // seq_len
    n_eval = args.eval_samples if args.eval_samples > 0 else max_eval
    n_eval = min(n_eval, max_eval)
    if n_eval <= 0:
        raise ValueError(f"WikiText-2 test has no full chunks for seq_len={seq_len}")
    return make_chunks_from_ids(ids, seq_len, n_eval), n_eval


# ---------------------------------------------------------------------------
# BlockKVCache construction
# ---------------------------------------------------------------------------


def mixed_precision_bits(
    kbits: int | float, vbits: int | float
) -> tuple[float, float, float, float]:
    """High/low K/V bit pairs for page mixed precision (plan §6.2)."""
    return 4.0, 2.0, float(kbits), float(vbits)


def build_block_cache(
    args: argparse.Namespace,
    method: str,
    kbits: int | float,
    vbits: int | float,
    reorder_path: Path | None,
) -> BlockKVCache:
    policy = HybridPolicy(sink_size=args.sink, window_size=args.window_size)
    common = dict(
        block_size=args.block_size,
        policy=policy,
        group_size=args.group_size,
        clipping=args.clip,
        granularity="per-vector",
    )

    if method == "tq_block":
        if not (is_integer_or_half_bit(kbits) and is_integer_or_half_bit(vbits)):
            raise ValueError("tq_block requires integer or 1.5-bit widths")
        if kbits == 1.5 or vbits == 1.5:
            raise ValueError("TurboQuant block backend does not support 1.5-bit")
        cfg = BlockCacheConfig(
            **common,
            quant_backend="turboquant",
            mixed_precision=False,
            key_bits=int(kbits),
            value_bits=int(vbits),
        )
        return BlockKVCache(cfg)

    if method == "tq_block_mixed":
        if kbits == 1.5 or vbits == 1.5:
            raise ValueError("TurboQuant block mixed precision requires integer low bit-widths")
        hk, hv, lk, lv = mixed_precision_bits(kbits, vbits)
        if lk != round(lk) or lv != round(lv):
            raise ValueError("TurboQuant block mixed precision requires integer low bit-widths")
        cfg = BlockCacheConfig(
            **common,
            quant_backend="turboquant",
            mixed_precision=True,
            importance_metric=args.importance_metric,
            important_ratio=args.important_ratio,
            high_key_bits=int(hk),
            high_value_bits=int(hv),
            low_key_bits=int(lk),
            low_value_bits=int(lv),
        )
        return BlockKVCache(cfg)

    if method == "skvq_page_fixed":
        cfg = BlockCacheConfig(
            **common,
            quant_backend="skvq",
            mixed_precision=False,
            key_bits=float(kbits),
            value_bits=float(vbits),
            reorder_file=str(reorder_path) if reorder_path else None,
        )
        return BlockKVCache(cfg)

    if method == "skvq_page_mixed":
        hk, hv, lk, lv = mixed_precision_bits(kbits, vbits)
        cfg = BlockCacheConfig(
            **common,
            quant_backend="skvq",
            mixed_precision=True,
            importance_metric=args.importance_metric,
            important_ratio=args.important_ratio,
            high_key_bits=hk,
            high_value_bits=hv,
            low_key_bits=lk,
            low_value_bits=lv,
            reorder_file=str(reorder_path) if reorder_path else None,
        )
        return BlockKVCache(cfg)

    raise ValueError(f"unknown method: {method}")


def method_needs_reorder(method: str) -> bool:
    return method.startswith("skvq_")


def method_supports_bits(method: str, kbits: int | float, vbits: int | float) -> bool | str:
    if method in ("tq_block", "tq_block_mixed") and (kbits == 1.5 or vbits == 1.5):
        return "TurboQuant block backend does not support 1.5-bit"
    if method.startswith("skvq_") or method in ("tq_block", "tq_block_mixed"):
        if is_integer_or_half_bit(kbits) and is_integer_or_half_bit(vbits):
            return True
        return "unsupported bit width"
    return True


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_ppl(
    model,
    chunks: list[torch.Tensor],
    cache: BlockKVCache | None = None,
) -> tuple[float, dict | None]:
    loss_fct = nn.CrossEntropyLoss()
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    last_report = None

    for chunk in chunks:
        batch = chunk.to(device)
        if cache is None:
            outputs = model(batch, use_cache=True)
        else:
            outputs = model(batch, past_key_values=cache, use_cache=True)
            cache.reset()
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous().to(shift_logits.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        n_tokens = shift_labels.numel()
        total_nll += loss.float().item() * n_tokens
        total_tokens += n_tokens
        if cache is not None:
            last_report = cache.memory_report()

    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item())
    return ppl, last_report


# ---------------------------------------------------------------------------
# Result IO (same schema as SKVQ)
# ---------------------------------------------------------------------------


def combo_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        row["dataset"],
        str(row["seq_len"]),
        row["method"],
        row["kbits"],
        row["vbits"],
    )


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [{field: row.get(field, "") for field in CSV_FIELDS} for row in csv.DictReader(f)]


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def base_row(
    args: argparse.Namespace,
    dataset: str,
    seq_len: int,
    method: str,
    kbits: int | float,
    vbits: int | float,
    n_eval: int,
) -> dict:
    return {
        "dataset": dataset,
        "seq_len": seq_len,
        "method": method,
        "kbits": bit_tag(kbits),
        "vbits": bit_tag(vbits),
        "status": "ok",
        "ppl": "",
        "elapsed_sec": "",
        "peak_allocated_gb": "",
        "n_eval": n_eval,
        "calib_dataset": args.calib_dataset,
        "calib_samples": args.calib_samples,
        "group_size": args.group_size,
        "window_size": args.window_size,
        "sink": args.sink,
        "clip": args.clip,
        "protect_layers": "",
        "protected_bits": "",
        "error": "",
    }


def write_summary(
    results_dir: Path,
    rows: list[dict],
    methods: list[str],
    bits_combos: list[tuple[int | float, int | float]],
) -> None:
    for dataset in sorted({row["dataset"] for row in rows}):
        for seq_len in sorted({int(row["seq_len"]) for row in rows if row["dataset"] == dataset}):
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and int(row["seq_len"]) == seq_len
            ]
            out_path = results_dir / f"summary_{dataset}_len{seq_len}.csv"
            fieldnames = ["method"] + [
                f"k{bit_tag(k)}-v{bit_tag(v)}_ppl" for k, v in bits_combos
            ]
            fieldnames += [
                f"k{bit_tag(k)}-v{bit_tag(v)}_status" for k, v in bits_combos
            ]
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for method in methods:
                    out = {"method": method}
                    for kbits, vbits in bits_combos:
                        match = next(
                            (
                                row
                                for row in subset
                                if row["method"] == method
                                and row["kbits"] == bit_tag(kbits)
                                and row["vbits"] == bit_tag(vbits)
                            ),
                            None,
                        )
                        prefix = f"k{bit_tag(kbits)}-v{bit_tag(vbits)}"
                        out[f"{prefix}_ppl"] = match["ppl"] if match else ""
                        out[f"{prefix}_status"] = match["status"] if match else "missing"
                    writer.writerow(out)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bits_combos = parse_bits(args.bits)
    methods = parse_csv_list(args.methods)
    datasets = parse_csv_list(args.datasets)
    seq_lens = parse_seq_lens(args)
    csv_path = results_dir / "exp1_ppl_ablation.csv"

    if not args.resume and csv_path.exists():
        csv_path.unlink()

    existing_rows = read_existing_rows(csv_path) if args.resume else []
    completed = {
        combo_key(row) for row in existing_rows if row.get("status") in {"ok", "error", "unsupported"}
    }
    if args.resume:
        print(f"[resume] keeping {len(existing_rows)} existing rows")

    print(f"[load] tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    print(f"[load] Llama from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    print(
        f"[load] layers={len(model.model.layers)}, "
        f"kv_heads={model.config.num_key_value_heads}, "
        f"head_dim={model.config.hidden_size // model.config.num_attention_heads}"
    )

    new_rows: list[dict] = []
    fp16_cache: dict[tuple[str, int], dict] = {}

    for seq_len in seq_lens:
        reorder_path = results_dir / (
            f"reorder-{args.calib_dataset}-g{args.group_size}-n{args.calib_samples}-len{seq_len}-minmax.pt"
        )
        if any(method_needs_reorder(m) for m in methods):
            ensure_reorder(model, tokenizer, args, seq_len, reorder_path)

        for dataset in datasets:
            print(f"[data] {dataset} eval chunks, seq_len={seq_len}...")
            try:
                eval_chunks, n_eval = load_eval_chunks(tokenizer, args, dataset, seq_len)
            except Exception as exc:
                print(f"[data-error] {exc}")
                for kbits, vbits in bits_combos:
                    for method in methods:
                        row = base_row(args, dataset, seq_len, method, kbits, vbits, 0)
                        row["status"] = "error"
                        row["error"] = str(exc)[:500]
                        if combo_key(row) not in completed:
                            append_row(csv_path, row)
                            new_rows.append(row)
                            completed.add(combo_key(row))
                continue

            print(f"[data] n_eval={n_eval} chunks of length {seq_len}")

            for kbits, vbits in bits_combos:
                for method in methods:
                    row = base_row(args, dataset, seq_len, method, kbits, vbits, n_eval)
                    key = combo_key(row)
                    if key in completed:
                        print(f"[skip] {dataset} len{seq_len} {method} k{row['kbits']}-v{row['vbits']}")
                        continue

                    support = method_supports_bits(method, kbits, vbits)
                    if support is not True:
                        row["status"] = "unsupported"
                        row["error"] = str(support)
                        append_row(csv_path, row)
                        new_rows.append(row)
                        completed.add(key)
                        continue

                    if method == "fp16" and (dataset, seq_len) in fp16_cache:
                        cached = fp16_cache[(dataset, seq_len)]
                        row.update(
                            {
                                "status": cached["status"],
                                "ppl": cached["ppl"],
                                "elapsed_sec": "0.00",
                                "peak_allocated_gb": cached["peak_allocated_gb"],
                                "error": cached["error"],
                            }
                        )
                        append_row(csv_path, row)
                        new_rows.append(row)
                        completed.add(key)
                        continue

                    print("\n" + "=" * 60)
                    print(
                        f"[eval] {dataset} len{seq_len} {method} "
                        f"k{row['kbits']}-v{row['vbits']} n_eval={n_eval}"
                    )
                    print("=" * 60)

                    cache = None
                    start = time.perf_counter()
                    try:
                        if method != "fp16":
                            cache = build_block_cache(
                                args, method, kbits, vbits, reorder_path
                            )
                            print(
                                f"[cache] backend={cache.config.quant_backend} "
                                f"mixed_precision={cache.config.mixed_precision}"
                            )

                        torch.cuda.empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                        ppl, report = eval_ppl(model, eval_chunks, cache)
                        elapsed = time.perf_counter() - start
                        peak_gb = (
                            torch.cuda.max_memory_allocated() / 1024**3
                            if torch.cuda.is_available()
                            else 0.0
                        )
                        row["ppl"] = f"{ppl:.6f}"
                        row["elapsed_sec"] = f"{elapsed:.2f}"
                        row["peak_allocated_gb"] = f"{peak_gb:.3f}"
                        extra = ""
                        if report:
                            extra = (
                                f" ratio={report['compression_ratio']:.2f}x"
                                f" compressed={report['n_compressed_blocks']}"
                            )
                        print(f"[result] ppl={ppl:.6f} time={elapsed:.1f}s peak={peak_gb:.2f}GB{extra}")
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        row["status"] = "error"
                        row["elapsed_sec"] = f"{elapsed:.2f}"
                        row["peak_allocated_gb"] = (
                            f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}"
                            if torch.cuda.is_available()
                            else "0.000"
                        )
                        row["error"] = str(exc).splitlines()[0][:500]
                        print(f"[error] {row['error']}")
                    finally:
                        if cache is not None:
                            cache.reset()
                        gc.collect()
                        torch.cuda.empty_cache()

                    if method == "fp16":
                        fp16_cache[(dataset, seq_len)] = dict(row)

                    append_row(csv_path, row)
                    new_rows.append(row)
                    completed.add(key)

    all_rows = existing_rows + new_rows
    if all_rows:
        deduped = {combo_key(row): row for row in all_rows}
        final_rows = sorted(
            deduped.values(),
            key=lambda r: (r["dataset"], int(r["seq_len"]), r["method"], r["kbits"], r["vbits"]),
        )
        write_rows(csv_path, final_rows)
        write_summary(results_dir, final_rows, methods_for_summary(final_rows), bits_combos)

    print("\n" + "=" * 60)
    print(f"RESULTS ({csv_path})")
    print("=" * 60)
    for row in new_rows:
        print(
            f"{row['dataset']:<10} {row['seq_len']:<6} {row['method']:<18} "
            f"k{row['kbits']}-v{row['vbits']:<6} {row['status']:<12} {row['ppl']}"
        )


if __name__ == "__main__":
    main()
