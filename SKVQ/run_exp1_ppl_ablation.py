"""
Experiment 1: Extremely Low-Bit PPL Ablation (SKVQ + TurboQuant)

Runs the core PPL matrix for:
  - methods: fp16, KIVI, skvq_baseline, tq_replace, tq_hybrid
  - bits: k2-v2, k2-v1.5, k1.5-v1.5
  - datasets: WikiText-2 and C4
  - sequence lengths: 2048 and 4096

The script writes a tall CSV for all runs plus one pivot-style summary CSV
per dataset/sequence length.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.modeling_mistral_skvq import MistralForCausalLM
from experiments.utils import plug_quantizer_into_model
from KVcache_manager import ModelKVCacheManager


MODEL_CLASSES = {
    "llama": LlamaForCausalLM,
    "mistral": MistralForCausalLM,
}

DEFAULT_METHODS = "fp16,KIVI,skvq_baseline,tq_replace,tq_hybrid"
DEFAULT_BITS = "2+2,2+1.5,1.5+1.5"
DEFAULT_DATASETS = "wikitext2,c4"
DEFAULT_SEQ_LENS = "2048,4096"

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
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: extremely low-bit PPL ablation")
    p.add_argument("--model-path", required=True, help="HuggingFace model path or local dir")
    p.add_argument("--model-family", required=True, choices=["llama", "mistral"])
    p.add_argument("--results-dir", default="experiments/results/exp1")
    p.add_argument("--seq-len", type=int, default=None, help="Single sequence length; overrides --seq-lens")
    p.add_argument("--seq-lens", default=DEFAULT_SEQ_LENS, help="Comma-separated sequence lengths")
    p.add_argument("--calib-samples", type=int, default=256, help="Calibration samples from WikiText-2 train")
    p.add_argument("--eval-samples", type=int, default=0, help="Eval chunks per dataset (0=all WikiText-2, --c4-eval-samples for C4)")
    p.add_argument("--c4-eval-samples", type=int, default=128, help="C4 eval chunks when --eval-samples=0")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--sink", type=int, default=5)
    p.add_argument("--clip", type=float, default=0.96)
    p.add_argument("--bits", default=DEFAULT_BITS, help="Comma-separated k+v bit pairs")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated: wikitext2,c4")
    p.add_argument("--calib-dataset", default="wikitext2", choices=["wikitext2"])
    p.add_argument("--c4-name", default="allenai/c4")
    p.add_argument("--c4-config", default="en")
    p.add_argument("--c4-split", default="validation")
    p.add_argument("--c4-streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-recalib", action="store_true")
    p.add_argument("--resume", action="store_true", help="Keep existing rows and skip completed combos")
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


# ---------------------------------------------------------------------------
# Calibration
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
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            lambda m, x, y, i=layer_idx: hook(m, x, y, "k", i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            lambda m, x, y, i=layer_idx: hook(m, x, y, "v", i)))

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


def make_chunks_from_ids(ids: torch.Tensor, seq_len: int, nsamples: int, *, repeat: bool = False) -> list[torch.Tensor]:
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

    print(f"[calib] loading WikiText-2 train split for seq_len={seq_len}...")
    ids = load_wikitext_token_ids(tokenizer, "train")
    chunks = make_chunks_from_ids(ids, seq_len, args.calib_samples, repeat=True)

    print(f"[calib] collecting min/max stats from {args.calib_samples} chunks...")
    stats = collect_minmax_stats(model, chunks)

    cfg = model.config
    kv_hidden = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    n_clusters = kv_hidden // args.group_size
    if n_clusters <= 0:
        raise ValueError(f"group_size={args.group_size} is larger than kv_hidden={kv_hidden}")

    print(f"[calib] building reorder cache: kv_hidden={kv_hidden}, n_clusters={n_clusters}")
    build_reorder_cache(stats, n_clusters, reorder_path)
    print(f"[calib] saved {reorder_path}")


# ---------------------------------------------------------------------------
# Eval data
# ---------------------------------------------------------------------------

def c4_text_iter(args: argparse.Namespace) -> Iterable[str]:
    try:
        ds = load_dataset(args.c4_name, args.c4_config, split=args.c4_split, streaming=args.c4_streaming)
    except Exception:
        ds = load_dataset("c4", args.c4_config, split=args.c4_split, streaming=args.c4_streaming)
    for row in ds:
        text = row.get("text", "") if isinstance(row, dict) else ""
        if text.strip():
            yield text


def load_c4_chunks(tokenizer, args: argparse.Namespace, seq_len: int, nsamples: int) -> list[torch.Tensor]:
    needed = nsamples * seq_len
    token_parts: list[torch.Tensor] = []
    total = 0
    for text in c4_text_iter(args):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.numel() == 0:
            continue
        token_parts.append(ids)
        total += ids.shape[1]
        if total >= needed:
            break
    if total < needed:
        raise ValueError(f"C4 {args.c4_split}: need {needed} tokens, got {total}")
    ids = torch.cat(token_parts, dim=1)[:, :needed]
    return make_chunks_from_ids(ids, seq_len, nsamples)


def load_eval_chunks(tokenizer, args: argparse.Namespace, dataset: str, seq_len: int) -> tuple[list[torch.Tensor], int]:
    if dataset == "wikitext2":
        ids = load_wikitext_token_ids(tokenizer, "test")
        max_eval = ids.shape[1] // seq_len
        n_eval = args.eval_samples if args.eval_samples > 0 else max_eval
        n_eval = min(n_eval, max_eval)
        if n_eval <= 0:
            raise ValueError(f"WikiText-2 test has no full chunks for seq_len={seq_len}")
        return make_chunks_from_ids(ids, seq_len, n_eval), n_eval

    if dataset == "c4":
        n_eval = args.eval_samples if args.eval_samples > 0 else args.c4_eval_samples
        if n_eval <= 0:
            raise ValueError("C4 requires --eval-samples or --c4-eval-samples > 0")
        return load_c4_chunks(tokenizer, args, seq_len, n_eval), n_eval

    raise ValueError(f"unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Quantizer construction
# ---------------------------------------------------------------------------

def detach_quantizer(model) -> None:
    for layer in model.model.layers:
        layer.self_attn.KV_cache_manager = None
    model.model.model_kv_manager = None
    model.model_kv_manager = None


def clear_quantizer(model) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


def build_manager(
    model,
    args: argparse.Namespace,
    method: str,
    kbits: int | float,
    vbits: int | float,
    reorder_path: Path | None,
) -> ModelKVCacheManager:
    num_layers = len(model.model.layers)
    clipping = [args.clip for _ in range(num_layers)]

    if method == "KIVI":
        if args.window_size < args.group_size or args.window_size % args.group_size != 0:
            raise ValueError("KIVI requires window_size >= group_size and window_size % group_size == 0")
        return ModelKVCacheManager.create(
            model=model,
            kbits=kbits,
            vbits=vbits,
            gsize=args.group_size,
            reorder_file=None,
            smooth_file=None,
            window_size=args.window_size,
            pre_rope=False,
            clipping=clipping,
            attn_sink=0,
            full_prefill=True,
            KIVI_mode=True,
            fp8=True,
            fake_quant=True,
        )

    common = dict(
        model=model,
        kbits=kbits,
        vbits=vbits,
        gsize=args.group_size,
        window_size=args.window_size,
        pre_rope=True,
        clipping=clipping,
        attn_sink=args.sink,
        full_prefill=False,
        fp8=True,
        fake_quant=True,
    )
    if method == "skvq_baseline":
        return ModelKVCacheManager.create(reorder_file=str(reorder_path), **common)
    if method == "tq_replace":
        return ModelKVCacheManager.create(
            reorder_file=None,
            turboquant_config={"use_reorder": False, "protected_layers": 0, "seed_base": 42},
            **common,
        )
    if method == "tq_hybrid":
        return ModelKVCacheManager.create(
            reorder_file=str(reorder_path),
            turboquant_config={"use_reorder": True, "protected_layers": 0, "seed_base": 42},
            **common,
        )
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, chunks: list[torch.Tensor]) -> float:
    loss_fct = nn.CrossEntropyLoss()
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for chunk in chunks:
        batch = chunk.to(device)
        outputs = model(batch, use_cache=True)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous().to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_tokens = shift_labels.numel()
        total_nll += loss.float().item() * n_tokens
        total_tokens += n_tokens
        clear_quantizer(model)

    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------

def combo_key(row: dict) -> tuple[str, str, str, str, str]:
    return (row["dataset"], str(row["seq_len"]), row["method"], row["kbits"], row["vbits"])


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({field: row.get(field, "") for field in CSV_FIELDS})
        return rows


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


def base_row(args: argparse.Namespace, dataset: str, seq_len: int, method: str, kbits: int | float, vbits: int | float, n_eval: int) -> dict:
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
        "error": "",
    }


def make_error_row(
    args: argparse.Namespace,
    dataset: str,
    seq_len: int,
    method: str,
    kbits: int | float,
    vbits: int | float,
    n_eval: int,
    status: str,
    error: str,
) -> dict:
    row = base_row(args, dataset, seq_len, method, kbits, vbits, n_eval)
    row["status"] = status
    row["error"] = error.splitlines()[0][:500]
    return row


def write_summary(results_dir: Path, rows: list[dict], methods: list[str], bits_combos: list[tuple[int | float, int | float]]) -> None:
    for dataset in sorted({row["dataset"] for row in rows}):
        for seq_len in sorted({row["seq_len"] for row in rows if row["dataset"] == dataset}, key=lambda x: int(x)):
            subset = [row for row in rows if row["dataset"] == dataset and row["seq_len"] == seq_len]
            out_path = results_dir / f"summary_{dataset}_len{seq_len}.csv"
            fieldnames = ["method"] + [f"k{bit_tag(k)}-v{bit_tag(v)}_ppl" for k, v in bits_combos]
            fieldnames += [f"k{bit_tag(k)}-v{bit_tag(v)}_status" for k, v in bits_combos]
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for method in methods:
                    out = {"method": method}
                    for kbits, vbits in bits_combos:
                        match = next(
                            (
                                row for row in subset
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bits_combos = parse_bits(args.bits)
    methods = parse_csv_list(args.methods)
    datasets = parse_csv_list(args.datasets)
    seq_lens = parse_seq_lens(args)
    model_cls = MODEL_CLASSES[args.model_family]
    csv_path = results_dir / "exp1_ppl_ablation.csv"

    existing_rows = read_existing_rows(csv_path) if args.resume else []
    completed = {combo_key(row) for row in existing_rows if row.get("status") in {"ok", "error", "unsupported"}}
    if args.resume:
        print(f"[resume] keeping {len(existing_rows)} existing rows")

    print(f"[load] tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    print(f"[load] {model_cls.__name__} from {args.model_path}")
    model = model_cls.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"[load] device_map={getattr(model, 'hf_device_map', None)}")
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
        ensure_reorder(model, tokenizer, args, seq_len, reorder_path)

        for dataset in datasets:
            print(f"[data] loading {dataset} eval chunks for seq_len={seq_len}...")
            try:
                eval_chunks, n_eval = load_eval_chunks(tokenizer, args, dataset, seq_len)
            except Exception as exc:
                print(f"[data-error] {dataset} len{seq_len}: {exc}")
                for kbits, vbits in bits_combos:
                    for method in methods:
                        row = make_error_row(args, dataset, seq_len, method, kbits, vbits, 0, "error", str(exc))
                        if combo_key(row) not in completed:
                            append_row(csv_path, row)
                            new_rows.append(row)
                            completed.add(combo_key(row))
                continue

            for kbits, vbits in bits_combos:
                for method in methods:
                    row = base_row(args, dataset, seq_len, method, kbits, vbits, n_eval)
                    key = combo_key(row)
                    if key in completed:
                        print(f"[skip] {dataset} len{seq_len} {method} k{row['kbits']}-v{row['vbits']}")
                        continue

                    if method.startswith("tq") and not (is_integer_or_half_bit(kbits) and is_integer_or_half_bit(vbits)):
                        row["status"] = "unsupported"
                        row["error"] = "TurboQuant supports integer and 1.5-bit widths in this backend"
                        append_row(csv_path, row)
                        new_rows.append(row)
                        completed.add(key)
                        continue

                    if method == "fp16" and (dataset, seq_len) in fp16_cache:
                        cached = fp16_cache[(dataset, seq_len)]
                        row.update({
                            "status": cached["status"],
                            "ppl": cached["ppl"],
                            "elapsed_sec": "0.00",
                            "peak_allocated_gb": cached["peak_allocated_gb"],
                            "error": cached["error"],
                        })
                        append_row(csv_path, row)
                        new_rows.append(row)
                        completed.add(key)
                        continue

                    print("\n" + "=" * 60)
                    print(f"[eval] {dataset} len{seq_len} {method} k{row['kbits']}-v{row['vbits']} n_eval={n_eval}")
                    print("=" * 60)

                    detach_quantizer(model)
                    start = time.perf_counter()
                    try:
                        if method != "fp16":
                            manager = build_manager(model, args, method, kbits, vbits, reorder_path)
                            plug_quantizer_into_model(model, manager)
                            print(manager)

                        torch.cuda.empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                        ppl = eval_ppl(model, eval_chunks)
                        elapsed = time.perf_counter() - start
                        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                        row["ppl"] = f"{ppl:.6f}"
                        row["elapsed_sec"] = f"{elapsed:.2f}"
                        row["peak_allocated_gb"] = f"{peak_gb:.3f}"
                        print(f"[result] ppl={ppl:.6f} time={elapsed:.1f}s peak={peak_gb:.2f}GB")
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        row["status"] = "error"
                        row["elapsed_sec"] = f"{elapsed:.2f}"
                        row["peak_allocated_gb"] = (
                            f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}" if torch.cuda.is_available() else "0.000"
                        )
                        row["error"] = str(exc).splitlines()[0][:500]
                        print(f"[error] {row['error']}")
                    finally:
                        detach_quantizer(model)
                        gc.collect()
                        torch.cuda.empty_cache()

                    if method == "fp16":
                        fp16_cache[(dataset, seq_len)] = dict(row)

                    append_row(csv_path, row)
                    new_rows.append(row)
                    completed.add(key)

    all_rows = existing_rows + new_rows
    if all_rows:
        deduped: dict[tuple[str, str, str, str, str], dict] = {}
        for row in all_rows:
            deduped[combo_key(row)] = row
        final_rows = list(deduped.values())
        final_rows.sort(key=lambda r: (r["dataset"], int(r["seq_len"]), r["method"], r["kbits"], r["vbits"]))
        write_rows(csv_path, final_rows)
        write_summary(results_dir, final_rows, methods, bits_combos)

    print("\n" + "=" * 60)
    print(f"RESULTS ({csv_path})")
    print("=" * 60)
    print(f"{'dataset':<10} {'len':<6} {'method':<16} {'bits':<12} {'status':<12} {'PPL':<12}")
    print("-" * 72)
    for row in new_rows:
        bits = f"k{row['kbits']}-v{row['vbits']}"
        print(f"{row['dataset']:<10} {row['seq_len']:<6} {row['method']:<16} {bits:<12} {row['status']:<12} {row['ppl']:<12}")


if __name__ == "__main__":
    main()
