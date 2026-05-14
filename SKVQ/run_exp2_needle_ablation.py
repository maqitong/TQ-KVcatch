"""
Experiment 2: long-context Needle-in-a-Haystack ablation.

Runs retrieval accuracy over a context-length x depth grid for:
  fp16, KIVI, skvq_baseline, tq_replace, tq_hybrid, tq_asym_protect

The default bit-width is k2-v2. Results are written incrementally so a long
4090 run can be resumed after OOM or interruption.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from experiments.utils import plug_quantizer_into_model
from run_exp1_ppl_ablation import (
    MODEL_CLASSES,
    bit_tag,
    build_manager,
    detach_quantizer,
    ensure_reorder,
    parse_bit_value,
    parse_csv_list,
)


DEFAULT_METHODS = "fp16,KIVI,skvq_baseline,tq_replace,tq_hybrid,tq_asym_protect"
DEFAULT_CONTEXTS = "4096,8192,16384,32768"
DEFAULT_DEPTHS = "10,25,50,75,90"

DETAIL_FIELDS = [
    "method",
    "kbits",
    "vbits",
    "context_length",
    "depth_percent",
    "status",
    "hit",
    "elapsed_sec",
    "new_tokens",
    "prompt_tokens",
    "peak_allocated_gb",
    "expected",
    "prediction",
    "error",
    "group_size",
    "window_size",
    "sink",
    "clip",
    "protect_layers",
    "protected_bits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp2: Needle-in-a-Haystack long-context ablation")
    parser.add_argument("--model-path", required=True, help="HuggingFace model path or local dir")
    parser.add_argument("--model-family", required=True, choices=["llama", "mistral"])
    parser.add_argument("--results-dir", default="experiments/results/exp2_needle")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS, help="Comma-separated context token lengths")
    parser.add_argument("--depths", default=DEFAULT_DEPTHS, help="Comma-separated needle depths in percent")
    parser.add_argument("--bits", default="2+2", help="Single k+v bit pair, default k2-v2")
    parser.add_argument("--calib-seq-len", type=int, default=4096)
    parser.add_argument("--calib-samples", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.96)
    parser.add_argument("--protect-layers", type=int, default=4)
    parser.add_argument("--protected-bits", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-recalib", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-heatmap", action="store_true")
    return parser.parse_args()


def parse_bit_pair(spec: str) -> tuple[int | float, int | float]:
    parts = [item.strip() for item in spec.split("+") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"--bits must look like 2+2, got {spec!r}")
    return parse_bit_value(parts[0]), parse_bit_value(parts[1])


def parse_int_list(spec: str) -> list[int]:
    return [int(item) for item in parse_csv_list(spec)]


def parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in parse_csv_list(spec)]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def result_key(row: dict) -> tuple[str, str, str]:
    return (row["method"], str(row["context_length"]), str(row["depth_percent"]))


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{field: row.get(field, "") for field in DETAIL_FIELDS} for row in reader]


def append_detail(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_needle_prompt(tokenizer, context_tokens: int, depth_percent: float) -> tuple[str, str]:
    expected = "AURORA-7749"
    needle = f" The secret retrieval code is {expected}. Remember this exact code. "
    question = "\n\nQuestion: What is the secret retrieval code? Answer with only the code.\nAnswer:"
    filler = (
        "This archival report discusses municipal planning notes, bridge inspections, "
        "budget meetings, weather observations, software migration logs, and routine maintenance updates. "
    )

    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    available = max(context_tokens - len(needle_ids) - len(question_ids), 64)
    repeated = (filler_ids * (available // len(filler_ids) + 2))[:available]
    insert_at = int(len(repeated) * depth_percent / 100.0)
    context_ids = repeated[:insert_at] + needle_ids + repeated[insert_at:]
    context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return f"{context}{question}", expected


def clear_method(model) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


def attach_method(model, args: argparse.Namespace, method: str, kbits: int | float, vbits: int | float, reorder_path: Path):
    detach_quantizer(model)
    if method == "fp16":
        return None
    method_reorder = None if method == "tq_replace" else reorder_path
    manager = build_manager(model, args, method, kbits, vbits, method_reorder)
    plug_quantizer_into_model(model, manager)
    return manager


@torch.no_grad()
def generate_once(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float, int, int]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[-1])
    start = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    elapsed = time.perf_counter() - start
    new_tokens = int(output.shape[-1] - prompt_tokens)
    pred = tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True).strip()
    clear_method(model)
    return pred, elapsed, new_tokens, prompt_tokens


def make_row(
    args: argparse.Namespace,
    method: str,
    kbits: int | float,
    vbits: int | float,
    context_length: int,
    depth_percent: float,
    expected: str,
) -> dict:
    return {
        "method": method,
        "kbits": bit_tag(kbits),
        "vbits": bit_tag(vbits),
        "context_length": context_length,
        "depth_percent": depth_percent,
        "status": "ok",
        "hit": "",
        "elapsed_sec": "",
        "new_tokens": "",
        "prompt_tokens": "",
        "peak_allocated_gb": "",
        "expected": expected,
        "prediction": "",
        "error": "",
        "group_size": args.group_size,
        "window_size": args.window_size,
        "sink": args.sink,
        "clip": args.clip,
        "protect_layers": args.protect_layers if method == "tq_asym_protect" else "",
        "protected_bits": args.protected_bits if method == "tq_asym_protect" else "",
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], str(row["context_length"]), str(row["depth_percent"])), []).append(row)

    summary = []
    for (method, context_length, depth_percent), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], int(item[0][1]), float(item[0][2]))
    ):
        ok_rows = [row for row in group if row["status"] == "ok"]
        hits = sum(int(row["hit"]) for row in ok_rows if str(row["hit"]).strip() != "")
        total = len(ok_rows)
        summary.append(
            {
                "method": method,
                "context_length": context_length,
                "depth_percent": depth_percent,
                "ok": total,
                "errors": len(group) - total,
                "hits": hits,
                "hit_rate": f"{hits / total:.4f}" if total else "",
            }
        )
    return summary


def write_heatmaps(rows: list[dict], results_dir: Path, contexts: list[int], depths: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[heatmap] skipped: {exc}")
        return

    methods = sorted({row["method"] for row in rows})
    for method in methods:
        matrix = []
        for ctx in contexts:
            line = []
            for depth in depths:
                matches = [
                    row for row in rows
                    if row["method"] == method
                    and int(row["context_length"]) == ctx
                    and float(row["depth_percent"]) == depth
                    and row["status"] == "ok"
                ]
                if not matches:
                    line.append(float("nan"))
                else:
                    line.append(sum(int(row["hit"]) for row in matches) / len(matches))
            matrix.append(line)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(depths)), [f"{depth:g}" for depth in depths])
        ax.set_yticks(range(len(contexts)), [str(ctx) for ctx in contexts])
        ax.set_xlabel("Needle depth (%)")
        ax.set_ylabel("Context length")
        ax.set_title(method)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                label = "NA" if value != value else f"{value:.0%}"
                ax.text(x, y, label, ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="Hit rate")
        fig.tight_layout()
        safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
        fig.savefig(results_dir / f"heatmap_{safe_method}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    detail_path = results_dir / "needle_details.csv"
    summary_path = results_dir / "needle_summary.csv"

    methods = parse_csv_list(args.methods)
    contexts = parse_int_list(args.contexts)
    depths = parse_float_list(args.depths)
    kbits, vbits = parse_bit_pair(args.bits)
    model_cls = MODEL_CLASSES[args.model_family]

    if not args.resume and detail_path.exists():
        detail_path.unlink()

    existing_rows = read_existing(detail_path) if args.resume else []
    completed = {result_key(row) for row in existing_rows if row.get("status") in {"ok", "oom", "error"}}
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
        use_flash_attention_2=args.flash_attn and is_flash_attn_2_available(),
    ).eval()
    print(f"[load] device_map={getattr(model, 'hf_device_map', None)}")

    reorder_path = results_dir / (
        f"reorder-wikitext2-g{args.group_size}-n{args.calib_samples}-len{args.calib_seq_len}-minmax.pt"
    )
    ensure_reorder(model, tokenizer, args, args.calib_seq_len, reorder_path)

    new_rows: list[dict] = []
    prompts: dict[tuple[int, float], tuple[str, str]] = {}

    for method in methods:
        print(f"[method] {method}")
        try:
            manager = attach_method(model, args, method, kbits, vbits, reorder_path)
            if manager is not None:
                print(manager)
        except Exception as exc:
            print(f"[method-error] {method}: {exc}")
            for ctx in contexts:
                for depth in depths:
                    prompt, expected = prompts.setdefault((ctx, depth), make_needle_prompt(tokenizer, ctx, depth))
                    row = make_row(args, method, kbits, vbits, ctx, depth, expected)
                    row["status"] = "error"
                    row["error"] = str(exc).splitlines()[0][:500]
                    if result_key(row) not in completed:
                        append_detail(detail_path, row)
                        new_rows.append(row)
                        completed.add(result_key(row))
            continue

        for ctx in contexts:
            for depth in depths:
                prompt, expected = prompts.setdefault((ctx, depth), make_needle_prompt(tokenizer, ctx, depth))
                row = make_row(args, method, kbits, vbits, ctx, depth, expected)
                key = result_key(row)
                if key in completed:
                    print(f"[skip] {method} ctx={ctx} depth={depth:g}")
                    continue

                print(f"[eval] {method} ctx={ctx} depth={depth:g}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    pred, elapsed, new_tokens, prompt_tokens = generate_once(
                        model, tokenizer, prompt, args.max_new_tokens
                    )
                    row["hit"] = int(norm_text(expected) in norm_text(pred))
                    row["elapsed_sec"] = f"{elapsed:.2f}"
                    row["new_tokens"] = new_tokens
                    row["prompt_tokens"] = prompt_tokens
                    row["peak_allocated_gb"] = (
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}" if torch.cuda.is_available() else "0.000"
                    )
                    row["prediction"] = pred.replace("\n", "\\n")[:500]
                    print(f"[result] hit={row['hit']} pred={row['prediction'][:80]}")
                except torch.cuda.OutOfMemoryError as exc:
                    row["status"] = "oom"
                    row["elapsed_sec"] = f"{time.perf_counter() - start:.2f}"
                    row["peak_allocated_gb"] = (
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}" if torch.cuda.is_available() else "0.000"
                    )
                    row["error"] = str(exc).splitlines()[0][:500]
                    clear_method(model)
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"[oom] {method} ctx={ctx} depth={depth:g}: {row['error']}")
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "out of memory" in message or "cuda error" in message:
                        row["status"] = "oom"
                    else:
                        row["status"] = "error"
                    row["elapsed_sec"] = f"{time.perf_counter() - start:.2f}"
                    row["peak_allocated_gb"] = (
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}" if torch.cuda.is_available() else "0.000"
                    )
                    row["error"] = str(exc).splitlines()[0][:500]
                    clear_method(model)
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"[{row['status']}] {method} ctx={ctx} depth={depth:g}: {row['error']}")
                append_detail(detail_path, row)
                new_rows.append(row)
                completed.add(key)

        detach_quantizer(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_rows = existing_rows + new_rows
    if all_rows:
        deduped: dict[tuple[str, str, str], dict] = {}
        for row in all_rows:
            deduped[result_key(row)] = row
        final_rows = list(deduped.values())
        final_rows.sort(key=lambda row: (row["method"], int(row["context_length"]), float(row["depth_percent"])))
        write_csv(detail_path, final_rows, DETAIL_FIELDS)
        summary_rows = summarize(final_rows)
        write_csv(summary_path, summary_rows, ["method", "context_length", "depth_percent", "ok", "errors", "hits", "hit_rate"])
        if not args.no_heatmap:
            write_heatmaps(final_rows, results_dir, contexts, depths)

    print(f"[done] details: {detail_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
