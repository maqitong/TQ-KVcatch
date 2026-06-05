"""Memory and latency profiler for KVcatch cache backends.

The profiler runs a compact NIAH generation case for one or more backends and
records wall time, CUDA peak memory when available, and `BlockKVCache`
compression statistics. It is meant for quick engineering checks before
running larger PPL/NIAH/LongBench experiments.

Example:
    python -m turboquant.block_cache.profile_memory \
        --model D:\model\Llama3.2_3B --local-files-only \
        --backend all --context-length 512 --position 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from turboquant.block_cache.eval_niah import (
    _cache_factory,
    _dtype_from_name,
    _parse_optional_int,
    _selected_backends,
    build_prompt,
)


@dataclass
class ProfileResult:
    backend: str
    model: str
    context_length: int
    position: float
    seed: int
    found: bool
    expected: str
    response: str
    input_tokens: int
    output_tokens: int
    seconds: float
    tokens_per_second: float
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    compression_ratio: float | None
    n_compressed_blocks: int | None
    n_fp16_blocks: int | None
    bit_histogram: dict | None
    precision_histogram: dict | None
    config: dict


def _parse_bits(value: str) -> float:
    bits = float(value)
    return int(bits) if bits == round(bits) else bits


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def profile_backend(model, tokenizer, args, backend: str) -> ProfileResult:
    prompt, expected = build_prompt(
        tokenizer, args.context_length, args.position, args.seed
    )
    device = _model_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.context_length + args.prompt_slack,
    ).to(device)
    if not args.pass_attention_mask:
        inputs.pop("attention_mask", None)

    cache = _cache_factory(args, backend)()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    output = model.generate(
        **inputs,
        past_key_values=cache,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        output_attentions=args.record_attentions,
        return_dict_in_generate=args.record_attentions,
    )
    seconds = time.perf_counter() - started

    if args.record_attentions:
        if cache is not None:
            cache.record_attentions(getattr(output, "attentions", None))
        sequences = output.sequences
    else:
        sequences = output

    new_tokens = sequences[0, inputs.input_ids.shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    found = expected.lower() in response.lower()
    report = cache.memory_report() if cache is not None else None

    peak_allocated = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    output_tokens = int(new_tokens.shape[0])
    return ProfileResult(
        backend=backend,
        model=args.model,
        context_length=args.context_length,
        position=args.position,
        seed=args.seed,
        found=found,
        expected=expected,
        response=response,
        input_tokens=int(inputs.input_ids.shape[1]),
        output_tokens=output_tokens,
        seconds=seconds,
        tokens_per_second=output_tokens / seconds if seconds > 0 else 0.0,
        cuda_peak_allocated_bytes=peak_allocated,
        cuda_peak_reserved_bytes=peak_reserved,
        compression_ratio=report["compression_ratio"] if report else None,
        n_compressed_blocks=report["n_compressed_blocks"] if report else None,
        n_fp16_blocks=report["n_fp16_blocks"] if report else None,
        bit_histogram=report["bit_histogram"] if report else None,
        precision_histogram=report["precision_histogram"] if report else None,
        config={
            "policy": args.policy,
            "block_size": args.block_size,
            "sink": args.sink,
            "window": args.window,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "granularity": args.granularity,
            "mixed": backend.endswith("_mix"),
            "importance_metric": args.importance_metric,
            "important_ratio": args.important_ratio,
            "high_key_bits": args.high_key_bits,
            "high_value_bits": args.high_value_bits,
            "low_key_bits": args.low_key_bits,
            "low_value_bits": args.low_value_bits,
            "group_size": args.group_size,
            "key_group_size": args.key_group_size,
            "value_group_size": args.value_group_size,
            "protected_layers": args.protected_layers,
            "protected_key_bits": args.protected_key_bits,
            "protected_value_bits": args.protected_value_bits,
            "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
            "incremental_materialize": args.incremental_materialize,
            "quant_budget_per_update": args.quant_budget_per_update,
        },
    )


def _write_outputs(results: list[ProfileResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "profile_memory.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    csv_path = output_dir / "profile_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "found",
                "seconds",
                "tokens_per_second",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "compression_ratio",
                "n_compressed_blocks",
                "n_fp16_blocks",
                "response",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.backend,
                    int(r.found),
                    f"{r.seconds:.4f}",
                    f"{r.tokens_per_second:.4f}",
                    r.cuda_peak_allocated_bytes,
                    r.cuda_peak_reserved_bytes,
                    r.compression_ratio,
                    r.n_compressed_blocks,
                    r.n_fp16_blocks,
                    r.response,
                ]
            )

    md_path = output_dir / "profile_memory.md"
    lines = [
        "# KVcatch Memory Profile",
        "",
        "| Backend | Found | Seconds | Tok/s | CUDA Peak Allocated | Ratio | Blocks |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        peak = (
            f"{r.cuda_peak_allocated_bytes / 1024 ** 2:.1f} MiB"
            if r.cuda_peak_allocated_bytes is not None
            else "-"
        )
        ratio = f"{r.compression_ratio:.3f}" if r.compression_ratio is not None else "-"
        blocks = (
            f"{r.n_compressed_blocks} compressed / {r.n_fp16_blocks} fp16"
            if r.n_compressed_blocks is not None
            else "-"
        )
        lines.append(
            f"| {r.backend} | {int(r.found)} | {r.seconds:.3f} | "
            f"{r.tokens_per_second:.3f} | {peak} | {ratio} | {blocks} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        choices=["dynamic", "block_tq", "block_tq_mix", "block_skvq", "block_skvq_mix", "all"],
        default="all",
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--position", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-slack", type=int, default=256)
    parser.add_argument("--pass-attention-mask", action="store_true")
    parser.add_argument("--record-attentions", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--policy", choices=["token", "window", "hybrid"], default="hybrid")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--key-bits", type=_parse_bits, default=2)
    parser.add_argument("--value-bits", type=_parse_bits, default=2)
    parser.add_argument("--granularity", choices=["per-vector", "per-block"], default="per-vector")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--key-group-size", type=int, default=None)
    parser.add_argument("--value-group-size", type=int, default=None)
    parser.add_argument("--clipping", type=float, default=0.92)
    parser.add_argument("--reorder-file", default=None)
    parser.add_argument("--max-cached-decompressed-blocks", type=int, default=0)
    parser.add_argument(
        "--incremental-materialize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache the dense materialized KV prefix and rebuild only changed suffix blocks.",
    )
    parser.add_argument(
        "--quant-budget-per-update",
        type=_parse_optional_int,
        default=None,
        help="Pseudo-async quant cursor budget: all/none or 0/1/2/... pages per update.",
    )

    parser.add_argument("--importance-metric", default="k_norm")
    parser.add_argument("--important-ratio", type=float, default=0.3)
    parser.add_argument("--high-key-bits", type=_parse_bits, default=4)
    parser.add_argument("--high-value-bits", type=_parse_bits, default=4)
    parser.add_argument("--low-key-bits", type=_parse_bits, default=2)
    parser.add_argument("--low-value-bits", type=_parse_bits, default=2)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--protected-layers", type=int, default=0)
    parser.add_argument("--protected-key-bits", type=_parse_bits, default=8)
    parser.add_argument("--protected-value-bits", type=_parse_bits, default=8)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path("runs") / f"profile_memory_{stamp}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    model_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "device_map": args.device_map,
        "dtype": _dtype_from_name(args.dtype),
    }
    attn_impl = args.attn_implementation or ("eager" if args.record_attentions else None)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if args.num_layers is None:
        args.num_layers = getattr(model.config, "num_hidden_layers", None)

    results: list[ProfileResult] = []
    for backend in _selected_backends(args.backend):
        print(f"\n=== backend={backend} ===")
        result = profile_backend(model, tokenizer, args, backend)
        results.append(result)
        ratio = (
            f"{result.compression_ratio:.3f}x"
            if result.compression_ratio is not None
            else "-"
        )
        peak = (
            f"{result.cuda_peak_allocated_bytes / 1024 ** 2:.1f} MiB"
            if result.cuda_peak_allocated_bytes is not None
            else "-"
        )
        print(
            f"found={result.found} ratio={ratio} peak={peak} "
            f"seconds={result.seconds:.2f} tok/s={result.tokens_per_second:.2f}"
        )
        print(f"response={result.response[:240]!r}")

    _write_outputs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
