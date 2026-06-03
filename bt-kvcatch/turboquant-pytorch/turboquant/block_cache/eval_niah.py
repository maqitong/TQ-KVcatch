"""Needle-in-a-Haystack evaluation for KVcatch cache backends.

The benchmark inserts a secret string into a synthetic long document at one or
more relative positions, asks the model to recover it, and records FOUND/MISS
plus cache memory statistics.

Examples:
    python -m turboquant.block_cache.eval_niah \
        --model D:\model\Llama3.2_3B --local-files-only \
        --backend block_tq_mix --context-lengths 512 --positions 0.5

    python -m turboquant.block_cache.eval_niah \
        --model Qwen/Qwen2.5-3B-Instruct --backend all \
        --context-lengths 2048,4096,8192 --positions 0.1,0.5,0.9 \
        --output-dir runs/niah_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from turboquant.block_cache import BlockKVCache
from turboquant.block_cache.methods import (
    NIAH_ALL_BACKENDS,
    build_policy as _shared_build_policy,
    cache_factory_for_backend,
    parse_backend_selection,
)


FILLER_SENTENCES = [
    "The committee reviewed routine budget notes and office maintenance plans.",
    "Several teams discussed documentation updates, onboarding checklists, and archive cleanup.",
    "The report included ordinary project milestones, scheduling details, and meeting summaries.",
    "Regional coordinators shared travel logistics, vendor renewals, and quarterly reminders.",
    "No sensitive operational decisions were included in this paragraph of background material.",
]


@dataclass
class NIAHResult:
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
    compression_ratio: float | None
    n_compressed_blocks: int | None
    n_fp16_blocks: int | None
    bit_histogram: dict | None
    precision_histogram: dict | None
    config: dict


def _parse_bits(value: str) -> float:
    bits = float(value)
    return int(bits) if bits == round(bits) else bits


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"all", "none", "null", "sync"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or all")
    return parsed


def _parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _selected_backends(name: str) -> list[str]:
    return parse_backend_selection(name, all_backends=NIAH_ALL_BACKENDS)


def _build_policy(args):
    return _shared_build_policy(args, window_uses_sink=False)


def _cache_factory(args, backend: str) -> Callable[[], BlockKVCache | None]:
    return cache_factory_for_backend(
        args,
        backend,
        window_uses_sink=False,
    )


def _is_attention_importance(name: str) -> bool:
    return name.replace("-", "_") in {"attention", "attention_score", "attn", "attn_score"}


def _needs_attention_feedback(args, cache) -> bool:
    return (
        cache is not None
        and getattr(getattr(cache, "config", None), "mixed_precision", False)
        and _is_attention_importance(args.importance_metric)
    )


@torch.no_grad()
def _greedy_generate_with_attention_feedback(
    model,
    inputs,
    cache,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Greedy decoding loop that records attentions before the next cache update.

    HuggingFace ``generate()`` returns attentions only after generation finishes,
    which is too late for attention-score PageMix. This loop feeds each forward's
    attentions back into ``BlockKVCache`` immediately, so deferred pages can be
    scored and compressed before subsequent decode steps.
    """
    generated = inputs["input_ids"]
    if max_new_tokens <= 0:
        return generated

    attention_mask = inputs.get("attention_mask")
    model_inputs = dict(inputs)
    outputs = model(
        **model_inputs,
        past_key_values=cache,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )
    cache.record_attentions(getattr(outputs, "attentions", None))

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)
    if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
        return generated

    for _ in range(1, max_new_tokens):
        step_inputs = {"input_ids": next_token}
        if attention_mask is not None:
            step_mask = torch.ones(
                attention_mask.shape[0],
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, step_mask], dim=-1)
            step_inputs["attention_mask"] = attention_mask

        outputs = model(
            **step_inputs,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        cache.record_attentions(getattr(outputs, "attentions", None))
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
            break

    return generated


def _secret_for_seed(seed: int) -> str:
    return f"AURORA-{(7749 + seed * 37) % 10000:04d}"


def _filler_unit(seed: int) -> str:
    shift = seed % len(FILLER_SENTENCES)
    ordered = FILLER_SENTENCES[shift:] + FILLER_SENTENCES[:shift]
    return " ".join(ordered) + "\n"


def build_prompt(tokenizer, context_length: int, position: float, seed: int) -> tuple[str, str]:
    expected = _secret_for_seed(seed)
    needle = (
        "\n--- Internal Retrieval Memo ---\n"
        f"The secret project code name is {expected}.\n"
        "--- End Internal Retrieval Memo ---\n"
    )
    prefix = (
        "You are given a long document. Read it carefully. "
        "At the end, answer with only the secret project code name.\n\n"
        "Document begins:\n"
    )
    suffix = "\nDocument ends.\nQuestion: What is the secret project code name?\nAnswer:"
    unit = _filler_unit(seed)

    fixed_tokens = len(tokenizer(prefix + needle + suffix, add_special_tokens=False).input_ids)
    unit_tokens = max(1, len(tokenizer(unit, add_special_tokens=False).input_ids))
    n_units = max(1, (context_length - fixed_tokens) // unit_tokens)
    insert_at = min(n_units, max(0, int(round(n_units * position))))

    parts: list[str] = []
    for i in range(n_units):
        if i == insert_at:
            parts.append(needle)
        parts.append(unit)
    if insert_at == n_units:
        parts.append(needle)

    prompt = prefix + "".join(parts) + suffix
    return prompt, expected


@torch.no_grad()
def run_case(model, tokenizer, args, backend: str, context_length: int, position: float, seed: int) -> NIAHResult:
    prompt, expected = build_prompt(tokenizer, context_length, position, seed)
    device = _model_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=context_length + args.prompt_slack,
    ).to(device)
    if not args.pass_attention_mask:
        # Single-sample NIAH prompts do not need padding masks. Omitting the
        # mask avoids a Transformers/Llama cache-mask path that can receive a
        # tensor-shaped kv_length with custom cache implementations.
        inputs.pop("attention_mask", None)

    cache = _cache_factory(args, backend)()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    started = time.perf_counter()
    if _needs_attention_feedback(args, cache):
        sequences = _greedy_generate_with_attention_feedback(
            model,
            inputs,
            cache,
            args.max_new_tokens,
            tokenizer.eos_token_id,
        )
    else:
        output = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_attentions=args.record_attentions,
            return_dict_in_generate=args.record_attentions,
        )
        if args.record_attentions:
            if cache is not None:
                cache.record_attentions(getattr(output, "attentions", None))
            sequences = output.sequences
        else:
            sequences = output
    seconds = time.perf_counter() - started

    new_tokens = sequences[0, inputs.input_ids.shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    found = expected.lower() in response.lower()
    report = cache.memory_report() if cache is not None else None

    return NIAHResult(
        backend=backend,
        model=args.model,
        context_length=context_length,
        position=position,
        seed=seed,
        found=found,
        expected=expected,
        response=response,
        input_tokens=int(inputs.input_ids.shape[1]),
        output_tokens=int(new_tokens.shape[0]),
        seconds=seconds,
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
            "mixed": backend.endswith("_mix"),
            "importance_metric": args.importance_metric,
            "important_ratio": args.important_ratio,
            "high_key_bits": args.high_key_bits,
            "high_value_bits": args.high_value_bits,
            "low_key_bits": args.low_key_bits,
            "low_value_bits": args.low_value_bits,
            "num_layers": args.num_layers,
            "protected_layers": args.protected_layers,
            "protected_key_bits": args.protected_key_bits,
            "protected_value_bits": args.protected_value_bits,
            "group_size": args.group_size,
            "key_group_size": args.key_group_size,
            "value_group_size": args.value_group_size,
            "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
            "quant_budget_per_update": args.quant_budget_per_update,
            "attention_feedback": _needs_attention_feedback(args, cache),
        },
    )


def _write_outputs(results: list[NIAHResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "niah_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    csv_path = output_dir / "niah_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "context_length",
                "position",
                "seed",
                "found",
                "compression_ratio",
                "seconds",
                "input_tokens",
                "response",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.backend,
                    r.context_length,
                    r.position,
                    r.seed,
                    int(r.found),
                    r.compression_ratio,
                    f"{r.seconds:.4f}",
                    r.input_tokens,
                    r.response,
                ]
            )

    md_path = output_dir / "niah_summary.md"
    grouped: dict[tuple[str, int], list[NIAHResult]] = {}
    for result in results:
        grouped.setdefault((result.backend, result.context_length), []).append(result)

    lines = [
        "# NIAH Summary",
        "",
        "| Backend | Context | Found | Total | Found Rate | Avg Ratio | Avg Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (backend, context), group in sorted(grouped.items()):
        found = sum(1 for r in group if r.found)
        total = len(group)
        ratios = [r.compression_ratio for r in group if r.compression_ratio is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        avg_seconds = sum(r.seconds for r in group) / total
        lines.append(
            f"| {backend} | {context} | {found} | {total} | "
            f"{found / total:.3f} | "
            f"{avg_ratio:.3f}" if avg_ratio is not None else f"| {backend} | {context} | {found} | {total} | {found / total:.3f} | -"
        )
        if avg_ratio is not None:
            lines[-1] += f" | {avg_seconds:.3f} |"
        else:
            lines[-1] += f" | {avg_seconds:.3f} |"

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
    parser.add_argument("--context-lengths", type=_parse_int_list, default="2048,4096")
    parser.add_argument("--positions", type=_parse_float_list, default="0.1,0.5,0.9")
    parser.add_argument("--seeds", type=_parse_int_list, default="0")
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
        "--quant-budget-per-update",
        type=_parse_optional_int,
        default=None,
        help=(
            "Pseudo-async quant cursor budget. Use all/none for synchronous "
            "compression, or 0/1/2/... pages per cache update."
        ),
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
    attn_impl = args.attn_implementation or (
        "eager" if args.record_attentions or _is_attention_importance(args.importance_metric) else None
    )
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    if args.num_layers is None:
        args.num_layers = getattr(model.config, "num_hidden_layers", None)

    results: list[NIAHResult] = []
    for backend in _selected_backends(args.backend):
        for context_length in args.context_lengths:
            for position in args.positions:
                for seed in args.seeds:
                    print(
                        f"\n=== backend={backend} ctx={context_length} "
                        f"pos={position:.2f} seed={seed} ==="
                    )
                    result = run_case(model, tokenizer, args, backend, context_length, position, seed)
                    results.append(result)
                    ratio = (
                        f"{result.compression_ratio:.3f}x"
                        if result.compression_ratio is not None
                        else "-"
                    )
                    print(
                        f"found={result.found} expected={result.expected} "
                        f"tokens={result.input_tokens} ratio={ratio} "
                        f"seconds={result.seconds:.2f}"
                    )
                    print(f"response={result.response[:240]!r}")

    if args.output_dir:
        _write_outputs(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
