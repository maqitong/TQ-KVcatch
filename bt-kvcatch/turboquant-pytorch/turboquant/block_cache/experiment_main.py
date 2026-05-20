"""Main KVcatch comparison experiment.

This script is the M1.4 entry point from `tasks/kvcatch_completion_plan.md`.
It runs a compact Needle-in-a-Haystack comparison across the core methods:

  1. FP16 / DynamicCache
  2. SKVQ Baseline / TokenBlockPolicy + SKVQ
  3. TurboQuant Baseline / TokenBlockPolicy + TurboQuant
  4. Hybrid + SKVQ + Block
  5. Hybrid + TurboQuant + Block
  6. Hybrid + TurboQuant + Block + page-level mixed precision

The default settings are intentionally modest so the script can be smoke-tested
locally, then scaled up on a 4090 server by increasing context lengths,
positions, and seeds.
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

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    HybridPolicy,
    TokenBlockPolicy,
)
from turboquant.block_cache.eval_niah import build_prompt


@dataclass
class MethodSpec:
    name: str
    backend: str
    quant_backend: str | None
    policy: str
    method_group: str = "method"
    page_quant_scheme: str = "none"
    mixed_precision: bool = False
    importance_metric: str = "k_norm"
    key_bits: float = 2
    value_bits: float = 2
    high_key_bits: float = 4
    high_value_bits: float = 4
    low_key_bits: float = 2
    low_value_bits: float = 2


@dataclass
class MainResult:
    method: str
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


def _policy_from_name(name: str, args):
    if name == "token":
        return TokenBlockPolicy()
    if name == "hybrid":
        return HybridPolicy(sink_size=args.sink, window_size=args.window)
    raise ValueError(f"unknown method policy: {name}")


def build_methods(args) -> list[MethodSpec]:
    baseline_scheme = f"Uniform K{args.key_bits}/V{args.value_bits}"
    mix_scheme = (
        f"Mixed high K{args.high_key_bits}/V{args.high_value_bits}, "
        f"low K{args.low_key_bits}/V{args.low_value_bits}"
    )
    methods = [
        MethodSpec(
            name="FP16",
            backend="dynamic",
            quant_backend=None,
            policy="none",
            method_group="reference",
            page_quant_scheme="FP16",
        ),
        MethodSpec(
            name="SKVQ Baseline",
            backend="block_skvq",
            quant_backend="skvq",
            policy="token",
            method_group="baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="TurboQuant Baseline",
            backend="block_tq",
            quant_backend="turboquant",
            policy="token",
            method_group="baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="Hybrid+SKVQ+Block",
            backend="block_skvq",
            quant_backend="skvq",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="Hybrid+TQ+Block",
            backend="block_tq",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
        ),
        MethodSpec(
            name="Hybrid+TQ+Block+PageMix",
            backend="block_tq_mix",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="method",
            page_quant_scheme=mix_scheme,
            mixed_precision=True,
            importance_metric=args.importance_metric,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            high_key_bits=args.high_key_bits,
            high_value_bits=args.high_value_bits,
            low_key_bits=args.low_key_bits,
            low_value_bits=args.low_value_bits,
        ),
    ]
    if args.include_random_mix:
        methods.append(
            MethodSpec(
                name="Hybrid+TQ+RandomMix",
                backend="block_tq_random_mix",
                quant_backend="turboquant",
                policy="hybrid",
                method_group="ablation",
                page_quant_scheme=mix_scheme,
                mixed_precision=True,
                importance_metric="random",
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                high_key_bits=args.high_key_bits,
                high_value_bits=args.high_value_bits,
                low_key_bits=args.low_key_bits,
                low_value_bits=args.low_value_bits,
            )
        )
    return methods


def _cache_factory(args, method: MethodSpec) -> Callable[[], BlockKVCache | None]:
    if method.quant_backend is None:
        return lambda: None

    def make_cache() -> BlockKVCache:
        cfg = BlockCacheConfig(
            block_size=args.block_size,
            key_bits=method.key_bits,
            value_bits=method.value_bits,
            granularity=args.granularity,
            policy=_policy_from_name(method.policy, args),
            quant_backend=method.quant_backend,
            mixed_precision=method.mixed_precision,
            importance_metric=method.importance_metric,
            important_ratio=args.important_ratio,
            high_key_bits=method.high_key_bits,
            high_value_bits=method.high_value_bits,
            low_key_bits=method.low_key_bits,
            low_value_bits=method.low_value_bits,
            num_layers=args.num_layers,
            protected_layers=args.protected_layers,
            protected_key_bits=args.protected_key_bits,
            protected_value_bits=args.protected_value_bits,
            group_size=args.group_size,
            key_group_size=args.key_group_size,
            value_group_size=args.value_group_size,
            clipping=args.clipping,
            reorder_file=args.reorder_file,
            max_cached_decompressed_blocks=args.max_cached_decompressed_blocks,
        )
        return BlockKVCache(cfg)

    return make_cache


@torch.no_grad()
def run_niah_case(
    model,
    tokenizer,
    args,
    method: MethodSpec,
    context_length: int,
    position: float,
    seed: int,
) -> MainResult:
    prompt, expected = build_prompt(tokenizer, context_length, position, seed)
    device = _model_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=context_length + args.prompt_slack,
    ).to(device)
    if not args.pass_attention_mask:
        inputs.pop("attention_mask", None)

    cache = _cache_factory(args, method)()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    return MainResult(
        method=method.name,
        backend=method.backend,
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
            "block_size": args.block_size,
            "method_group": method.method_group,
            "sink": args.sink if method.policy == "hybrid" else None,
            "window": args.window if method.policy == "hybrid" else None,
            "policy": method.policy,
            "quant_backend": method.quant_backend,
            "page_quant_scheme": method.page_quant_scheme,
            "key_bits": method.key_bits,
            "value_bits": method.value_bits,
            "mixed_precision": method.mixed_precision,
            "importance_metric": method.importance_metric if method.mixed_precision else None,
            "important_ratio": args.important_ratio if method.mixed_precision else None,
            "high_key_bits": method.high_key_bits if method.mixed_precision else None,
            "high_value_bits": method.high_value_bits if method.mixed_precision else None,
            "low_key_bits": method.low_key_bits if method.mixed_precision else None,
            "low_value_bits": method.low_value_bits if method.mixed_precision else None,
            "num_layers": args.num_layers,
            "protected_layers": args.protected_layers,
            "protected_key_bits": args.protected_key_bits,
            "protected_value_bits": args.protected_value_bits,
            "group_size": args.group_size,
            "key_group_size": args.key_group_size,
            "value_group_size": args.value_group_size,
            "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
        },
    )


def _write_outputs(results: list[MainResult], args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "main_exp_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    csv_path = output_dir / "main_exp_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "method_group",
                "quant_backend",
                "policy",
                "page_quant_scheme",
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
                    r.method,
                    r.config.get("method_group"),
                    r.config.get("quant_backend"),
                    r.config.get("policy"),
                    r.config.get("page_quant_scheme"),
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

    md_path = output_dir / "main_exp.md"
    grouped: dict[tuple[str, int], list[MainResult]] = {}
    for result in results:
        grouped.setdefault((result.method, result.context_length), []).append(result)

    method_order = {method.name: idx for idx, method in enumerate(build_methods(args))}
    lines = [
        "# KVcatch Main Experiment",
        "",
        "## NIAH Summary",
        "",
        "| Method | Group | Policy | Scheme | Context | Found | Total | Found Rate | Avg Ratio | Avg Seconds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (method, context), group in sorted(
        grouped.items(), key=lambda item: (item[0][1], method_order.get(item[0][0], 999))
    ):
        found = sum(1 for r in group if r.found)
        total = len(group)
        ratios = [r.compression_ratio for r in group if r.compression_ratio is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        avg_seconds = sum(r.seconds for r in group) / total
        ratio_cell = f"{avg_ratio:.3f}" if avg_ratio is not None else "-"
        group_name = group[0].config.get("method_group", "-")
        policy = group[0].config.get("policy", "-")
        scheme = group[0].config.get("page_quant_scheme", "-")
        lines.append(
            f"| {method} | {group_name} | {policy} | {scheme} | {context} | "
            f"{found} | {total} | {found / total:.3f} | {ratio_cell} | {avg_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Settings",
            "",
            "```json",
            json.dumps(
                {
                    "model": args.model,
                    "context_lengths": args.context_lengths,
                    "positions": args.positions,
                    "seeds": args.seeds,
                    "block_size": args.block_size,
                    "sink": args.sink,
                    "window": args.window,
                    "key_bits": args.key_bits,
                    "value_bits": args.value_bits,
                    "group_size": args.group_size,
                    "key_group_size": args.key_group_size,
                    "value_group_size": args.value_group_size,
                    "max_cached_decompressed_blocks": args.max_cached_decompressed_blocks,
                    "important_ratio": args.important_ratio,
                    "high_bits": [args.high_key_bits, args.high_value_bits],
                    "low_bits": [args.low_key_bits, args.low_value_bits],
                    "num_layers": args.num_layers,
                    "protected_layers": args.protected_layers,
                    "protected_bits": [
                        args.protected_key_bits,
                        args.protected_value_bits,
                    ],
                },
                indent=2,
            ),
            "```",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def _namespace_lists(args) -> None:
    args.context_lengths = (
        args.context_lengths
        if isinstance(args.context_lengths, list)
        else _parse_int_list(args.context_lengths)
    )
    args.positions = (
        args.positions if isinstance(args.positions, list) else _parse_float_list(args.positions)
    )
    args.seeds = args.seeds if isinstance(args.seeds, list) else _parse_int_list(args.seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-lengths", default="2048,4096,8192")
    parser.add_argument("--positions", default="0.1,0.5,0.9")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-slack", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--pass-attention-mask", action="store_true")
    parser.add_argument("--record-attentions", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-random-mix", action="store_true")

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
    _namespace_lists(args)

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path("runs") / f"main_exp_{stamp}")

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

    results: list[MainResult] = []
    methods = build_methods(args)
    for context_length in args.context_lengths:
        for position in args.positions:
            for seed in args.seeds:
                for method in methods:
                    print(
                        f"\n=== method={method.name} ctx={context_length} "
                        f"pos={position:.2f} seed={seed} ==="
                    )
                    result = run_niah_case(
                        model, tokenizer, args, method, context_length, position, seed
                    )
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

    _write_outputs(results, args)


if __name__ == "__main__":
    main()
