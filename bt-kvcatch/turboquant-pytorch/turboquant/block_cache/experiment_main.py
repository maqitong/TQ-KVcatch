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
import gc
import json
import re
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
    paper_baseline: str | None = None


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
            name="SKVQ skvq_baseline (native)",
            backend="skvq_native",
            quant_backend="skvq",
            policy="window",
            method_group="paper_baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="skvq_native",
        ),
        MethodSpec(
            name="TurboQuant V3 flat (rw=128, K2/V2)",
            backend="v3_flat",
            quant_backend=None,
            policy="none",
            method_group="paper_baseline",
            page_quant_scheme=f"V3 flat K{args.key_bits}/V{args.value_bits}",
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="v3_flat",
        ),
        MethodSpec(
            name="TurboQuant pure (tq_replace)",
            backend="block_tq",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="paper_baseline",
            page_quant_scheme=baseline_scheme,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            paper_baseline="tq_pure",
        ),
        MethodSpec(
            name="TurboQuant pure+PageMix",
            backend="block_tq_pure_mix",
            quant_backend="turboquant",
            policy="hybrid",
            method_group="paper_baseline",
            page_quant_scheme=mix_scheme,
            mixed_precision=True,
            importance_metric=args.importance_metric,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            high_key_bits=args.high_key_bits,
            high_value_bits=args.high_value_bits,
            low_key_bits=args.low_key_bits,
            low_value_bits=args.low_value_bits,
            paper_baseline="tq_pure_mix",
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


def _paper_tq_pure_policy():
    from turboquant.block_cache.skvq_native_integration import PAPER_SINK, PAPER_WINDOW

    return HybridPolicy(sink_size=PAPER_SINK, window_size=PAPER_WINDOW)


def _cache_factory(args, method: MethodSpec) -> Callable:
    if method.backend == "skvq_native":
        return lambda: None

    if method.backend == "v3_flat":
        from turboquant.block_cache.v3_flat_cache import V3FlatCache

        def make_v3() -> V3FlatCache:
            return V3FlatCache(
                key_bits=int(method.key_bits),
                value_bits=int(method.value_bits),
                residual_window=int(args.residual_window),
                protected_layers=0,
                n_layers=int(args.num_layers or 32),
                seed=42,
            )

        return make_v3

    if method.quant_backend is None:
        return lambda: None

    def make_cache() -> BlockKVCache:
        if method.paper_baseline in ("tq_pure", "tq_pure_mix"):
            from turboquant.block_cache.skvq_native_integration import (
                PAPER_CLIP,
                paper_pure_layer_protection,
            )

            policy = _paper_tq_pure_policy()
            reorder_file = None
            protected_layers, prot_k, prot_v = paper_pure_layer_protection(
                method.paper_baseline, args
            )
            clipping = PAPER_CLIP
        else:
            prot_k = args.protected_key_bits
            prot_v = args.protected_value_bits
            policy = _policy_from_name(method.policy, args)
            reorder_file = args.reorder_file
            protected_layers = args.protected_layers
            clipping = args.clipping

        cfg = BlockCacheConfig(
            block_size=args.block_size,
            key_bits=method.key_bits,
            value_bits=method.value_bits,
            granularity=args.granularity,
            policy=policy,
            quant_backend=method.quant_backend,
            mixed_precision=method.mixed_precision,
            importance_metric=method.importance_metric,
            important_ratio=args.important_ratio,
            high_key_bits=method.high_key_bits,
            high_value_bits=method.high_value_bits,
            low_key_bits=method.low_key_bits,
            low_value_bits=method.low_value_bits,
            num_layers=args.num_layers,
            protected_layers=protected_layers,
            protected_key_bits=prot_k,
            protected_value_bits=prot_v,
            group_size=args.group_size,
            key_group_size=args.key_group_size,
            value_group_size=args.value_group_size,
            clipping=clipping,
            reorder_file=reorder_file,
            max_cached_decompressed_blocks=args.max_cached_decompressed_blocks,
        )
        return BlockKVCache(cfg)

    return make_cache


def _method_config(args, method: MethodSpec) -> dict:
    if method.paper_baseline == "v3_flat":
        return {
            "method_group": method.method_group,
            "policy": "v3_flat",
            "page_quant_scheme": method.page_quant_scheme,
            "key_bits": method.key_bits,
            "value_bits": method.value_bits,
            "residual_window": int(args.residual_window),
            "mixed_precision": False,
            "paper_baseline": "v3_flat",
            "integration": "turboquant/V3FlatCache+MSECompressor",
        }
    if method.paper_baseline in ("tq_pure", "tq_pure_mix"):
        from turboquant.block_cache.skvq_native_integration import (
            PAPER_CLIP,
            PAPER_SINK,
            PAPER_WINDOW,
            paper_pure_layer_protection,
        )

        prot_layers, prot_k, prot_v = paper_pure_layer_protection(
            method.paper_baseline, args
        )
        return {
            "block_size": args.block_size,
            "method_group": method.method_group,
            "policy": "hybrid",
            "quant_backend": method.quant_backend,
            "page_quant_scheme": method.page_quant_scheme,
            "sink": PAPER_SINK,
            "window": PAPER_WINDOW,
            "key_bits": method.key_bits,
            "value_bits": method.value_bits,
            "mixed_precision": method.mixed_precision,
            "important_ratio": args.important_ratio if method.mixed_precision else None,
            "importance_metric": method.importance_metric if method.mixed_precision else None,
            "high_key_bits": method.high_key_bits if method.mixed_precision else None,
            "high_value_bits": method.high_value_bits if method.mixed_precision else None,
            "low_key_bits": method.low_key_bits if method.mixed_precision else None,
            "low_value_bits": method.low_value_bits if method.mixed_precision else None,
            "paper_baseline": method.paper_baseline,
            "reorder": False,
            "protected_layers": prot_layers,
            "protected_key_bits": prot_k if method.paper_baseline == "tq_pure_mix" else None,
            "protected_value_bits": prot_v if method.paper_baseline == "tq_pure_mix" else None,
            "clipping": PAPER_CLIP,
            "integration": "turboquant-pytorch/BlockKVCache",
        }
    if method.paper_baseline == "skvq_native":
        from turboquant.block_cache.skvq_native_integration import skvq_native_config

        cfg = skvq_native_config(
            key_bits=method.key_bits,
            value_bits=method.value_bits,
            reorder_file=args.reorder_file,
        )
        cfg["method_group"] = method.method_group
        cfg["page_quant_scheme"] = method.page_quant_scheme
        cfg["quant_backend"] = method.quant_backend
        return cfg
    return {
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
    }


def _niah_run_key(method: str, context_length: int, position: float, seed: int) -> str:
    return f"{method}|{context_length}|{position:.4f}|{seed}"


def _parse_resume_log(log_path: Path) -> set[str]:
    pattern = re.compile(
        r"^=== method=(.+?) ctx=(\d+) pos=([\d.]+) seed=(\d+) ===$"
    )
    completed: set[str] = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line.strip())
        if m:
            completed.add(
                _niah_run_key(
                    m.group(1),
                    int(m.group(2)),
                    float(m.group(3)),
                    int(m.group(4)),
                )
            )
    return completed


def _load_results_jsonl(jsonl_path: Path) -> tuple[list[MainResult], set[str]]:
    results: list[MainResult] = []
    completed: set[str] = set()
    if not jsonl_path.exists():
        return results, completed
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        result = MainResult(**row)
        results.append(result)
        completed.add(
            _niah_run_key(
                result.method,
                result.context_length,
                result.position,
                result.seed,
            )
        )
    return results, completed


def _append_result_jsonl(jsonl_path: Path, result: MainResult) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


@torch.no_grad()
def run_niah_case_skvq_native(
    model,
    tokenizer,
    args,
    method: MethodSpec,
    context_length: int,
    position: float,
    seed: int,
) -> MainResult:
    from turboquant.block_cache.skvq_native_integration import (
        build_skvq_baseline_manager,
        clear_quantizer,
        detach_quantizer,
        plug_quantizer,
    )

    if args.reorder_file is None:
        raise ValueError("skvq_native requires --reorder-file")

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

    detach_quantizer(model)
    manager = build_skvq_baseline_manager(
        model,
        reorder_file=args.reorder_file,
        key_bits=method.key_bits,
        value_bits=method.value_bits,
        group_size=args.group_size,
        skvq_root=args.skvq_root,
    )
    plug_quantizer(model, manager)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from turboquant.block_cache.skvq_native_integration import skvq_native_generate

    started = time.perf_counter()
    oom = False
    error_msg = ""
    try:
        sequences = skvq_native_generate(
            model,
            tokenizer,
            inputs,
            max_new_tokens=args.max_new_tokens,
        )
        new_tokens = sequences[0, inputs.input_ids.shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        found = expected.lower() in response.lower()
        output_tokens = int(new_tokens.shape[0])
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        error_msg = str(exc).splitlines()[0][:500]
        response = ""
        found = False
        output_tokens = 0
        sequences = None
    finally:
        seconds = time.perf_counter() - started
        clear_quantizer(model)
        detach_quantizer(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = _method_config(args, method)
    if oom:
        config["status"] = "oom"
        config["error"] = error_msg
    elif error_msg:
        config["status"] = "error"
        config["error"] = error_msg

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
        output_tokens=output_tokens,
        seconds=seconds,
        compression_ratio=None,
        n_compressed_blocks=None,
        n_fp16_blocks=None,
        bit_histogram=None,
        precision_histogram=None,
        config=config,
    )


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
        config=_method_config(args, method),
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
    parser.add_argument(
        "--only-paper-baselines",
        action="store_true",
        help="Run paper baselines: SKVQ native, TQ pure, TQ pure+PageMix, V3 flat",
    )
    parser.add_argument(
        "--only-v3-baselines",
        action="store_true",
        help="Run only TurboQuant V3 flat (no block) and TurboQuant pure+PageMix",
    )
    parser.add_argument(
        "--filter-paper-baseline",
        default=None,
        help="Run only methods with this paper_baseline tag (e.g. tq_pure_mix)",
    )
    parser.add_argument(
        "--residual-window",
        type=int,
        default=128,
        help="V3FlatCache: recent tokens kept in FP16 (author default 128; 0 = no tail)",
    )
    parser.add_argument("--skvq-root", default=None, help="Path to SKVQ repo for native baseline")
    parser.add_argument(
        "--resume-log",
        action="append",
        default=None,
        help="Skip keys already present in log file(s); repeatable",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Load existing output-dir/main_results.jsonl and append new rows",
    )

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

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    methods = build_methods(args)
    if args.only_paper_baselines:
        methods = [m for m in methods if m.paper_baseline is not None]
    if args.only_v3_baselines:
        methods = [
            m
            for m in methods
            if m.paper_baseline in ("v3_flat", "tq_pure_mix")
        ]
    if args.filter_paper_baseline:
        methods = [m for m in methods if m.paper_baseline == args.filter_paper_baseline]

    block_methods = [m for m in methods if m.backend != "skvq_native"]
    native_methods = [m for m in methods if m.backend == "skvq_native"]

    if args.num_layers is None:
        cfg = AutoConfig.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        args.num_layers = getattr(cfg, "num_hidden_layers", None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "main_exp_results.jsonl"

    results: list[MainResult] = []
    completed: set[str] = set()
    if args.append_results:
        results, completed = _load_results_jsonl(jsonl_path)
        print(f"[resume] loaded {len(results)} rows from {jsonl_path}")
    if args.resume_log:
        for log_path in args.resume_log:
            from_log = _parse_resume_log(Path(log_path))
            completed |= from_log
            print(f"[resume] {len(from_log)} keys from log {log_path}")

    def _has_pending_block_work() -> bool:
        for context_length in args.context_lengths:
            for position in args.positions:
                for seed in args.seeds:
                    for method in block_methods:
                        if _niah_run_key(method.name, context_length, position, seed) not in completed:
                            return True
        return False

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    if block_methods and _has_pending_block_work():
        print(f"Loading HF model: {args.model}")
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

    skipped = 0

    def _run_method(method: MethodSpec, context_length: int, position: float, seed: int) -> None:
        nonlocal skipped
        key = _niah_run_key(method.name, context_length, position, seed)
        if key in completed:
            skipped += 1
            print(
                f"\n=== method={method.name} ctx={context_length} "
                f"pos={position:.2f} seed={seed} === [skip resume]"
            )
            return
        print(
            f"\n=== method={method.name} ctx={context_length} "
            f"pos={position:.2f} seed={seed} ==="
        )
        if method.backend == "skvq_native":
            raise RuntimeError("skvq_native must run in native phase")
        result = run_niah_case(model, tokenizer, args, method, context_length, position, seed)
        results.append(result)
        completed.add(key)
        _append_result_jsonl(jsonl_path, result)
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

    if block_methods and model is not None:
        for context_length in args.context_lengths:
            for position in args.positions:
                for seed in args.seeds:
                    for method in block_methods:
                        _run_method(method, context_length, position, seed)
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if native_methods:
        from turboquant.block_cache.skvq_native_integration import load_skvq_llama

        print("\n=== Loading SKVQ native Llama model ===")
        skvq_model, skvq_tokenizer = load_skvq_llama(
            args.model,
            skvq_root=args.skvq_root,
            dtype=_dtype_from_name(args.dtype),
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        for context_length in args.context_lengths:
            for position in args.positions:
                for seed in args.seeds:
                    for method in native_methods:
                        key = _niah_run_key(method.name, context_length, position, seed)
                        if key in completed:
                            skipped += 1
                            print(
                                f"\n=== method={method.name} ctx={context_length} "
                                f"pos={position:.2f} seed={seed} === [skip resume]"
                            )
                            continue
                        print(
                            f"\n=== method={method.name} ctx={context_length} "
                            f"pos={position:.2f} seed={seed} ==="
                        )
                        try:
                            result = run_niah_case_skvq_native(
                                skvq_model,
                                skvq_tokenizer,
                                args,
                                method,
                                context_length,
                                position,
                                seed,
                            )
                        except RuntimeError as exc:
                            msg = str(exc).lower()
                            if "out of memory" not in msg and "cuda error" not in msg:
                                raise
                            result = MainResult(
                                method=method.name,
                                backend=method.backend,
                                model=args.model,
                                context_length=context_length,
                                position=position,
                                seed=seed,
                                found=False,
                                expected="",
                                response="",
                                input_tokens=0,
                                output_tokens=0,
                                seconds=0.0,
                                compression_ratio=None,
                                n_compressed_blocks=None,
                                n_fp16_blocks=None,
                                bit_histogram=None,
                                precision_histogram=None,
                                config={
                                    **_method_config(args, method),
                                    "status": "oom",
                                    "error": str(exc).splitlines()[0][:500],
                                },
                            )
                        results.append(result)
                        completed.add(key)
                        _append_result_jsonl(jsonl_path, result)
                        status = result.config.get("status", "ok")
                        print(
                            f"[{status}] found={result.found} expected={result.expected} "
                            f"tokens={result.input_tokens} seconds={result.seconds:.2f}"
                        )
                        if result.response:
                            print(f"response={result.response[:240]!r}")
        del skvq_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[resume] skipped {skipped} already-completed runs")
    _write_outputs(results, args)


if __name__ == "__main__":
    main()
