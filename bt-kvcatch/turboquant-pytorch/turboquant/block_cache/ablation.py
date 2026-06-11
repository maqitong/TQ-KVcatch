"""Ablation runner for KVcatch page-level quantization experiments.

This script loads the model once and sweeps cache parameters such as
`block_size`, `important_ratio`, K/V bits, and K/V group sizes on the NIAH
benchmark. It is intentionally small and JSONL/CSV/Markdown oriented so long
server runs can be resumed or analyzed with ordinary tools.

Example:
    python -m turboquant.block_cache.ablation \
        --model D:\model\Llama3.2_3B --local-files-only \
        --backend block_tq_mix --context-lengths 512 --positions 0.5 \
        --sweep block_size=8,16 --sweep important_ratio=0.2,0.4

Config-file example:
    {
      "model": "D:\\model\\Llama3.2_3B",
      "backend": ["block_tq", "block_tq_mix"],
      "context_lengths": [2048, 4096],
      "positions": [0.1, 0.5, 0.9],
      "seeds": [0, 1, 2],
      "sweep": {
        "block_size": [8, 16, 32],
        "important_ratio": [0.2, 0.3, 0.5],
        "high_key_bits": [4],
        "high_value_bits": [4],
        "low_key_bits": [2],
        "low_value_bits": [2],
        "key_group_size": [128],
        "value_group_size": [64, 128]
      }
    }
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from turboquant.block_cache.eval_niah import (
    _dtype_from_name,
    _is_attention_importance,
    _parse_float_list,
    _parse_int_list,
    _parse_optional_int,
    _selected_backends,
    run_case,
)


DEFAULT_SWEEP: dict[str, list[Any]] = {}


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"none", "null", "all", "sync"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number == round(number) else number


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML configs; use JSON instead") from exc
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return loaded or {}


def _parse_sweep_overrides(values: list[str]) -> dict[str, list[Any]]:
    sweep: dict[str, list[Any]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--sweep must be name=a,b,c, got: {item}")
        name, raw_values = item.split("=", 1)
        parsed = [_parse_scalar(v) for v in raw_values.split(",") if v.strip()]
        if not parsed:
            raise ValueError(f"empty sweep values for {name}")
        sweep[name.strip().replace("-", "_")] = parsed
    return sweep


def _as_list(value: Any, parser=None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if parser is not None and isinstance(value, str):
        return parser(value)
    return [value]


def _sweep_points(sweep: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(sweep)
    values = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _base_args(cli_args, config: dict[str, Any]) -> SimpleNamespace:
    merged = {
        "model": cli_args.model or config.get("model"),
        "backend": config.get("backend", cli_args.backend),
        "context_lengths": config.get("context_lengths", cli_args.context_lengths),
        "positions": config.get("positions", cli_args.positions),
        "seeds": config.get("seeds", cli_args.seeds),
        "max_new_tokens": config.get("max_new_tokens", cli_args.max_new_tokens),
        "prompt_slack": config.get("prompt_slack", cli_args.prompt_slack),
        "pass_attention_mask": config.get("pass_attention_mask", cli_args.pass_attention_mask),
        "record_attentions": config.get("record_attentions", cli_args.record_attentions),
        "dtype": config.get("dtype", cli_args.dtype),
        "device_map": config.get("device_map", cli_args.device_map),
        "attn_implementation": config.get("attn_implementation", cli_args.attn_implementation),
        "local_files_only": config.get("local_files_only", cli_args.local_files_only),
        "trust_remote_code": config.get("trust_remote_code", cli_args.trust_remote_code),
        "policy": config.get("policy", cli_args.policy),
        "block_size": config.get("block_size", cli_args.block_size),
        "sink": config.get("sink", cli_args.sink),
        "window": config.get("window", cli_args.window),
        "key_bits": config.get("key_bits", cli_args.key_bits),
        "value_bits": config.get("value_bits", cli_args.value_bits),
        "granularity": config.get("granularity", cli_args.granularity),
        "group_size": config.get("group_size", cli_args.group_size),
        "key_group_size": config.get("key_group_size", cli_args.key_group_size),
        "value_group_size": config.get("value_group_size", cli_args.value_group_size),
        "clipping": config.get("clipping", cli_args.clipping),
        "reorder_file": config.get("reorder_file", cli_args.reorder_file),
        "max_cached_decompressed_blocks": config.get(
            "max_cached_decompressed_blocks", cli_args.max_cached_decompressed_blocks
        ),
        "incremental_materialize": config.get(
            "incremental_materialize", cli_args.incremental_materialize
        ),
        "quant_budget_per_update": config.get(
            "quant_budget_per_update", cli_args.quant_budget_per_update
        ),
        "importance_metric": config.get("importance_metric", cli_args.importance_metric),
        "important_ratio": config.get("important_ratio", cli_args.important_ratio),
        "pagemix_max_high_runs": config.get(
            "pagemix_max_high_runs", cli_args.pagemix_max_high_runs
        ),
        "high_key_bits": config.get("high_key_bits", cli_args.high_key_bits),
        "high_value_bits": config.get("high_value_bits", cli_args.high_value_bits),
        "low_key_bits": config.get("low_key_bits", cli_args.low_key_bits),
        "low_value_bits": config.get("low_value_bits", cli_args.low_value_bits),
        "num_layers": config.get("num_layers", cli_args.num_layers),
        "protected_layers": config.get("protected_layers", cli_args.protected_layers),
        "protected_key_bits": config.get("protected_key_bits", cli_args.protected_key_bits),
        "protected_value_bits": config.get("protected_value_bits", cli_args.protected_value_bits),
    }
    if merged["model"] is None:
        raise ValueError("--model is required unless provided in --config")
    merged["context_lengths"] = _as_list(merged["context_lengths"], _parse_int_list)
    merged["positions"] = _as_list(merged["positions"], _parse_float_list)
    merged["seeds"] = _as_list(merged["seeds"], _parse_int_list)
    if isinstance(merged["quant_budget_per_update"], str):
        merged["quant_budget_per_update"] = _parse_optional_int(
            merged["quant_budget_per_update"]
        )
    return SimpleNamespace(**merged)


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "ablation_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "ablation_summary.csv"
    columns = [
        "sweep_id",
        "backend",
        "context_length",
        "position",
        "seed",
        "found",
        "compression_ratio",
        "seconds",
        "input_tokens",
        "block_size",
        "important_ratio",
        "pagemix_max_high_runs",
        "high_key_bits",
        "high_value_bits",
        "low_key_bits",
        "low_value_bits",
        "key_group_size",
        "value_group_size",
        "incremental_materialize",
        "quant_budget_per_update",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "ablation.md"
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["sweep_id"], row["backend"]), []).append(row)

    lines = [
        "# KVcatch Ablation",
        "",
        "| Sweep | Backend | Found | Total | Found Rate | Avg Ratio | Avg Seconds | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for (sweep_id, backend), group in sorted(grouped.items()):
        found = sum(1 for r in group if r["found"])
        total = len(group)
        ratios = [r["compression_ratio"] for r in group if r["compression_ratio"] is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        avg_seconds = sum(float(r["seconds"]) for r in group) / total
        params = group[0]["sweep_params"]
        ratio_cell = f"{avg_ratio:.3f}" if avg_ratio is not None else "-"
        lines.append(
            f"| {sweep_id} | {backend} | {found} | {total} | "
            f"{found / total:.3f} | {ratio_cell} | {avg_seconds:.3f} | "
            f"`{json.dumps(params, ensure_ascii=False)}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default="block_tq_mix")
    parser.add_argument("--context-lengths", type=_parse_int_list, default="2048")
    parser.add_argument("--positions", type=_parse_float_list, default="0.5")
    parser.add_argument("--seeds", type=_parse_int_list, default="0")
    parser.add_argument("--sweep", action="append", default=[])
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-slack", type=int, default=256)
    parser.add_argument("--pass-attention-mask", action="store_true")
    parser.add_argument("--record-attentions", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--policy", choices=["token", "window", "hybrid"], default="hybrid")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--key-bits", type=_parse_scalar, default=2)
    parser.add_argument("--value-bits", type=_parse_scalar, default=2)
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
        default=False,
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
    parser.add_argument("--pagemix-max-high-runs", type=int, default=1)
    parser.add_argument("--high-key-bits", type=_parse_scalar, default=4)
    parser.add_argument("--high-value-bits", type=_parse_scalar, default=4)
    parser.add_argument("--low-key-bits", type=_parse_scalar, default=2)
    parser.add_argument("--low-value-bits", type=_parse_scalar, default=2)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--protected-layers", type=int, default=0)
    parser.add_argument("--protected-key-bits", type=_parse_scalar, default=8)
    parser.add_argument("--protected-value-bits", type=_parse_scalar, default=8)
    args = parser.parse_args()

    config = _load_config(args.config)
    base = _base_args(args, config)
    sweep = dict(DEFAULT_SWEEP)
    sweep.update(config.get("sweep", {}))
    sweep.update(_parse_sweep_overrides(args.sweep))
    sweep_points = _sweep_points({k: _as_list(v) for k, v in sweep.items()})

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path("runs") / f"ablation_{stamp}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {base.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base.model,
        local_files_only=base.local_files_only,
        trust_remote_code=base.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {base.model}")
    model_kwargs = {
        "local_files_only": base.local_files_only,
        "trust_remote_code": base.trust_remote_code,
        "device_map": base.device_map,
        "dtype": _dtype_from_name(base.dtype),
    }
    attn_impl = base.attn_implementation or (
        "eager"
        if base.record_attentions or _is_attention_importance(base.importance_metric)
        else None
    )
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(base.model, **model_kwargs)
    model.eval()
    if base.num_layers is None:
        base.num_layers = getattr(model.config, "num_hidden_layers", None)

    backends: list[str] = []
    for backend in _as_list(base.backend):
        backends.extend(_selected_backends(backend))

    rows: list[dict[str, Any]] = []
    for sweep_id, point in enumerate(sweep_points):
        run_args = SimpleNamespace(**vars(base))
        for key, value in point.items():
            setattr(run_args, key, value)

        print(f"\n=== sweep_id={sweep_id} params={point} ===")
        for backend in backends:
            for context_length in run_args.context_lengths:
                for position in run_args.positions:
                    for seed in run_args.seeds:
                        print(
                            f"backend={backend} ctx={context_length} "
                            f"pos={position:.2f} seed={seed}"
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        result = run_case(
                            model,
                            tokenizer,
                            run_args,
                            backend,
                            int(context_length),
                            float(position),
                            int(seed),
                        )
                        row = asdict(result)
                        row.update(point)
                        row["sweep_id"] = sweep_id
                        row["sweep_params"] = point
                        rows.append(row)
                        ratio = (
                            f"{result.compression_ratio:.3f}x"
                            if result.compression_ratio is not None
                            else "-"
                        )
                        print(
                            f"found={result.found} ratio={ratio} "
                            f"seconds={result.seconds:.2f} response={result.response[:120]!r}"
                        )

    _write_outputs(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
