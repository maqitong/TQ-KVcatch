"""Perplexity evaluation for KVcatch cache backends.

Examples:
    python -m turboquant.block_cache.eval_ppl \
        --model D:\model\Llama3.2_3B \
        --inline-text "The quick brown fox jumps over the lazy dog." \
        --backend all --seq-len 64 --stride 32

    python -m turboquant.block_cache.eval_ppl \
        --model Qwen/Qwen2.5-3B-Instruct \
        --dataset wikitext --dataset-config wikitext-103-raw-v1 \
        --split test --max-samples 64 --backend block_tq_mix
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from turboquant.block_cache import BlockKVCache
from turboquant.block_cache.bpw_metrics import attach_bpw_fields
from turboquant.block_cache.methods import (
    PPL_ALL_BACKENDS,
    build_policy as _shared_build_policy,
    cache_factory_for_backend,
    paper_tq_pure_policy as _shared_paper_tq_pure_policy,
    parse_backend_selection,
)
from turboquant.block_cache.v2_paper_cache import V2PaperCache
from turboquant.block_cache.v3_flat_cache import V3FlatCache


@dataclass
class PPLResult:
    backend: str
    model: str
    tokens: int
    loss: float
    ppl: float
    seconds: float
    peak_memory_bytes: int | None
    avg_compression_ratio: float | None
    avg_compressed_blocks: float | None
    avg_fp16_blocks: float | None
    config: dict
    k_bpw: float | None = None
    v_bpw: float | None = None
    avg_bpw: float | None = None
    effective_bpw: float | None = None


def _parse_bits(value: str) -> float:
    bits = float(value)
    return int(bits) if bits == round(bits) else bits


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


def _paper_tq_pure_policy():
    return _shared_paper_tq_pure_policy()


def _build_policy(args):
    return _shared_build_policy(args, window_uses_sink=True)


def _cache_factory(args, backend: str) -> Callable[[], BlockKVCache | V2PaperCache | V3FlatCache | None]:
    return cache_factory_for_backend(
        args,
        backend,
        include_v2=True,
        include_v3=True,
        window_uses_sink=True,
    )


def _selected_backends(name: str) -> list[str]:
    return parse_backend_selection(name, all_backends=PPL_ALL_BACKENDS)


def _load_texts(args) -> list[str]:
    if args.inline_text:
        return [args.inline_text]

    if args.text_file:
        path = Path(args.text_file)
        return [path.read_text(encoding=args.text_encoding)]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required unless --text-file or --inline-text is provided"
        ) from exc

    kwargs = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    ds = load_dataset(args.dataset, split=args.split, **kwargs)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    column = args.text_column
    if column is None:
        column = "text" if "text" in ds.column_names else ds.column_names[0]

    texts = [str(row[column]) for row in ds if str(row[column]).strip()]
    if not texts:
        raise ValueError("no non-empty text samples found")
    return texts


def _tokenize_corpus(tokenizer, texts: Iterable[str], max_tokens: int) -> torch.Tensor:
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids
    if max_tokens > 0:
        input_ids = input_ids[:, :max_tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("need at least two tokens to compute perplexity")
    return input_ids


@torch.no_grad()
def evaluate_backend(model, input_ids: torch.Tensor, args, backend: str) -> PPLResult:
    device = _model_input_device(model)
    cache_factory = _cache_factory(args, backend)
    total_len = input_ids.shape[1]
    max_length = min(args.seq_len, total_len)

    nll_sum = 0.0
    n_loss_tokens = 0
    cache_reports: list[dict] = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    prev_end = 0
    for begin_loc in range(0, total_len, args.stride):
        end_loc = min(begin_loc + max_length, total_len)
        trg_len = end_loc - prev_end
        if trg_len <= 0:
            break

        chunk = input_ids[:, begin_loc:end_loc].to(device)
        target = chunk.clone()
        target[:, :-trg_len] = -100

        cache = cache_factory()
        outputs = model(
            input_ids=chunk,
            labels=target,
            past_key_values=cache,
            use_cache=True,
            output_attentions=args.record_attentions,
        )
        if cache is not None and args.record_attentions:
            cache.record_attentions(getattr(outputs, "attentions", None))

        valid_tokens = int((target[:, 1:] != -100).sum().item())
        if valid_tokens > 0:
            nll_sum += float(outputs.loss.item()) * valid_tokens
            n_loss_tokens += valid_tokens

        if cache is not None and hasattr(cache, "memory_report"):
            cache_reports.append(cache.memory_report())

        prev_end = end_loc
        if end_loc == total_len:
            break

    seconds = time.perf_counter() - started
    loss = nll_sum / max(n_loss_tokens, 1)
    ppl = math.exp(loss) if loss < 20 else float("inf")
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    ratios = [r["compression_ratio"] for r in cache_reports if r.get("compression_ratio")]
    compressed = [r["n_compressed_blocks"] for r in cache_reports]
    fp16 = [r["n_fp16_blocks"] for r in cache_reports]

    bit_histogram: dict[str, int] = {}
    for report in cache_reports:
        for key, count in (report.get("bit_histogram") or {}).items():
            bit_histogram[key] = bit_histogram.get(key, 0) + int(count)

    paper_sink = args.sink
    paper_window = args.window
    if backend in ("block_tq_pure", "block_tq_pure_mix"):
        from turboquant.block_cache.skvq_native_integration import PAPER_SINK, PAPER_WINDOW

        paper_sink = PAPER_SINK
        paper_window = PAPER_WINDOW

    result = PPLResult(
        backend=backend,
        model=args.model,
        tokens=n_loss_tokens,
        loss=loss,
        ppl=ppl,
        seconds=seconds,
        peak_memory_bytes=peak,
        avg_compression_ratio=sum(ratios) / len(ratios) if ratios else None,
        avg_compressed_blocks=sum(compressed) / len(compressed) if compressed else None,
        avg_fp16_blocks=sum(fp16) / len(fp16) if fp16 else None,
        config={
            "seq_len": args.seq_len,
            "stride": args.stride,
            "policy": (
                "v2_paper"
                if backend == "v2_paper"
                else ("v3_flat" if backend == "v3_flat" else args.policy)
            ),
            "block_size": args.block_size,
            "sink": paper_sink,
            "window": paper_window,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "mixed": backend.endswith("_mix"),
            "paper_baseline": (
                "tq_pure_mix"
                if backend == "block_tq_pure_mix"
                else ("tq_pure" if backend == "block_tq_pure" else None)
            ),
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
            "residual_window": args.residual_window if backend == "v3_flat" else None,
            "integration": (
                "turboquant/V2PaperCache+CompressorV2(QJL)"
                if backend == "v2_paper"
                else (
                    "turboquant/V3FlatCache+MSECompressor"
                    if backend == "v3_flat"
                    else "turboquant-pytorch/BlockKVCache"
                )
            ),
        },
    )
    payload = attach_bpw_fields(
        {
            "backend": backend,
            "avg_compression_ratio": result.avg_compression_ratio,
            "bit_histogram": bit_histogram or None,
            "config": result.config,
        }
    )
    result.k_bpw = payload.get("k_bpw")
    result.v_bpw = payload.get("v_bpw")
    result.avg_bpw = payload.get("avg_bpw")
    result.effective_bpw = payload.get("effective_bpw")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "dynamic",
            "block_tq",
            "block_tq_mix",
            "block_skvq",
            "block_skvq_mix",
            "block_tq_pure",
            "block_tq_pure_mix",
            "v2_paper",
            "v3_flat",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--text-encoding", default="utf-8")
    parser.add_argument("--inline-text", default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--record-attentions", action="store_true")
    parser.add_argument("--output", default=None)

    parser.add_argument("--policy", choices=["token", "window", "hybrid"], default="hybrid")
    parser.add_argument(
        "--residual-window",
        type=int,
        default=128,
        help="TurboQuant V3 flat only: recent FP16 tail length (author default 128)",
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = _load_texts(args)
    input_ids = _tokenize_corpus(tokenizer, texts, args.max_tokens)
    print(f"Evaluation tokens: {input_ids.shape[1]}")

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

    results = []
    for backend in _selected_backends(args.backend):
        print(f"\n=== {backend} ===")
        result = evaluate_backend(model, input_ids, args, backend)
        results.append(result)
        print(
            f"ppl={result.ppl:.4f} loss={result.loss:.4f} "
            f"tokens={result.tokens} seconds={result.seconds:.2f}"
        )
        if result.avg_compression_ratio is not None:
            print(
                f"avg_ratio={result.avg_compression_ratio:.3f} "
                f"avg_compressed_blocks={result.avg_compressed_blocks:.1f} "
                f"avg_fp16_blocks={result.avg_fp16_blocks:.1f}"
            )
        if result.avg_bpw is not None:
            eff = (
                f"{result.effective_bpw:.3f}"
                if result.effective_bpw is not None
                else "n/a"
            )
            print(
                f"bpw K={result.k_bpw:.2f} V={result.v_bpw:.2f} "
                f"avg={result.avg_bpw:.2f} effective={eff}"
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        print(f"\nWrote JSONL: {out_path}")


if __name__ == "__main__":
    main()
