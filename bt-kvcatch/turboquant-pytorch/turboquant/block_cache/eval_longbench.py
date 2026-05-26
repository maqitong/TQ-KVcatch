"""LongBench-style QA evaluation for KVcatch cache backends.

The script targets the single-document QA subsets used in the completion plan:
`narrativeqa`, `qasper`, and `multifieldqa_en`. It also provides `--toy-sample`
so the end-to-end path can be smoke-tested without downloading datasets.

Examples:
    python -m turboquant.block_cache.eval_longbench \
        --model D:\model\Llama3.2_3B --local-files-only \
        --toy-sample --backend block_tq_mix

    python -m turboquant.block_cache.eval_longbench \
        --model Qwen/Qwen2.5-3B-Instruct --backend all \
        --subsets narrativeqa,qasper,multifieldqa_en --max-samples 16
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from turboquant.block_cache.methods import (
    LONGBENCH_ALL_BACKENDS,
    backend_config as _shared_backend_config,
    build_policy as _shared_build_policy,
    cache_factory_for_backend,
    paper_tq_pure_policy as _shared_paper_tq_pure_policy,
    parse_backend_selection,
)


@dataclass
class LongBenchExample:
    subset: str
    context: str
    question: str
    answers: list[str]


@dataclass
class LongBenchResult:
    backend: str
    model: str
    subset: str
    sample_idx: int
    score: float
    prediction: str
    answers: list[str]
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


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


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
    return parse_backend_selection(name, all_backends=LONGBENCH_ALL_BACKENDS)


def _build_policy(args):
    return _shared_build_policy(args, window_uses_sink=False)


def _paper_tq_pure_policy():
    return _shared_paper_tq_pure_policy()


def _backend_config(args, backend: str) -> dict:
    return _shared_backend_config(args, backend)


def _cache_factory(args, backend: str) -> Callable:
    return cache_factory_for_backend(
        args,
        backend,
        include_v3=True,
        include_skvq_native=True,
        window_uses_sink=False,
        v3_protected_layers=0,
    )


def _normalize_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [tok for tok in text.split() if tok]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: str, answer: str) -> float:
    pred_tokens = _normalize_text(prediction)
    answer_tokens = _normalize_text(answer)
    lcs = _lcs_len(pred_tokens, answer_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / max(len(pred_tokens), 1)
    recall = lcs / max(len(answer_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def score_prediction(prediction: str, answers: Iterable[str]) -> float:
    return max((rouge_l_f1(prediction, answer) for answer in answers), default=0.0)


def build_prompt(example: LongBenchExample) -> str:
    return (
        "You are given a long document and a question. "
        "Answer the question as concisely as possible using only the document.\n\n"
        f"Document:\n{example.context}\n\n"
        f"Question: {example.question}\n"
        "Answer:"
    )


def _toy_examples() -> list[LongBenchExample]:
    return [
        LongBenchExample(
            subset="toy",
            context=(
                "The project archive describes several unrelated initiatives. "
                "One paragraph mentions office repairs, another mentions onboarding. "
                "The relevant note says that the secret launch city is Hangzhou. "
                "The remaining paragraphs discuss budgets and meeting schedules."
            ),
            question="What is the secret launch city?",
            answers=["Hangzhou"],
        )
    ]


def _row_to_example(subset: str, row: dict) -> LongBenchExample:
    context = str(row.get("context", ""))
    question = str(row.get("input", row.get("question", "")))
    answers = row.get("answers", row.get("answer", []))
    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []
    else:
        answers = [str(answer) for answer in answers]
    return LongBenchExample(
        subset=subset,
        context=context,
        question=question,
        answers=answers,
    )


def load_examples(args) -> list[LongBenchExample]:
    if args.toy_sample:
        return _toy_examples()

    if args.input_jsonl:
        examples = []
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                examples.append(_row_to_example(str(row.get("subset", "jsonl")), row))
        return examples[: args.max_samples] if args.max_samples else examples

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required unless --toy-sample or --input-jsonl is provided"
        ) from exc

    examples: list[LongBenchExample] = []
    for subset in _parse_csv(args.subsets):
        ds = load_dataset(args.dataset, subset, split=args.split)
        limit = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
        for idx in range(limit):
            examples.append(_row_to_example(subset, ds[idx]))
    return examples


@torch.no_grad()
def run_example(model, tokenizer, args, backend: str, example: LongBenchExample, sample_idx: int) -> LongBenchResult:
    prompt = build_prompt(example)
    device = _model_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(device)
    if not args.pass_attention_mask:
        inputs.pop("attention_mask", None)

    cache = _cache_factory(args, backend)()
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
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    score = score_prediction(prediction, example.answers)
    report = cache.memory_report() if cache is not None else None

    return LongBenchResult(
        backend=backend,
        model=args.model,
        subset=example.subset,
        sample_idx=sample_idx,
        score=score,
        prediction=prediction,
        answers=example.answers,
        input_tokens=int(inputs.input_ids.shape[1]),
        output_tokens=int(new_tokens.shape[0]),
        seconds=seconds,
        compression_ratio=report["compression_ratio"] if report else None,
        n_compressed_blocks=report["n_compressed_blocks"] if report else None,
        n_fp16_blocks=report["n_fp16_blocks"] if report else None,
        bit_histogram=report["bit_histogram"] if report else None,
        precision_histogram=report["precision_histogram"] if report else None,
        config=_backend_config(args, backend),
    )


@torch.no_grad()
def run_example_skvq_native(
    model,
    tokenizer,
    args,
    example: LongBenchExample,
    sample_idx: int,
) -> LongBenchResult:
    from turboquant.block_cache.skvq_native_integration import (
        build_skvq_baseline_manager,
        clear_quantizer,
        detach_quantizer,
        plug_quantizer,
    )

    if args.reorder_file is None:
        raise ValueError("skvq_native requires --reorder-file")

    import gc

    skvq_budget = getattr(args, "skvq_max_input_tokens", None) or args.max_input_tokens
    prompt = build_prompt(example)
    device = _model_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=skvq_budget,
    ).to(device)
    if not args.pass_attention_mask:
        inputs.pop("attention_mask", None)

    config = _backend_config(args, "skvq_native")
    config["max_input_tokens"] = skvq_budget

    detach_quantizer(model)
    manager = build_skvq_baseline_manager(
        model,
        reorder_file=args.reorder_file,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
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
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        score = score_prediction(prediction, example.answers)
        output_tokens = int(new_tokens.shape[0])
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        error_msg = str(exc).splitlines()[0][:500]
        prediction = ""
        score = 0.0
        output_tokens = 0
    finally:
        seconds = time.perf_counter() - started
        clear_quantizer(model)
        detach_quantizer(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if oom:
        config["status"] = "oom"
        config["error"] = error_msg

    return LongBenchResult(
        backend="skvq_native",
        model=args.model,
        subset=example.subset,
        sample_idx=sample_idx,
        score=score,
        prediction=prediction,
        answers=example.answers,
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


def _run_key(backend: str, subset: str, sample_idx: int) -> str:
    return f"{backend}|{subset}|{sample_idx}"


def _parse_resume_log(log_path: Path) -> set[str]:
    """Parse `=== backend=... subset=... sample=N ===` lines from a log file."""
    pattern = re.compile(
        r"^=== backend=(\S+) subset=(\S+) sample=(\d+) ===$"
    )
    completed: set[str] = set()
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line.strip())
        if m:
            completed.add(_run_key(m.group(1), m.group(2), int(m.group(3))))
    return completed


def _load_results_jsonl(jsonl_path: Path) -> tuple[list[LongBenchResult], set[str]]:
    results: list[LongBenchResult] = []
    completed: set[str] = set()
    if not jsonl_path.exists():
        return results, completed
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        result = LongBenchResult(**row)
        results.append(result)
        completed.add(_run_key(result.backend, result.subset, result.sample_idx))
    return results, completed


def _append_result_jsonl(jsonl_path: Path, result: LongBenchResult) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _write_outputs(results: list[LongBenchResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "longbench_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    csv_path = output_dir / "longbench_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "subset",
                "sample_idx",
                "score",
                "compression_ratio",
                "seconds",
                "input_tokens",
                "prediction",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.backend,
                    r.subset,
                    r.sample_idx,
                    f"{r.score:.6f}",
                    r.compression_ratio,
                    f"{r.seconds:.4f}",
                    r.input_tokens,
                    r.prediction,
                ]
            )

    grouped: dict[tuple[str, str], list[LongBenchResult]] = {}
    for result in results:
        grouped.setdefault((result.backend, result.subset), []).append(result)

    lines = [
        "# LongBench Summary",
        "",
        "| Backend | Subset | Samples | Avg ROUGE-L | Avg Ratio | Avg Seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (backend, subset), group in sorted(grouped.items()):
        avg_score = sum(r.score for r in group) / len(group)
        ratios = [r.compression_ratio for r in group if r.compression_ratio is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else None
        avg_seconds = sum(r.seconds for r in group) / len(group)
        ratio_cell = f"{avg_ratio:.3f}" if avg_ratio is not None else "-"
        lines.append(
            f"| {backend} | {subset} | {len(group)} | "
            f"{avg_score:.4f} | {ratio_cell} | {avg_seconds:.3f} |"
        )

    md_path = output_dir / "longbench_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        default="all",
        help=(
            "Backend name, comma-separated list, or 'all' "
            "(includes block_tq_pure and skvq_native paper baselines)"
        ),
    )
    parser.add_argument("--skvq-root", default=None, help="Path to SKVQ repo for skvq_native backend")
    parser.add_argument("--dataset", default="THUDM/LongBench")
    parser.add_argument("--subsets", default="narrativeqa,qasper,multifieldqa_en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--toy-sample", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument(
        "--skvq-max-input-tokens",
        type=int,
        default=4096,
        help="Input token budget for skvq_native backend (eager attn; 8192 often OOM on 24GB)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--pass-attention-mask", action="store_true")
    parser.add_argument("--record-attentions", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume-log",
        default=None,
        help="Skip (backend, subset, sample) already present in this log file",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Load existing output-dir/longbench_results.jsonl and append new rows",
    )

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
    parser.add_argument(
        "--residual-window",
        type=int,
        default=128,
        help="V3FlatCache FP16 tail length (author default 128; 0 = compress full history)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path("runs") / f"longbench_{stamp}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(args)
    print(f"Loaded examples: {len(examples)}")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "longbench_results.jsonl"

    results: list[LongBenchResult] = []
    completed: set[str] = set()
    if args.append_results:
        results, completed = _load_results_jsonl(jsonl_path)
        print(f"[resume] loaded {len(results)} rows from {jsonl_path}")
    if args.resume_log:
        from_log = _parse_resume_log(Path(args.resume_log))
        completed |= from_log
        print(f"[resume] {len(from_log)} completed keys from log {args.resume_log}")

    skipped = 0
    backends = _selected_backends(args.backend)
    hf_backends = [b for b in backends if b != "skvq_native"]

    for backend in hf_backends:
        for sample_idx, example in enumerate(examples):
            key = _run_key(backend, example.subset, sample_idx)
            if key in completed:
                skipped += 1
                print(
                    f"\n=== backend={backend} subset={example.subset} "
                    f"sample={sample_idx} === [skip resume]"
                )
                continue
            print(f"\n=== backend={backend} subset={example.subset} sample={sample_idx} ===")
            result = run_example(model, tokenizer, args, backend, example, sample_idx)
            results.append(result)
            completed.add(key)
            _append_result_jsonl(jsonl_path, result)
            ratio = (
                f"{result.compression_ratio:.3f}x"
                if result.compression_ratio is not None
                else "-"
            )
            print(
                f"score={result.score:.4f} tokens={result.input_tokens} "
                f"ratio={ratio} seconds={result.seconds:.2f}"
            )
            print(f"prediction={result.prediction[:240]!r}")

    if "skvq_native" in backends:
        import gc

        from turboquant.block_cache.skvq_native_integration import load_skvq_llama

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n=== Loading SKVQ native Llama model ===")
        skvq_model, skvq_tokenizer = load_skvq_llama(
            args.model,
            skvq_root=args.skvq_root,
            dtype=_dtype_from_name(args.dtype),
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        for sample_idx, example in enumerate(examples):
            key = _run_key("skvq_native", example.subset, sample_idx)
            if key in completed:
                skipped += 1
                print(
                    f"\n=== backend=skvq_native subset={example.subset} "
                    f"sample={sample_idx} === [skip resume]"
                )
                continue
            print(f"\n=== backend=skvq_native subset={example.subset} sample={sample_idx} ===")
            result = run_example_skvq_native(
                skvq_model, skvq_tokenizer, args, example, sample_idx
            )
            results.append(result)
            completed.add(key)
            _append_result_jsonl(jsonl_path, result)
            print(
                f"score={result.score:.4f} tokens={result.input_tokens} "
                f"seconds={result.seconds:.2f}"
            )
            print(f"prediction={result.prediction[:240]!r}")
        del skvq_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[resume] skipped {skipped} already-completed runs")
    _write_outputs(results, output_dir)


if __name__ == "__main__":
    main()
