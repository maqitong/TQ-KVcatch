from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model
from run_llama32_ablation import build_manager, detach_quantizer, ensure_reorder


METHODS = ["fp16", "skvq_baseline", "tq_replace", "tq_hybrid", "tq_asym_protect"]


LONGBENCH_LITE = [
    {
        "task": "passage_retrieval_en",
        "prompt": (
            "You are given five passages.\n"
            "Passage 1: The museum opened a new wing about ocean navigation.\n"
            "Passage 2: A chef described how to ferment black beans.\n"
            "Passage 3: The city council approved the North River bridge repair budget on Tuesday.\n"
            "Passage 4: A software team migrated an API from REST to GraphQL.\n"
            "Passage 5: Researchers observed a comet before sunrise.\n\n"
            "Question: Which passage mentions the North River bridge repair budget? Answer with only the passage number."
        ),
        "answer": "3",
        "max_new_tokens": 8,
    },
    {
        "task": "lcc",
        "prompt": (
            "Complete the following Python function. Return only code after the cursor.\n\n"
            "def count_even(numbers):\n"
            "    count = 0\n"
            "    for n in numbers:\n"
            "        if n % 2 == 0:\n"
            "            count += 1\n"
            "    "
        ),
        "answer": "return count",
        "max_new_tokens": 24,
    },
    {
        "task": "gov_report",
        "prompt": (
            "Summarize the report in one sentence.\n\n"
            "Report: The transportation department inspected 42 rural bridges after heavy spring flooding. "
            "Seven bridges require immediate lane closures, while fifteen need minor repairs before winter. "
            "The department requests emergency funding and will publish weekly repair progress updates."
        ),
        "answer": "transportation department inspected rural bridges after flooding and requested emergency funding for repairs",
        "max_new_tokens": 48,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extra local metrics for Llama3.2 SKVQ+TurboQuant")
    parser.add_argument("--model-path", default=r"D:\model\Llama3.2_3B")
    parser.add_argument("--results-dir", default="experiments/results/llama32_3b_extra")
    parser.add_argument("--reorder-dir", default="experiments/results/llama32_3b_formal")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--calib-samples", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.96)
    parser.add_argument("--protect-layers", type=int, default=4)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--needle-methods", default="fp16,skvq_baseline,tq_asym_protect")
    parser.add_argument("--needle-contexts", default="4096,8192,16384")
    parser.add_argument("--needle-depths", default="25,50,75")
    parser.add_argument("--latency-prompt-tokens", type=int, default=512)
    parser.add_argument("--latency-new-tokens", type=int, default=32)
    parser.add_argument(
        "--metrics",
        default="longbench,needle,compression,latency",
        help="Comma-separated subset: longbench, needle, compression, latency",
    )
    parser.add_argument("--force-recalib", action="store_true")
    return parser.parse_args()


def split_csv(value: str, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def attach_method(model, args: argparse.Namespace, method: str, reorder_path: Path):
    detach_quantizer(model)
    if method == "fp16":
        return None
    manager = build_manager(model, args, method, reorder_path if method != "tq_replace" else None)
    plug_quantizer_into_model(model, manager)
    return manager


def clear_method(model):
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    new_tokens = int(out.shape[-1] - inputs.input_ids.shape[-1])
    text = tokenizer.decode(out[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    clear_method(model)
    return text.strip(), elapsed, new_tokens


def generate_text_or_error(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float, int, str]:
    try:
        pred, elapsed, new_tokens = generate_text(model, tokenizer, prompt, max_new_tokens)
        return pred, elapsed, new_tokens, "ok"
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "cuda error" not in message:
            raise
        clear_method(model)
        torch.cuda.empty_cache()
        gc.collect()
        return f"ERROR: {str(exc).splitlines()[0]}", 0.0, 0, "oom"


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_longbench(task: str, pred: str, answer: str) -> float:
    pred_norm = norm_text(pred)
    answer_norm = norm_text(answer)
    if task == "passage_retrieval_en":
        return 1.0 if re.search(r"\b3\b|passage\s*3", pred_norm) else 0.0
    if task == "lcc":
        return 1.0 if "return" in pred_norm and "count" in pred_norm else 0.0
    pred_tokens = set(re.findall(r"[a-z]+", pred_norm))
    answer_tokens = set(re.findall(r"[a-z]+", answer_norm))
    if not answer_tokens:
        return 0.0
    return len(pred_tokens & answer_tokens) / len(answer_tokens)


def make_needle_prompt(tokenizer, context_tokens: int, depth: int) -> str:
    needle = "The secret project code is AURORA-7749."
    filler = (
        "This archival report describes routine municipal planning, budget notes, weather observations, "
        "meeting minutes, and maintenance logs. "
    )
    question = "\n\nQuestion: What is the secret project code? Answer with only the code."
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    needle_ids = tokenizer(" " + needle + " ", add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    available = max(context_tokens - len(needle_ids) - len(question_ids), 32)
    repeated = (filler_ids * (available // len(filler_ids) + 2))[:available]
    insert_at = int(len(repeated) * depth / 100)
    context_ids = repeated[:insert_at] + needle_ids + repeated[insert_at:]
    context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return f"{context}{question}"


def estimate_kv_bytes(args: argparse.Namespace, method: str, context_tokens: int, cfg) -> tuple[int, int, float]:
    layers = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_hidden = kv_heads * head_dim
    fp16_per_token_layer = 2 * kv_hidden * 2
    fp16_total = context_tokens * layers * fp16_per_token_layer
    fp16_tokens = min(context_tokens, args.sink + args.window_size)
    quant_tokens = max(context_tokens - fp16_tokens, 0)

    if method == "fp16":
        quant_total = fp16_total
    else:
        total = fp16_tokens * layers * fp16_per_token_layer
        groups = math.ceil(kv_hidden / args.group_size)
        for layer_idx in range(layers):
            protected = (
                method == "tq_asym_protect"
                and (layer_idx < args.protect_layers or layer_idx >= layers - args.protect_layers)
            )
            if protected:
                kbits, vbits = 8, 8
            elif method == "tq_asym_protect":
                kbits, vbits = 4, 2
            else:
                kbits, vbits = 4, 4
            data_bytes = quant_tokens * kv_hidden * (kbits + vbits) / 8
            if method == "skvq_baseline":
                overhead = quant_tokens * groups * 4
            else:
                overhead = quant_tokens * kv_heads * 2 * 2
            total += data_bytes + overhead
        quant_total = int(total)

    ratio = fp16_total / quant_total if quant_total else 0.0
    return int(fp16_total), int(quant_total), ratio


def run_longbench_lite(model, tokenizer, args, methods, reorder_path, out_path: Path):
    rows = []
    for method in methods:
        print(f"[longbench-lite] {method}")
        attach_method(model, args, method, reorder_path)
        for sample in LONGBENCH_LITE:
            pred, elapsed, new_tokens = generate_text(model, tokenizer, sample["prompt"], sample["max_new_tokens"])
            rows.append(
                {
                    "method": method,
                    "task": sample["task"],
                    "score": f"{score_longbench(sample['task'], pred, sample['answer']):.4f}",
                    "elapsed_sec": f"{elapsed:.2f}",
                    "new_tokens": new_tokens,
                    "prediction": pred.replace("\n", "\\n"),
                }
            )
        detach_quantizer(model)
        torch.cuda.empty_cache()
        gc.collect()
    write_csv(out_path, rows)


def run_needle_lite(model, tokenizer, args, methods, reorder_path, out_path: Path):
    rows = []
    contexts = split_csv(args.needle_contexts, int)
    depths = split_csv(args.needle_depths, int)
    for method in methods:
        print(f"[needle-lite] {method}")
        attach_method(model, args, method, reorder_path)
        for ctx in contexts:
            for depth in depths:
                prompt = make_needle_prompt(tokenizer, ctx, depth)
                pred, elapsed, new_tokens, status = generate_text_or_error(model, tokenizer, prompt, 24)
                hit = int("aurora-7749" in norm_text(pred))
                rows.append(
                    {
                        "method": method,
                        "context_tokens": ctx,
                        "depth": depth,
                        "hit": hit,
                        "status": status,
                        "elapsed_sec": f"{elapsed:.2f}",
                        "new_tokens": new_tokens,
                        "prediction": pred.replace("\n", "\\n"),
                    }
                )
        detach_quantizer(model)
        torch.cuda.empty_cache()
        gc.collect()
    write_csv(out_path, rows)


def run_compression(args, methods, cfg, out_path: Path):
    rows = []
    for method in methods:
        for ctx in [512, 4096, 8192, 16384]:
            fp16_bytes, quant_bytes, ratio = estimate_kv_bytes(args, method, ctx, cfg)
            rows.append(
                {
                    "method": method,
                    "context_tokens": ctx,
                    "fp16_kv_mb": f"{fp16_bytes / 1024**2:.2f}",
                    "estimated_quant_kv_mb": f"{quant_bytes / 1024**2:.2f}",
                    "compression_ratio": f"{ratio:.3f}",
                }
            )
    write_csv(out_path, rows)


def run_latency(model, tokenizer, args, methods, reorder_path, out_path: Path):
    rows = []
    prompt = make_needle_prompt(tokenizer, args.latency_prompt_tokens, 50)
    for method in methods:
        print(f"[latency] {method}")
        attach_method(model, args, method, reorder_path)
        torch.cuda.reset_peak_memory_stats()
        pred, elapsed, new_tokens = generate_text(model, tokenizer, prompt, args.latency_new_tokens)
        rows.append(
            {
                "method": method,
                "prompt_tokens": args.latency_prompt_tokens,
                "new_tokens": new_tokens,
                "elapsed_sec": f"{elapsed:.2f}",
                "tokens_per_sec": f"{new_tokens / elapsed:.4f}" if elapsed else "0",
                "peak_allocated_gb": f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}" if torch.cuda.is_available() else "0",
                "prediction_prefix": pred[:80].replace("\n", "\\n"),
            }
        )
        detach_quantizer(model)
        torch.cuda.empty_cache()
        gc.collect()
    write_csv(out_path, rows)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[write] skipped empty {path}")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    args.results_dir = str(Path(args.results_dir))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reorder_dir = Path(args.reorder_dir)
    reorder_path = reorder_dir / f"llama32_3b-local-n{args.calib_samples}-len{args.seq_len}-g{args.group_size}-minmax-rod.pt"

    methods = split_csv(args.methods, str)
    needle_methods = split_csv(args.needle_methods, str)
    metrics = set(split_csv(args.metrics, str))

    if metrics == {"compression"}:
        cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
        run_compression(args, methods, cfg, results_dir / "compression_estimates.csv")
        print("[done] compression metrics complete")
        return

    print("[load] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True, use_fast=False)
    print("[load] model")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"[load] device_map={getattr(model, 'hf_device_map', None)}")

    ensure_reorder(model, tokenizer, args, reorder_path)

    if "longbench" in metrics:
        run_longbench_lite(model, tokenizer, args, methods, reorder_path, results_dir / "longbench_lite.csv")
    if "needle" in metrics:
        run_needle_lite(model, tokenizer, args, needle_methods, reorder_path, results_dir / "needle_lite.csv")
    if "compression" in metrics:
        run_compression(args, methods, model.config, results_dir / "compression_estimates.csv")
    if "latency" in metrics:
        run_latency(model, tokenizer, args, methods, reorder_path, results_dir / "latency_decode.csv")
    print("[done] extra metrics complete")


if __name__ == "__main__":
    main()
