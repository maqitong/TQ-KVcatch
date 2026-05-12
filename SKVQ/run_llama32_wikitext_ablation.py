from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model
from run_llama32_ablation import build_manager, collect_minmax_stats, build_reorder_cache, detach_quantizer


METHODS = ["fp16", "skvq_baseline", "tq_replace", "tq_hybrid", "tq_asym_protect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiText2 PPL ablation for Llama3.2-3B SKVQ+TurboQuant")
    parser.add_argument("--model-path", default=r"D:\model\Llama3.2_3B")
    parser.add_argument("--results-dir", default="experiments/results/llama32_3b_wikitext")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--calib-samples", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.96)
    parser.add_argument("--protect-layers", type=int, default=4)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--force-recalib", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_wikitext_tokens(tokenizer, split: str) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids


def make_token_chunks(tokens: torch.Tensor, seq_len: int, nsamples: int, offset: int = 0) -> list[torch.Tensor]:
    needed = offset + nsamples * seq_len
    if tokens.shape[1] < needed:
        raise ValueError(f"not enough tokens: need {needed}, got {tokens.shape[1]}")
    return [
        tokens[:, offset + i * seq_len : offset + (i + 1) * seq_len].contiguous()
        for i in range(nsamples)
    ]


def ensure_wikitext_reorder(model, tokenizer, args: argparse.Namespace, reorder_path: Path) -> None:
    if reorder_path.exists() and not args.force_recalib:
        print(f"[calib] reuse {reorder_path}")
        return
    print("[data] load WikiText2 train for calibration")
    train_tokens = load_wikitext_tokens(tokenizer, "train")
    chunks = make_token_chunks(train_tokens, args.seq_len, args.calib_samples, offset=0)
    print(f"[calib] collect minmax stats from {args.calib_samples} WikiText2 chunks")
    stats = collect_minmax_stats(model, chunks)
    n_clusters = max(1, model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads) // args.group_size)
    build_reorder_cache(stats, n_clusters, reorder_path)
    print(f"[calib] saved {reorder_path}")


def clear_quantizer(model) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


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


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reorder_path = results_dir / f"llama32_3b-wikitext2-n{args.calib_samples}-len{args.seq_len}-g{args.group_size}-minmax-rod.pt"
    csv_path = results_dir / f"wikitext2_len{args.seq_len}_eval{args.eval_samples}.csv"

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

    ensure_wikitext_reorder(model, tokenizer, args, reorder_path)

    print("[data] load WikiText2 test")
    test_tokens = load_wikitext_tokens(tokenizer, "test")
    eval_chunks = make_token_chunks(test_tokens, args.seq_len, args.eval_samples, offset=args.seq_len)
    methods = split_csv(args.methods)

    rows = []
    for method in methods:
        print(f"[eval] {method}")
        detach_quantizer(model)
        if method != "fp16":
            method_reorder = None if method == "tq_replace" else reorder_path
            manager = build_manager(model, args, method, method_reorder)
            plug_quantizer_into_model(model, manager)
            print(manager)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        ppl = eval_ppl(model, eval_chunks)
        elapsed = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        rows.append(
            {
                "method": method,
                "ppl": f"{ppl:.6f}",
                "elapsed_sec": f"{elapsed:.2f}",
                "peak_allocated_gb": f"{peak_gb:.3f}",
                "seq_len": args.seq_len,
                "eval_samples": args.eval_samples,
                "calib_samples": args.calib_samples,
                "dataset": "wikitext-2-raw-v1",
                "group_size": args.group_size,
                "window_size": args.window_size,
                "sink": args.sink,
                "clip": args.clip,
            }
        )
        print(f"[eval] {method}: ppl={ppl:.6f}, elapsed={elapsed:.2f}s, peak={peak_gb:.3f}GB")
        detach_quantizer(model)
        gc.collect()
        torch.cuda.empty_cache()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
