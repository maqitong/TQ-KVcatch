from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import difflib
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model
from run_llama32_ablation import build_manager, detach_quantizer


TASKS = ["passage_retrieval_en", "lcc", "gov_report"]
METHODS = ["fp16", "skvq_baseline", "tq_replace", "tq_hybrid", "tq_asym_protect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official LongBench subset for Llama3.2 SKVQ+TurboQuant")
    parser.add_argument("--model-path", default=r"D:\model\Llama3.2_3B")
    parser.add_argument("--results-dir", default="experiments/results/llama32_3b_longbench_official")
    parser.add_argument("--reorder-path", default="experiments/results/llama32_3b_wikitext/llama32_3b-wikitext2-n4-len512-g128-minmax-rod.pt")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-gen-cap", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.96)
    parser.add_argument("--protect-layers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def completed_pairs(summary_path: Path) -> set[tuple[str, str]]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {(row["method"], row["task"]) for row in reader}


def completed_detail_rows(details_path: Path) -> dict[tuple[str, str], dict[int, dict]]:
    completed: dict[tuple[str, str], dict[int, dict]] = {}
    if not details_path.exists():
        return completed
    with details_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["method"], row["task"])
            completed.setdefault(key, {})[int(row["sample_idx"])] = row
    return completed


def build_chat(prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def truncate_prompt(tokenizer, prompt: str, max_input_tokens: int) -> tuple[str, int, bool]:
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original_len = int(ids.numel())
    if original_len <= max_input_tokens:
        return prompt, original_len, False
    half = max_input_tokens // 2
    prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return prompt, original_len, True


def score_one(dataset: str, prediction: str, answers: list[str], all_classes: list[str]) -> float:
    score = 0.0
    for answer in answers:
        if dataset == "passage_retrieval_en":
            score = max(score, retrieval_score(prediction, answer))
        elif dataset == "lcc":
            score = max(score, code_sim_score(prediction, answer))
        elif dataset == "gov_report":
            score = max(score, rouge_l_score(prediction, answer))
        else:
            raise ValueError(f"metric not implemented for {dataset}")
    return float(score)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if number == ground_truth_id) / len(numbers)


def code_sim_score(prediction: str, ground_truth: str) -> float:
    selected = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            selected = line
            break
    return difflib.SequenceMatcher(None, selected, ground_truth).ratio()


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    m, n = len(pred_tokens), len(truth_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == truth_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clear_method(model) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


def attach_method(model, args: argparse.Namespace, method: str, reorder_path: Path):
    detach_quantizer(model)
    if method == "fp16":
        return
    method_reorder = None if method == "tq_replace" else reorder_path
    manager = build_manager(model, args, method, method_reorder)
    plug_quantizer_into_model(model, manager)


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float, int, str]:
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        elapsed = time.perf_counter() - start
        pred = tokenizer.decode(output[inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        clear_method(model)
        return pred, elapsed, int(output.shape[-1] - inputs.input_ids.shape[-1]), "ok"
    except torch.cuda.OutOfMemoryError as exc:
        clear_method(model)
        torch.cuda.empty_cache()
        gc.collect()
        return f"ERROR: {str(exc).splitlines()[0]}", 0.0, 0, "oom"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reorder_path = Path(args.reorder_path)

    prompts = load_json("longbench_config/dataset2prompt.json")
    maxlens = load_json("longbench_config/dataset2maxlen.json")
    tasks = split_csv(args.tasks)
    methods = split_csv(args.methods)

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

    summary_path = results_dir / f"summary_n{args.samples_per_task}_ctx{args.max_input_tokens}_gen{args.max_gen_cap}.csv"
    details_path = results_dir / f"details_n{args.samples_per_task}_ctx{args.max_input_tokens}_gen{args.max_gen_cap}.csv"
    done = completed_pairs(summary_path) if args.resume else set()
    done_details = completed_detail_rows(details_path) if args.resume else {}

    for method in methods:
        print(f"[method] {method}")
        attach_method(model, args, method, reorder_path)
        for task in tasks:
            if (method, task) in done:
                print(f"[skip] {method}/{task} already complete")
                continue
            print(f"[task] {method}/{task}")
            data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            n = min(args.samples_per_task, len(data))
            task_scores = []
            task_elapsed = 0.0
            task_new_tokens = 0
            task_oom = 0
            max_new_tokens = min(int(maxlens[task]), args.max_gen_cap)
            existing_samples = done_details.get((method, task), {})
            for row in existing_samples.values():
                task_scores.append(float(row["score"]) / 100.0)
                task_elapsed += float(row["elapsed_sec"])
                task_new_tokens += int(row["new_tokens"])
                task_oom += int(row["status"] != "ok")
            for sample_idx in range(n):
                if sample_idx in existing_samples:
                    continue
                sample = data[sample_idx]
                raw_prompt = prompts[task].format(**sample)
                prompt, original_tokens, truncated = truncate_prompt(tokenizer, raw_prompt, args.max_input_tokens)
                if task not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    prompt = build_chat(prompt)
                pred, elapsed, new_tokens, status = generate_one(model, tokenizer, prompt, max_new_tokens)
                score = 0.0 if status != "ok" else score_one(task, pred, sample["answers"], sample["all_classes"])
                task_scores.append(score)
                task_elapsed += elapsed
                task_new_tokens += new_tokens
                task_oom += int(status != "ok")
                append_csv(
                    details_path,
                    {
                        "method": method,
                        "task": task,
                        "sample_idx": sample_idx,
                        "score": f"{100 * score:.4f}",
                        "status": status,
                        "elapsed_sec": f"{elapsed:.2f}",
                        "new_tokens": new_tokens,
                        "original_prompt_tokens": original_tokens,
                        "truncated": int(truncated),
                        "prediction": pred.replace("\n", "\\n")[:500],
                        "answers": json.dumps(sample["answers"], ensure_ascii=False),
                    },
                )
                torch.cuda.empty_cache()
            avg = 100 * sum(task_scores) / len(task_scores) if task_scores else 0.0
            append_csv(
                summary_path,
                {
                    "method": method,
                    "task": task,
                    "samples": n,
                    "score": f"{avg:.4f}",
                    "oom": task_oom,
                    "elapsed_sec": f"{task_elapsed:.2f}",
                    "new_tokens": task_new_tokens,
                    "max_input_tokens": args.max_input_tokens,
                    "max_gen": max_new_tokens,
                },
            )
            print(f"[score] {method}/{task}: {avg:.4f} over {n}")
        detach_quantizer(model)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {details_path}")


if __name__ == "__main__":
    main()
