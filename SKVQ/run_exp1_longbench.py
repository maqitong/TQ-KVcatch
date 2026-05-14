"""
Experiment 1: LongBench Evaluation (SKVQ + TurboQuant)

Replicates paper Table 1:
  - Tasks: lcc, repobench-p, passage_retrieval_en, trec, 2wikimqa, gov_report, multifieldqa_zh
  - Methods: fp16, KIVI, skvq_baseline, tq_replace, tq_hybrid
  - Default: k2-v2, group_size=128, window_size=128
  - Optional: add tq_asym_protect via --methods for protected-layer comparison
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import random
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from tqdm import tqdm

from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.modeling_mistral_skvq import MistralForCausalLM
from experiments.utils import plug_quantizer_into_model
from KVcache_manager import ModelKVCacheManager
try:
    from metrics import (
        code_sim_score,
        retrieval_score,
        qa_f1_score,
        qa_f1_zh_score,
        classification_score,
        rouge_score,
    )
    METRICS_IMPORT_ERROR = None
except ImportError as exc:
    code_sim_score = retrieval_score = qa_f1_score = qa_f1_zh_score = classification_score = rouge_score = None
    METRICS_IMPORT_ERROR = exc


MODEL_CLASSES = {
    "llama": LlamaForCausalLM,
    "mistral": MistralForCausalLM,
}

DEFAULT_METHODS = "fp16,KIVI,skvq_baseline,tq_replace,tq_hybrid"
VALID_METHODS = {
    "fp16",
    "KIVI",
    "skvq_baseline",
    "tq_replace",
    "tq_hybrid",
    "tq_asym_protect",
}

PAPER_TASKS = [
    "lcc",
    "repobench-p",
    "passage_retrieval_en",
    "trec",
    "2wikimqa",
    "gov_report",
    "multifieldqa_zh",
]

TASK_METRICS = {
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "passage_retrieval_en": retrieval_score,
    "trec": classification_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_score,
    "multifieldqa_zh": qa_f1_zh_score,
}

CSV_FIELDS = [
    "task", "method", "kbits", "vbits", "status", "score",
    "n_samples", "oom", "elapsed_sec", "peak_allocated_gb",
    "group_size", "window_size", "sink", "clip",
    "protect_layers", "protected_bits", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1: LongBench evaluation")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-family", required=True, choices=["llama", "mistral"])
    p.add_argument("--results-dir", default="experiments/results/exp1_longbench")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--kbits", type=float, default=2)
    p.add_argument("--vbits", type=float, default=2)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--sink", type=int, default=5)
    p.add_argument("--clip", type=float, default=0.96)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--reserve-new-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--calib-seq-len", type=int, default=4096)
    p.add_argument("--tasks", default=",".join(PAPER_TASKS))
    p.add_argument("--n-samples", type=int, default=0, help="0=all")
    p.add_argument("--protect-layers", type=int, default=4)
    p.add_argument("--protected-bits", type=int, default=8)
    p.add_argument("--force-recalib", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def parse_csv_list(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

def build_chat(prompt: str, model_family: str, model_path: str, tokenizer=None) -> str:
    lower = model_path.lower()
    should_chat = (
        "llama-2" in lower
        or "llama2" in lower
        or ("mistral" in lower and "instruct" in lower)
    )
    if should_chat and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    if "llama-2" in lower or "llama2" in lower:
        return f"[INST] {prompt} [/INST]"
    if "mistral" in lower and "instruct" in lower:
        return f"[INST] {prompt} [/INST]"
    return prompt


def truncate_to_budget(tokenizer, prompt: str, max_tokens: int) -> tuple[str, int, bool]:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original_len = int(tokenized.numel())
    if original_len <= max_tokens:
        return prompt, original_len, False
    half = max_tokens // 2
    prompt = (
        tokenizer.decode(tokenized[:half], skip_special_tokens=True)
        + tokenizer.decode(tokenized[-(max_tokens - half):], skip_special_tokens=True)
    )
    return prompt, original_len, True


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_minmax_stats(model, chunks: list[torch.Tensor]) -> dict:
    num_layers = len(model.model.layers)
    stats = {
        "min": {"k": [None] * num_layers, "v": [None] * num_layers},
        "max": {"k": [None] * num_layers, "v": [None] * num_layers},
    }

    def hook(_m, _x, output, ttype: str, li: int):
        flat = output.detach().reshape(-1, output.shape[-1]).float()
        cur_min, cur_max = flat.amin(dim=0).cpu(), flat.amax(dim=0).cpu()
        if stats["min"][ttype][li] is None:
            stats["min"][ttype][li], stats["max"][ttype][li] = cur_min, cur_max
        else:
            stats["min"][ttype][li] = torch.minimum(stats["min"][ttype][li], cur_min)
            stats["max"][ttype][li] = torch.maximum(stats["max"][ttype][li], cur_max)

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            lambda m, x, y, i=li: hook(m, x, y, "k", i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            lambda m, x, y, i=li: hook(m, x, y, "v", i)))

    device = next(model.parameters()).device
    for chunk in chunks:
        model.model(chunk.to(device), use_cache=False)
    for h in handles:
        h.remove()
    return stats


def build_reorder_cache(stats: dict, n_clusters: int, save_path: Path) -> None:
    reorder_indices, cluster_st_inds = [], []
    for li in range(len(stats["min"]["k"])):
        lr, ls = [], []
        for ttype in ("k", "v"):
            feat = torch.stack((stats["min"][ttype][li], stats["max"][ttype][li]), dim=1)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(feat.numpy())
            labels = torch.from_numpy(km.labels_)
            indices = labels.argsort()
            starts = torch.zeros(n_clusters + 1, dtype=torch.int64)
            starts[1:] = labels.bincount(minlength=n_clusters).cumsum(0).to(torch.int64)
            lr.append(indices)
            ls.append(starts)
        reorder_indices.append(tuple(lr))
        cluster_st_inds.append(tuple(ls))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reorder_indices": reorder_indices, "cluster_st_inds": cluster_st_inds}, save_path)


def ensure_reorder(model, tokenizer, args: argparse.Namespace, reorder_path: Path) -> None:
    if reorder_path.exists() and not args.force_recalib:
        print(f"[calib] reuse {reorder_path}")
        return
    print(f"[calib] WikiText-2 train seq_len={args.calib_seq_len} n={args.calib_samples}")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r for r in ds["text"] if r.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    needed = args.calib_samples * args.calib_seq_len
    if ids.shape[1] < needed:
        ids = ids.repeat(1, needed // ids.shape[1] + 2)
    chunks = [ids[:, i * args.calib_seq_len:(i + 1) * args.calib_seq_len] for i in range(args.calib_samples)]
    stats = collect_minmax_stats(model, chunks)
    cfg = model.config
    kv_hidden = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    n_clusters = kv_hidden // args.group_size
    print(f"[calib] kv_hidden={kv_hidden} n_clusters={n_clusters}")
    build_reorder_cache(stats, n_clusters, reorder_path)
    print(f"[calib] saved {reorder_path}")


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

def detach_quantizer(model) -> None:
    for layer in model.model.layers:
        layer.self_attn.KV_cache_manager = None
    model.model.model_kv_manager = None
    model.model_kv_manager = None


def clear_quantizer(model) -> None:
    m = getattr(model.model, "model_kv_manager", None)
    if m is not None:
        m.clear()


def build_manager(model, args, method, kbits, vbits, reorder_path):
    num_layers = len(model.model.layers)
    clipping = [args.clip] * num_layers

    if method == "KIVI":
        return ModelKVCacheManager.create(
            model=model, kbits=kbits, vbits=vbits, gsize=args.group_size,
            reorder_file=None, smooth_file=None, window_size=args.window_size,
            pre_rope=False, clipping=clipping, attn_sink=0,
            full_prefill=True, KIVI_mode=True, fp8=True, fake_quant=True,
        )

    common = dict(
        model=model, kbits=kbits, vbits=vbits, gsize=args.group_size,
        window_size=args.window_size, pre_rope=True, clipping=clipping,
        attn_sink=args.sink, full_prefill=False, fp8=True, fake_quant=True,
    )
    if method == "skvq_baseline":
        return ModelKVCacheManager.create(reorder_file=str(reorder_path), **common)
    if method == "tq_replace":
        return ModelKVCacheManager.create(
            reorder_file=None,
            turboquant_config={"use_reorder": False, "protected_layers": 0, "seed_base": 42},
            **common,
        )
    if method == "tq_hybrid":
        return ModelKVCacheManager.create(
            reorder_file=str(reorder_path),
            turboquant_config={"use_reorder": True, "head_local_reorder": True,
                               "protected_layers": 0, "seed_base": 42},
            **common,
        )
    if method == "tq_asym_protect":
        return ModelKVCacheManager.create(
            reorder_file=str(reorder_path),
            turboquant_config={"use_reorder": True, "head_local_reorder": True,
                               "protected_layers": args.protect_layers,
                               "protected_bits": args.protected_bits, "seed_base": 42},
            **common,
        )
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# LongBench
# ---------------------------------------------------------------------------

def load_longbench_config():
    config_dir = Path(__file__).parent / "longbench_config"
    with open(config_dir / "dataset2prompt.json", "r", encoding="utf-8") as f:
        d2p = json.load(f)
    with open(config_dir / "dataset2maxlen.json", "r", encoding="utf-8") as f:
        d2m = json.load(f)
    return d2p, d2m


@torch.no_grad()
def evaluate_task(model, tokenizer, task, data, args, d2p, d2m, pred_dir):
    prompt_fmt = d2p[task]
    max_gen = d2m[task]
    input_budget = max(1, args.max_len - max_gen) if args.reserve_new_tokens else args.max_len
    device = next(model.parameters()).device
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{task}.jsonl"

    score_fn = TASK_METRICS.get(task, qa_f1_score)
    scores = []
    oom_count = 0

    with open(pred_path, "w", encoding="utf-8") as pred_file:
        for idx, sample in enumerate(tqdm(data, desc=task)):
            raw_prompt = prompt_fmt.format(**sample)
            prompt = raw_prompt

            if task not in ("trec", "triviaqa", "samsum", "lcc", "repobench-p"):
                prompt = build_chat(prompt, args.model_family, args.model_path, tokenizer)
            prompt, original_prompt_tokens, truncated = truncate_to_budget(tokenizer, prompt, input_budget)

            inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=input_budget,
                return_tensors="pt",
            ).to(device)
            sample_status = "ok"
            sample_error = ""
            try:
                output = model.generate(
                    **inputs, max_new_tokens=max_gen, num_beams=1,
                    do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id,
                )[0]
                ctx_len = inputs.input_ids.shape[-1]
                pred = tokenizer.decode(output[ctx_len:], skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError as exc:
                print(f"  [oom] {task} #{idx}")
                oom_count += 1
                sample_status = "oom"
                sample_error = str(exc).splitlines()[0][:500]
                pred = ""
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message and "cuda error" not in message:
                    raise
                print(f"  [oom] {task} #{idx}: {str(exc).splitlines()[0]}")
                oom_count += 1
                sample_status = "oom"
                sample_error = str(exc).splitlines()[0][:500]
                pred = ""
            clear_quantizer(model)
            torch.cuda.empty_cache()

            answers = sample.get("answers", sample.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not pred or not answers:
                sample_score = 0.0
            else:
                kwargs = {}
                if score_fn is classification_score:
                    ac = sample.get("all_classes")
                    if ac:
                        kwargs["all_classes"] = ac
                sample_score = max(score_fn(pred, a, **kwargs) for a in answers)
            scores.append(sample_score)

            pred_file.write(
                json.dumps(
                    {
                        "pred": pred,
                        "answers": answers,
                        "all_classes": sample.get("all_classes", []),
                        "length": sample.get("length", 0),
                        "score": sample_score,
                        "status": sample_status,
                        "error": sample_error,
                        "original_prompt_tokens": original_prompt_tokens,
                        "prompt_tokens": int(inputs.input_ids.shape[-1]),
                        "truncated": int(truncated),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pred_file.flush()

    return (100.0 * float(np.mean(scores)) if scores else 0.0), oom_count


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def combo_key(row):
    return (
        row["task"], row["method"], row["kbits"], row["vbits"],
        str(row.get("protect_layers", "")), str(row.get("protected_bits", "")),
    )


def read_existing(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [{c: r.get(c, "") for c in CSV_FIELDS} for r in csv.DictReader(f)]


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def make_row(args, task, method):
    return {
        "task": task, "method": method,
        "kbits": str(args.kbits), "vbits": str(args.vbits),
        "status": "ok", "score": "", "n_samples": "",
        "oom": "", "elapsed_sec": "", "peak_allocated_gb": "",
        "group_size": args.group_size, "window_size": args.window_size,
        "sink": args.sink, "clip": args.clip,
        "protect_layers": args.protect_layers if method == "tq_asym_protect" else "",
        "protected_bits": args.protected_bits if method == "tq_asym_protect" else "",
        "error": "",
    }


def write_summary(results_dir, rows, methods, tasks):
    out = results_dir / "summary_longbench.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + tasks + ["avg"])
        w.writeheader()
        for method in methods:
            ms = []
            d = {"method": method}
            for task in tasks:
                hit = next(
                    (
                        r for r in rows
                        if r["task"] == task
                        and r["method"] == method
                        and r.get("score")
                        and r.get("status") not in {"error", "oom"}
                    ),
                    None,
                )
                d[task] = hit["score"] if hit else ""
                if hit and hit["score"]:
                    ms.append(float(hit["score"]))
            d["avg"] = f"{np.mean(ms):.2f}" if ms else ""
            w.writerow(d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    if METRICS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "LongBench metrics dependencies are missing. "
            "Install requirements.txt or run: pip install jieba rouge fuzzywuzzy"
        ) from METRICS_IMPORT_ERROR

    seed_all()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_csv_list(args.methods)
    tasks = parse_csv_list(args.tasks)
    unknown_methods = sorted(set(methods) - VALID_METHODS)
    if unknown_methods:
        raise ValueError(f"unknown methods: {','.join(unknown_methods)}")

    model_cls = MODEL_CLASSES[args.model_family]
    csv_path = results_dir / "exp1_longbench.csv"

    if not args.resume and csv_path.exists():
        csv_path.unlink()

    existing = read_existing(csv_path) if args.resume else []
    done = {
        combo_key(r)
        for r in existing
        if r.get("status") in {"ok", "error", "oom", "partial_oom"}
    }
    if args.resume:
        print(f"[resume] {len(existing)} rows")

    d2p, d2m = load_longbench_config()
    unknown_tasks = [t for t in tasks if t not in d2p or t not in d2m]
    if unknown_tasks:
        raise ValueError(f"missing LongBench config for tasks: {','.join(unknown_tasks)}")

    print(f"[load] tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    print(f"[load] {model_cls.__name__} from {args.model_path}")
    model = model_cls.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()

    reorder_path = results_dir / f"reorder-wikitext2-g{args.group_size}-n{args.calib_samples}-len{args.calib_seq_len}-minmax.pt"
    if any(m in methods for m in ("skvq_baseline", "tq_hybrid", "tq_asym_protect")):
        ensure_reorder(model, tokenizer, args, reorder_path)

    new_rows = []

    for method in methods:
        detach_quantizer(model)

        if method != "fp16":
            print(f"\n[quant] {method} k{args.kbits}-v{args.vbits}")
            mgr = build_manager(model, args, method, args.kbits, args.vbits, reorder_path)
            plug_quantizer_into_model(model, mgr)
            print(mgr)

        pred_dir = results_dir / f"pred_{method}_k{args.kbits}-v{args.vbits}"

        for task in tasks:
            row = make_row(args, task, method)
            key = combo_key(row)
            if key in done:
                print(f"[skip] {task} {method}")
                continue

            print(f"\n{'='*60}")
            print(f"[eval] {task} | {method} | k{args.kbits}-v{args.vbits}")
            print(f"{'='*60}")

            try:
                data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
                if args.n_samples > 0:
                    data = data.select(range(min(args.n_samples, len(data))))
                row["n_samples"] = len(data)
            except Exception as e:
                row["status"] = "error"
                row["error"] = str(e).splitlines()[0][:500]
                append_row(csv_path, row)
                new_rows.append(row)
                done.add(key)
                continue

            t0 = time.perf_counter()
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                avg, oom_count = evaluate_task(model, tokenizer, task, data, args, d2p, d2m, pred_dir)
                elapsed = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                row["score"] = f"{avg:.2f}"
                row["oom"] = str(oom_count)
                if oom_count >= len(data):
                    row["status"] = "oom"
                elif oom_count > 0:
                    row["status"] = "partial_oom"
                row["elapsed_sec"] = f"{elapsed:.2f}"
                row["peak_allocated_gb"] = f"{peak:.3f}"
                oom_msg = f", oom={oom_count}" if oom_count else ""
                print(f"[result] {task} {method}: {avg:.2f}{oom_msg} ({elapsed:.0f}s)")
            except Exception as e:
                row["status"] = "error"
                row["elapsed_sec"] = f"{time.perf_counter() - t0:.2f}"
                row["error"] = str(e).splitlines()[0][:500]
                print(f"[error] {row['error']}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            append_row(csv_path, row)
            new_rows.append(row)
            done.add(key)

        detach_quantizer(model)
        gc.collect()
        torch.cuda.empty_cache()

    all_rows = existing + new_rows
    if all_rows:
        deduped = {combo_key(r): r for r in all_rows}
        final = sorted(deduped.values(), key=lambda r: (r["task"], r["method"]))
        write_rows(csv_path, final)
        write_summary(results_dir, final, methods, tasks)

    print(f"\n{'='*60}")
    print(f"LONGBENCH RESULTS ({csv_path})")
    print(f"{'='*60}")
    print(f"{'task':<24} {'method':<18} {'score':<8} {'status':<8}")
    print("-" * 60)
    for r in new_rows:
        print(f"{r['task']:<24} {r['method']:<18} {r['score']:<8} {r['status']:<8}")


if __name__ == "__main__":
    main()
