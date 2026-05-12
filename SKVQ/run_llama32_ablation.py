from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from transformers import AutoConfig, AutoTokenizer

from KVcache_manager import ModelKVCacheManager
from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model


DEFAULT_CALIB_TEXTS = [
    "The quick brown fox jumps over the lazy dog. Large language models store keys and values in attention caches.",
    "KV cache quantization reduces memory pressure during long-context generation while preserving model quality.",
    "在长上下文推理中，键值缓存会占用大量显存，因此需要比较不同量化策略的困惑度变化。",
    "A fair ablation keeps the model, prompt set, window size, attention sink, clipping, and calibration data fixed.",
    "def quantize(x):\n    scale = x.abs().max(dim=-1, keepdim=True).values\n    return torch.round(x / scale)\n",
    "TurboQuant rotates normalized vectors before Lloyd-Max scalar quantization; SKVQ reorders channels by calibration.",
]

DEFAULT_EVAL_TEXTS = [
    "The assistant should answer clearly, reason carefully, and avoid adding unsupported claims.",
    "When comparing compression methods, perplexity is a useful first signal but downstream tasks are still necessary.",
    "这个实验用于比较原始 SKVQ、纯 TurboQuant 替换、reorder 加 TurboQuant，以及 K/V 不对称加层保护。",
    "Memory-efficient inference often trades reconstruction error against latency and implementation complexity.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Llama3.2-3B SKVQ + TurboQuant ablation")
    parser.add_argument("--model-path", default=r"D:\model\Llama3.2_3B")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--calib-samples", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink", type=int, default=5)
    parser.add_argument("--clip", type=float, default=0.96)
    parser.add_argument("--protect-layers", type=int, default=4)
    parser.add_argument("--results-dir", default="experiments/results/llama32_3b")
    parser.add_argument("--force-recalib", action="store_true")
    return parser.parse_args()


def make_chunks(tokenizer, texts: list[str], seq_len: int, nsamples: int, offset: int = 0) -> list[torch.Tensor]:
    joined = "\n\n".join(texts)
    while True:
        ids = tokenizer(joined, return_tensors="pt", add_special_tokens=True).input_ids
        needed = offset + nsamples * seq_len + 1
        if ids.shape[1] >= needed:
            break
        joined = joined + "\n\n" + joined

    chunks = []
    for i in range(nsamples):
        start = offset + i * seq_len
        chunks.append(ids[:, start : start + seq_len].contiguous())
    return chunks


def detach_quantizer(model: LlamaForCausalLM) -> None:
    for layer in model.model.layers:
        layer.self_attn.KV_cache_manager = None
    model.model.model_kv_manager = None
    model.model_kv_manager = None


def clear_quantizer(model: LlamaForCausalLM) -> None:
    manager = getattr(model.model, "model_kv_manager", None)
    if manager is not None:
        manager.clear()


@torch.no_grad()
def collect_minmax_stats(model: LlamaForCausalLM, chunks: list[torch.Tensor]) -> dict:
    num_layers = len(model.model.layers)
    stats = {
        "min": {"k": [None] * num_layers, "v": [None] * num_layers},
        "max": {"k": [None] * num_layers, "v": [None] * num_layers},
        "absmax": {"k": [None] * num_layers, "v": [None] * num_layers},
    }

    def hook(_module, _inputs, output, ttype: str, layer_idx: int):
        flat = output.detach().reshape(-1, output.shape[-1]).float()
        cur_min = flat.amin(dim=0).cpu()
        cur_max = flat.amax(dim=0).cpu()
        cur_absmax = flat.abs().amax(dim=0).cpu()
        if stats["min"][ttype][layer_idx] is None:
            stats["min"][ttype][layer_idx] = cur_min
            stats["max"][ttype][layer_idx] = cur_max
            stats["absmax"][ttype][layer_idx] = cur_absmax
        else:
            stats["min"][ttype][layer_idx] = torch.minimum(stats["min"][ttype][layer_idx], cur_min)
            stats["max"][ttype][layer_idx] = torch.maximum(stats["max"][ttype][layer_idx], cur_max)
            stats["absmax"][ttype][layer_idx] = torch.maximum(stats["absmax"][ttype][layer_idx], cur_absmax)

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(lambda m, x, y, i=layer_idx: hook(m, x, y, "k", i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(lambda m, x, y, i=layer_idx: hook(m, x, y, "v", i)))

    device = next(model.parameters()).device
    for chunk in chunks:
        model.model(chunk.to(device), use_cache=False)

    for handle in hooks:
        handle.remove()
    return stats


def build_reorder_cache(stats: dict, n_clusters: int, save_path: Path) -> None:
    reorder_indices = []
    cluster_st_inds = []
    num_layers = len(stats["min"]["k"])
    for layer_idx in range(num_layers):
        layer_reorders = []
        layer_starts = []
        for ttype in ("k", "v"):
            features = torch.stack((stats["min"][ttype][layer_idx], stats["max"][ttype][layer_idx]), dim=1)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(features.numpy())
            labels = torch.from_numpy(kmeans.labels_)
            indices = labels.argsort()
            starts = torch.zeros(n_clusters + 1, dtype=torch.int64)
            starts[1:] = labels.bincount(minlength=n_clusters).cumsum(0).to(torch.int64)
            layer_reorders.append(indices)
            layer_starts.append(starts)
        reorder_indices.append(tuple(layer_reorders))
        cluster_st_inds.append(tuple(layer_starts))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reorder_indices": reorder_indices, "cluster_st_inds": cluster_st_inds}, save_path)


def ensure_reorder(model, tokenizer, args: argparse.Namespace, reorder_path: Path) -> None:
    if reorder_path.exists() and not args.force_recalib:
        print(f"[calib] reuse {reorder_path}")
        return

    print("[calib] collecting min/max stats...")
    chunks = make_chunks(tokenizer, DEFAULT_CALIB_TEXTS, args.seq_len, args.calib_samples)
    stats = collect_minmax_stats(model, chunks)
    cfg = model.config
    kv_hidden = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    n_clusters = kv_hidden // args.group_size
    print(f"[calib] building reorder cache with {n_clusters} clusters...")
    build_reorder_cache(stats, n_clusters, reorder_path)
    print(f"[calib] saved {reorder_path}")


def build_manager(model, args: argparse.Namespace, method: str, reorder_path: Path | None):
    num_layers = len(model.model.layers)
    common = {
        "model": model,
        "gsize": args.group_size,
        "window_size": args.window_size,
        "pre_rope": True,
        "clipping": [args.clip for _ in range(num_layers)],
        "attn_sink": args.sink,
        "full_prefill": False,
        "fp8": True,
        "fake_quant": True,
    }
    if method == "skvq_baseline":
        return ModelKVCacheManager.create(kbits=4, vbits=4, reorder_file=str(reorder_path), **common)
    if method == "tq_replace":
        return ModelKVCacheManager.create(
            kbits=4,
            vbits=4,
            reorder_file=None,
            turboquant_config={"use_reorder": False, "protected_layers": 0, "seed_base": 42},
            **common,
        )
    if method == "tq_hybrid":
        return ModelKVCacheManager.create(
            kbits=4,
            vbits=4,
            reorder_file=str(reorder_path),
            turboquant_config={"use_reorder": True, "protected_layers": 0, "seed_base": 42},
            **common,
        )
    if method == "tq_asym_protect":
        return ModelKVCacheManager.create(
            kbits=4,
            vbits=2,
            reorder_file=str(reorder_path),
            turboquant_config={
                "use_reorder": True,
                "protected_layers": args.protect_layers,
                "protected_bits": 8,
                "seed_base": 42,
            },
            **common,
        )
    raise ValueError(f"unknown method: {method}")


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
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reorder_path = results_dir / f"llama32_3b-local-n{args.calib_samples}-len{args.seq_len}-g{args.group_size}-minmax-rod.pt"
    csv_path = results_dir / f"ablation_len{args.seq_len}_eval{args.eval_samples}.csv"

    print("[load] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True, use_fast=False)
    print("[load] config")
    _ = AutoConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    print("[load] SKVQ Llama model")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    print(f"[load] device_map={getattr(model, 'hf_device_map', None)}")

    ensure_reorder(model, tokenizer, args, reorder_path)
    eval_chunks = make_chunks(tokenizer, DEFAULT_EVAL_TEXTS, args.seq_len, args.eval_samples, offset=args.seq_len // 3)

    methods = [
        ("fp16", None),
        ("skvq_baseline", reorder_path),
        ("tq_replace", None),
        ("tq_hybrid", reorder_path),
        ("tq_asym_protect", reorder_path),
    ]

    rows = []
    for method, method_reorder in methods:
        print(f"[eval] {method}")
        detach_quantizer(model)
        if method != "fp16":
            manager = build_manager(model, args, method, method_reorder)
            plug_quantizer_into_model(model, manager)
            print(manager)
        torch.cuda.empty_cache()
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

    best = min(rows, key=lambda row: float(row["ppl"]))
    print(f"[done] wrote {csv_path}")
    print(f"[done] best={best['method']} ppl={best['ppl']}")


if __name__ == "__main__":
    main()
