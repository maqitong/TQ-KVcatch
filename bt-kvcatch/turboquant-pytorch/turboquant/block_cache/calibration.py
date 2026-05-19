"""Calibration utility for KVcatch reorder metadata.

The script collects K/V projection activation statistics from a real model and
builds a SKVQ-compatible reorder file:

    {
        "reorder_indices": [(k_perm, v_perm), ...],
        "cluster_st_inds": [(k_group_starts, v_group_starts), ...],
        "meta": {...}
    }

The default `minmax` metric uses KMeans over per-channel min/max features when
scikit-learn is available, and falls back to absmax sorting otherwise.

Examples:
    python -m turboquant.block_cache.calibration \
        --model D:\model\Llama3.2_3B --local-files-only \
        --inline-text "Calibration text ..." --n-samples 1 --seq-len 64 \
        --group-size 128 --output runs/llama32_reorder.pt

    python -m turboquant.block_cache.calibration \
        --model Qwen/Qwen2.5-3B-Instruct \
        --dataset wikitext --dataset-config wikitext-103-raw-v1 \
        --split train --n-samples 128 --seq-len 4096 \
        --key-group-size 128 --value-group-size 64 --output runs/qwen25_3b/reorder_meta.pt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class LayerStats:
    k_min: torch.Tensor | None = None
    k_max: torch.Tensor | None = None
    k_absmax: torch.Tensor | None = None
    v_min: torch.Tensor | None = None
    v_max: torch.Tensor | None = None
    v_absmax: torch.Tensor | None = None


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


def _load_texts(args) -> list[str]:
    if args.inline_text:
        return [args.inline_text]
    if args.text_file:
        return [Path(args.text_file).read_text(encoding=args.text_encoding)]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required unless --inline-text or --text-file is provided"
        ) from exc

    kwargs = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    ds = load_dataset(args.dataset, split=args.split, **kwargs)
    if args.max_dataset_rows is not None:
        ds = ds.select(range(min(args.max_dataset_rows, len(ds))))
    column = args.text_column or ("text" if "text" in ds.column_names else ds.column_names[0])
    texts = [str(row[column]) for row in ds if str(row[column]).strip()]
    if not texts:
        raise ValueError("no non-empty calibration texts found")
    return texts


def _sample_chunks(tokenizer, texts: Iterable[str], args) -> list[torch.Tensor]:
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=False).input_ids
    if encoded.shape[1] < args.seq_len:
        pad = args.seq_len - encoded.shape[1]
        if tokenizer.eos_token_id is None:
            raise ValueError("corpus shorter than seq_len and tokenizer has no eos token")
        encoded = torch.cat(
            [encoded, torch.full((1, pad), tokenizer.eos_token_id, dtype=encoded.dtype)],
            dim=1,
        )

    rng = random.Random(args.seed)
    max_start = max(0, encoded.shape[1] - args.seq_len)
    chunks = []
    for sample_idx in range(args.n_samples):
        if args.sequential:
            start = min(sample_idx * args.seq_len, max_start)
        else:
            start = rng.randint(0, max_start) if max_start > 0 else 0
        chunks.append(encoded[:, start : start + args.seq_len])
    return chunks


def _layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("could not locate decoder layers on model")


def _get_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module | None:
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _update_stats(stats: LayerStats, ttype: str, output: torch.Tensor) -> None:
    if isinstance(output, (tuple, list)):
        output = output[0]
    act = output.detach().float().reshape(-1, output.shape[-1]).cpu()
    amin = act.amin(dim=0)
    amax = act.amax(dim=0)
    absmax = act.abs().amax(dim=0)

    for suffix, value, reducer in (
        ("min", amin, torch.minimum),
        ("max", amax, torch.maximum),
        ("absmax", absmax, torch.maximum),
    ):
        attr = f"{ttype}_{suffix}"
        old = getattr(stats, attr)
        setattr(stats, attr, value if old is None else reducer(old, value))


def collect_stats(model, tokenizer, args) -> list[LayerStats]:
    layers = list(_layers(model))
    stats = [LayerStats() for _ in layers]
    handles = []

    for layer_idx, layer in enumerate(layers):
        for module_name, ttype in (
            ("self_attn.k_proj", "k"),
            ("self_attn.v_proj", "v"),
            ("attention.k_proj", "k"),
            ("attention.v_proj", "v"),
        ):
            module = _get_module(layer, module_name)
            if module is None:
                continue

            def hook(_module, _inputs, output, i=layer_idx, kind=ttype):
                _update_stats(stats[i], kind, output)

            handles.append(module.register_forward_hook(hook))

    if not handles:
        raise ValueError("no k_proj/v_proj modules found for calibration")

    texts = _load_texts(args)
    chunks = _sample_chunks(tokenizer, texts, args)
    device = _model_input_device(model)

    model.eval()
    with torch.no_grad():
        for idx, chunk in enumerate(chunks, start=1):
            print(f"[calib] sample {idx}/{len(chunks)} seq_len={chunk.shape[1]}")
            model(input_ids=chunk.to(device), use_cache=False)

    for handle in handles:
        handle.remove()

    missing = [
        idx
        for idx, s in enumerate(stats)
        if s.k_min is None or s.v_min is None
    ]
    if missing:
        raise RuntimeError(f"missing K/V stats for layers: {missing[:8]}")
    return stats


def _labels_to_reorder(labels: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.long()
    indices = torch.argsort(labels, stable=True).to(torch.long)
    counts = labels.bincount(minlength=n_clusters).to(torch.long)
    starts = torch.zeros(n_clusters + 1, dtype=torch.long)
    starts[1:] = counts.cumsum(0)
    return indices, starts


def _sort_reorder(score: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.argsort(score, descending=True).to(torch.long)
    hidden = int(score.numel())
    starts = torch.linspace(0, hidden, n_clusters + 1).round().long()
    starts[0] = 0
    starts[-1] = hidden
    return indices, starts


def _kmeans_reorder(features: torch.Tensor, n_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        score = features.abs().amax(dim=1)
        return _sort_reorder(score, n_clusters)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = torch.from_numpy(km.fit(features.numpy()).labels_)
    return _labels_to_reorder(labels, n_clusters)


def build_reorder_meta(stats: list[LayerStats], args) -> dict:
    first = stats[0]
    hidden = int(first.k_absmax.numel())
    key_group_size = args.key_group_size or args.group_size
    value_group_size = args.value_group_size or args.group_size
    key_clusters = args.n_clusters or max(1, hidden // key_group_size)
    value_clusters = args.n_clusters or max(1, hidden // value_group_size)
    key_clusters = max(1, min(key_clusters, hidden))
    value_clusters = max(1, min(value_clusters, hidden))

    reorder_indices = []
    cluster_st_inds = []
    for layer_idx, s in enumerate(stats):
        if args.metric == "minmax":
            k_features = torch.stack([s.k_min, s.k_max], dim=1)
            v_features = torch.stack([s.v_min, s.v_max], dim=1)
            k_idx, k_starts = _kmeans_reorder(k_features, key_clusters)
            v_idx, v_starts = _kmeans_reorder(v_features, value_clusters)
        elif args.metric == "absmax":
            k_idx, k_starts = _kmeans_reorder(s.k_absmax.unsqueeze(1), key_clusters)
            v_idx, v_starts = _kmeans_reorder(s.v_absmax.unsqueeze(1), value_clusters)
        elif args.metric == "absmax_sort":
            k_idx, k_starts = _sort_reorder(s.k_absmax, key_clusters)
            v_idx, v_starts = _sort_reorder(s.v_absmax, value_clusters)
        else:
            raise ValueError(f"unknown metric: {args.metric}")

        print(
            f"[calib] layer={layer_idx} hidden={hidden} "
            f"k_clusters={key_clusters} v_clusters={value_clusters} metric={args.metric}"
        )
        reorder_indices.append((k_idx, v_idx))
        cluster_st_inds.append((k_starts, v_starts))

    return {
        "reorder_indices": reorder_indices,
        "cluster_st_inds": cluster_st_inds,
        "meta": {
            "metric": args.metric,
            "group_size": args.group_size,
            "key_group_size": key_group_size,
            "value_group_size": value_group_size,
            "n_clusters": args.n_clusters,
            "key_clusters": key_clusters,
            "value_clusters": value_clusters,
            "n_layers": len(stats),
            "hidden": hidden,
            "n_samples": args.n_samples,
            "seq_len": args.seq_len,
            "model": args.model,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--text-encoding", default="utf-8")
    parser.add_argument("--inline-text", default=None)
    parser.add_argument("--max-dataset-rows", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--metric", choices=["minmax", "absmax", "absmax_sort"], default="minmax")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--key-group-size", type=int, default=None)
    parser.add_argument("--value-group-size", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[calib] loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[calib] loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        dtype=_dtype_from_name(args.dtype),
    )

    stats = collect_stats(model, tokenizer, args)
    meta = build_reorder_meta(stats, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta, out_path)
    print(f"[calib] saved reorder_meta: {out_path}")
    print("[calib] meta:")
    print(json.dumps(meta["meta"], indent=2))


if __name__ == "__main__":
    main()
