"""Comprehensive comparison: FP16 vs TurboQuant V3 vs BlockCache policies.

Three experiments:
  1. Compression ratio vs retrieval quality across methods and context lengths
  2. Bit-width sweep — find quality cliff for each method
  3. Granularity comparison — per-vector vs per-block quantization

Run:
    conda activate kvcatch
    python -m turboquant.block_cache.experiment
"""

from __future__ import annotations

import gc
import os
import sys
import time
import argparse
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

from turboquant.compressors_v3 import TurboQuantV3
from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    HybridPolicy,
    TokenBlockPolicy,
    WindowBlockPolicy,
)

MODEL_PATH = r"D:\model\Qwen2.5-3B-Instruct"

NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED = "AURORA-7749"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


# ---------------------------------------------------------------------------
# V3Cache — re-implementation matching generation_test.py but importable
# ---------------------------------------------------------------------------

class V3Cache(DynamicCache):
    """TurboQuant V3 flat cache (same as generation_test.V3Cache)."""

    def __init__(self, key_bits=4, value_bits=2, residual_window=128,
                 protected_layers=0, n_layers=36):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.protected_layers = protected_layers
        self.n_layers = n_layers
        self._compressors = {}
        self._chunks_k = {}
        self._chunks_v = {}
        self._fp16_recent_k = {}
        self._fp16_recent_v = {}
        self._total_seq = {}

    def _get_compressor(self, layer_idx, head_dim, device):
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=0,
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.protected_layers,
                seed=42,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, S_new, D = key_states.shape
        device = key_states.device
        comp = self._get_compressor(layer_idx, D, device)

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0

        self._total_seq[layer_idx] += S_new

        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        if rw == 0 or recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw

            to_compress_k = recent_k[:, :, :overflow, :]
            to_compress_v = recent_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

            recent_k = recent_k[:, :, overflow:, :]
            recent_v = recent_v[:, :, overflow:, :]
            self._fp16_recent_k[layer_idx] = [recent_k]
            self._fp16_recent_v[layer_idx] = [recent_v]

        parts_k = []
        parts_v = []
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        full_k = torch.cat(parts_k, dim=2)
        full_v = torch.cat(parts_v, dim=2)

        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer
            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        return self._total_seq.get(layer_idx, 0)

    def get_compression_ratio(self) -> float:
        """Estimate compression ratio based on configured bits."""
        total_compressed = 0
        total_fp16 = 0
        for layer_idx in self._total_seq:
            comp = self._compressors.get(layer_idx)
            if comp is None:
                continue
            S = self._total_seq[layer_idx]
            rw = min(self.residual_window, S)
            compressed_S = S - rw
            B = 1
            H = 2
            D = comp.head_dim

            compressed_k_bytes = compressed_S * D * comp.key_bits / 8
            compressed_v_bytes = compressed_S * D * comp.value_bits / 8
            fp16_window_bytes = rw * D * 2 * 2
            norm_bytes = compressed_S * 2 * 2

            total_compressed += compressed_k_bytes + compressed_v_bytes + fp16_window_bytes + norm_bytes
            total_fp16 += S * D * 2 * 2

        return total_fp16 / total_compressed if total_compressed > 0 else 1.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    found: bool
    response: str
    ratio: float
    seconds: float


def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.5):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return (
        f"<|im_start|>system\nYou are a helpful assistant. Answer concisely.<|im_end|>\n"
        f"<|im_start|>user\nRead this document:\n\n{haystack}\n\n"
        f"What is the secret project code name? Answer with just the code name.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_test(model, tokenizer, target_tokens, cache, label=""):
    prompt = build_prompt(tokenizer, target_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 512)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    n_tokens = input_ids.shape[1]

    print(f"    [{label}] {n_tokens} tokens...", end=" ", flush=True)

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
    dt = time.perf_counter() - t0

    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    found = EXPECTED.lower() in response.lower()

    if isinstance(cache, BlockKVCache):
        report = cache.memory_report()
        ratio = report["compression_ratio"]
    elif isinstance(cache, V3Cache):
        ratio = cache.get_compression_ratio()
    else:
        ratio = 1.0

    safe = response[:60].encode("ascii", errors="replace").decode("ascii")
    print(f"{'FOUND' if found else 'MISS':>5} | ratio={ratio:.2f}x | {dt:.1f}s | \"{safe}\"")

    return RunResult(found=found, response=response, ratio=ratio, seconds=dt)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def experiment1(model, tokenizer, n_layers):
    """Compression ratio vs retrieval quality across methods and context lengths."""
    print("\n" + "=" * 76)
    print("Experiment 1: Compression vs Quality (K6/V4, window=128)")
    print("=" * 76)

    key_bits, value_bits = 6, 4
    window = 128
    block_size = 16
    ctx_lengths = [1024, 2048, 3072, 4096]

    methods = [
        ("FP16 baseline", "fp16"),
        ("TQ V3 flat rw=128", "v3"),
        ("TokenBlock + TQ", "token"),
        ("WindowBlock + TQ", "window"),
        ("Hybrid sink=4 + TQ", "hybrid"),
    ]

    quality = {m[0]: {} for m in methods}
    ratios = {m[0]: {} for m in methods}

    for ctx in ctx_lengths:
        print(f"\n  Context: ~{ctx} tokens")
        print(f"  {'-' * 72}")
        for name, kind in methods:
            if kind == "fp16":
                cache = None
            elif kind == "v3":
                cache = V3Cache(
                    key_bits=key_bits, value_bits=value_bits,
                    residual_window=window, n_layers=n_layers,
                )
            elif kind == "token":
                cache = BlockKVCache(BlockCacheConfig(
                    block_size=block_size, key_bits=key_bits, value_bits=value_bits,
                    policy=TokenBlockPolicy(),
                ))
            elif kind == "window":
                cache = BlockKVCache(BlockCacheConfig(
                    block_size=block_size, key_bits=key_bits, value_bits=value_bits,
                    policy=WindowBlockPolicy(window_size=window),
                ))
            elif kind == "hybrid":
                cache = BlockKVCache(BlockCacheConfig(
                    block_size=block_size, key_bits=key_bits, value_bits=value_bits,
                    policy=HybridPolicy(sink_size=4, window_size=window),
                ))

            r = run_test(model, tokenizer, ctx, cache, label=name)
            quality[name][ctx] = r.found
            ratios[name][ctx] = r.ratio

            gc.collect()
            torch.cuda.empty_cache()

    # Print quality table
    print(f"\n{'=' * 76}")
    print("Results: Retrieval Quality (FOUND/MISS)")
    print(f"{'=' * 76}")
    header = f"  {'Method':<28s}"
    for ctx in ctx_lengths:
        header += f" {ctx:>6d}"
    print(header)
    print(f"  {'-' * 28}" + " ------" * len(ctx_lengths))
    for name, _ in methods:
        row = f"  {name:<28s}"
        for ctx in ctx_lengths:
            row += f" {'FOUND' if quality[name][ctx] else 'MISS':>6}"
        print(row)

    # Print compression table
    print(f"\nResults: Compression Ratio")
    print(f"{'=' * 76}")
    print(header)
    print(f"  {'-' * 28}" + " ------" * len(ctx_lengths))
    for name, _ in methods:
        row = f"  {name:<28s}"
        for ctx in ctx_lengths:
            row += f" {ratios[name][ctx]:>5.2f}x"
        print(row)
    print(f"{'=' * 76}")

    return quality, ratios


def experiment2(model, tokenizer, n_layers):
    """Bit-width sweep — find quality cliff."""
    print("\n" + "=" * 76)
    print("Experiment 2: Bit-Width Sweep (ctx=2048, window=128)")
    print("=" * 76)

    ctx = 2048
    window = 128
    block_size = 16

    bit_configs = [
        ("K8/V4", 8, 4),
        ("K6/V4", 6, 4),
        ("K4/V4", 4, 4),
        ("K4/V3", 4, 3),
        ("K4/V2", 4, 2),
    ]

    methods = [
        ("TQ V3 flat", "v3"),
        ("TokenBlock", "token"),
        ("WindowBlock", "window"),
    ]

    quality = {m[0]: {} for m in methods}
    ratios = {m[0]: {} for m in methods}

    for bit_name, kb, vb in bit_configs:
        print(f"\n  {bit_name} (K={kb}, V={vb})")
        print(f"  {'-' * 72}")
        for name, kind in methods:
            if kind == "v3":
                cache = V3Cache(
                    key_bits=kb, value_bits=vb,
                    residual_window=window, n_layers=n_layers,
                )
            elif kind == "token":
                cache = BlockKVCache(BlockCacheConfig(
                    block_size=block_size, key_bits=kb, value_bits=vb,
                    policy=TokenBlockPolicy(),
                ))
            elif kind == "window":
                cache = BlockKVCache(BlockCacheConfig(
                    block_size=block_size, key_bits=kb, value_bits=vb,
                    policy=WindowBlockPolicy(window_size=window),
                ))

            r = run_test(model, tokenizer, ctx, cache, label=f"{bit_name} {name}")
            quality[name][bit_name] = r.found
            ratios[name][bit_name] = r.ratio

            gc.collect()
            torch.cuda.empty_cache()

    # Print quality table
    print(f"\n{'=' * 76}")
    print("Results: Retrieval Quality by Bit-Width")
    print(f"{'=' * 76}")
    header = f"  {'Method':<20s}"
    for bit_name, _, _ in bit_configs:
        header += f" {bit_name:>7s}"
    print(header)
    print(f"  {'-' * 20}" + " -------" * len(bit_configs))
    for name, _ in methods:
        row = f"  {name:<20s}"
        for bit_name, _, _ in bit_configs:
            row += f" {'FOUND' if quality[name][bit_name] else 'MISS':>7}"
        print(row)

    # Print compression table
    print(f"\nResults: Compression Ratio by Bit-Width")
    print(f"{'=' * 76}")
    print(header)
    print(f"  {'-' * 20}" + " -------" * len(bit_configs))
    for name, _ in methods:
        row = f"  {name:<20s}"
        for bit_name, _, _ in bit_configs:
            row += f" {ratios[name][bit_name]:>6.2f}x"
        print(row)
    print(f"{'=' * 76}")

    return quality, ratios


def experiment3(model, tokenizer, n_layers):
    """Granularity comparison: per-vector vs per-block."""
    print("\n" + "=" * 76)
    print("Experiment 3: Quantization Granularity (WindowBlock, K6/V4, ctx=2048)")
    print("=" * 76)

    ctx = 2048
    window = 128
    block_size = 16
    key_bits, value_bits = 6, 4

    granularities = [
        ("per-vector", "per-vector"),
        ("per-block", "per-block"),
    ]

    results = {}

    for label, gran in granularities:
        print(f"\n  Granularity: {label}")
        print(f"  {'-' * 72}")

        cache = BlockKVCache(BlockCacheConfig(
            block_size=block_size, key_bits=key_bits, value_bits=value_bits,
            granularity=gran,
            policy=WindowBlockPolicy(window_size=window),
        ))

        r = run_test(model, tokenizer, ctx, cache, label=label)
        results[label] = r

        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison
    print(f"\n{'=' * 76}")
    print("Results: Granularity Comparison")
    print(f"{'=' * 76}")
    print(f"  {'Granularity':<20s} {'Quality':>10s} {'Ratio':>10s} {'Time':>10s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
    for label in ["per-vector", "per-block"]:
        r = results[label]
        print(f"  {label:<20s} {'FOUND' if r.found else 'MISS':>10} {r.ratio:>9.2f}x {r.seconds:>9.1f}s")
    print(f"{'=' * 76}")

    # Also test per-block at different bit widths to show trade-off
    print(f"\n  Extended: per-block across bit widths (WindowBlock, ctx=2048)")
    print(f"  {'-' * 72}")

    for kb, vb in [(6, 4), (4, 4), (4, 3)]:
        for gran in ["per-vector", "per-block"]:
            cache = BlockKVCache(BlockCacheConfig(
                block_size=block_size, key_bits=kb, value_bits=vb,
                granularity=gran,
                policy=WindowBlockPolicy(window_size=window),
            ))
            r = run_test(model, tokenizer, ctx, cache, label=f"K{kb}/V{vb} {gran}")
            gc.collect()
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive KV cache compression experiments")
    parser.add_argument("--skip-exp", nargs="+", type=int, default=[], help="Experiment numbers to skip (1, 2, or 3)")
    args = parser.parse_args()

    print()
    print("=" * 76)
    print("TurboQuant Block Cache — Comprehensive Experiment Suite")
    print(f"Model: {MODEL_PATH}")
    print(f"Needle: \"{NEEDLE}\"")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 76)

    print("\nLoading model (4-bit)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        ),
        device_map="auto", dtype=torch.float16,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. {n_layers} layers. GPU: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    if 1 not in args.skip_exp:
        experiment1(model, tokenizer, n_layers)
    if 2 not in args.skip_exp:
        experiment2(model, tokenizer, n_layers)
    if 3 not in args.skip_exp:
        experiment3(model, tokenizer, n_layers)

    print("\n" + "=" * 76)
    print("All experiments complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
