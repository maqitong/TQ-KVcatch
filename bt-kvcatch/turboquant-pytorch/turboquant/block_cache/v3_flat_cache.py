"""TurboQuant V3 flat KV cache (MSE / Lloyd-Max), optional residual window.

Aligns with author ``generation_test.V3Cache``: no block/sink; ``residual_window``
keeps the most recent tokens in FP16 (default 128). ``residual_window=0`` compresses
the full history (ablation only — poor generation in author README).
"""
from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from turboquant.compressors_v3 import TurboQuantV3


class V3FlatCache(DynamicCache):
    """Flat TurboQuant V3 cache (generation_test_v2.V3Cache), HF-compatible."""

    def __init__(
        self,
        key_bits: int = 2,
        value_bits: int = 2,
        residual_window: int = 128,
        protected_layers: int = 0,
        n_layers: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.key_bits = int(key_bits)
        self.value_bits = int(value_bits)
        self.residual_window = int(residual_window)
        self.protected_layers = int(protected_layers)
        self.n_layers = int(n_layers)
        self.seed = int(seed)
        self._compressors: dict[int, TurboQuantV3] = {}
        self._chunks_k: dict[int, list] = {}
        self._chunks_v: dict[int, list] = {}
        self._fp16_recent_k: dict[int, list] = {}
        self._fp16_recent_v: dict[int, list] = {}
        self._total_seq: dict[int, int] = {}
        self._compressed_tokens: dict[int, int] = {}
        self._batch_size: int = 1
        self._n_kv_heads: int = 0
        self._head_dim: int = 0

    def _get_compressor(self, layer_idx: int, head_dim: int, device: torch.device) -> TurboQuantV3:
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=0,
                layer_idx=layer_idx,
                n_layers=self.n_layers,
                protected_layers=self.protected_layers,
                seed=self.seed + layer_idx * 1000,
                device=str(device),
            )
        return self._compressors[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, S_new, D = key_states.shape
        device = key_states.device
        self._batch_size = B
        self._n_kv_heads = H
        self._head_dim = D
        comp = self._get_compressor(layer_idx, D, device)

        if layer_idx not in self._chunks_k:
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._fp16_recent_k[layer_idx] = []
            self._fp16_recent_v[layer_idx] = []
            self._total_seq[layer_idx] = 0
            self._compressed_tokens[layer_idx] = 0

        self._total_seq[layer_idx] += S_new
        self._fp16_recent_k[layer_idx].append(key_states)
        self._fp16_recent_v[layer_idx].append(value_states)

        recent_k = torch.cat(self._fp16_recent_k[layer_idx], dim=2)
        recent_v = torch.cat(self._fp16_recent_v[layer_idx], dim=2)
        rw = self.residual_window

        if rw == 0:
            if recent_k.shape[2] > 0:
                ck, cv = comp.compress_kv(recent_k, recent_v)
                self._chunks_k[layer_idx].append(ck)
                self._chunks_v[layer_idx].append(cv)
                self._compressed_tokens[layer_idx] += recent_k.shape[2]
                self._fp16_recent_k[layer_idx] = []
                self._fp16_recent_v[layer_idx] = []
        elif recent_k.shape[2] > rw:
            overflow = recent_k.shape[2] - rw
            ck, cv = comp.compress_kv(
                recent_k[:, :, :overflow, :],
                recent_v[:, :, :overflow, :],
            )
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)
            self._compressed_tokens[layer_idx] += overflow
            self._fp16_recent_k[layer_idx] = [recent_k[:, :, overflow:, :]]
            self._fp16_recent_v[layer_idx] = [recent_v[:, :, overflow:, :]]

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        for ck, cv in zip(self._chunks_k[layer_idx], self._chunks_v[layer_idx]):
            dk, dv = comp.decompress_kv(ck, cv)
            parts_k.append(dk.to(key_states.dtype))
            parts_v.append(dv.to(value_states.dtype))

        if self._fp16_recent_k[layer_idx]:
            parts_k.append(torch.cat(self._fp16_recent_k[layer_idx], dim=2))
            parts_v.append(torch.cat(self._fp16_recent_v[layer_idx], dim=2))

        full_k = torch.cat(parts_k, dim=2) if parts_k else key_states
        full_v = torch.cat(parts_v, dim=2) if parts_v else value_states

        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer

            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def memory_report(self) -> dict[str, Any]:
        """Align with BlockKVCache.memory_report() for eval_ppl / bpw."""
        compressed_bytes = 0
        fp16_baseline = 0
        n_compressed_blocks = 0
        n_fp16_blocks = 0
        bit_histogram: dict[str, int] = {}
        bit_key = f"K{self.key_bits}/V{self.value_bits}"

        for layer_idx, seq_len in self._total_seq.items():
            comp = self._compressors.get(layer_idx)
            if comp is None or seq_len <= 0:
                continue
            stats = comp.memory_bytes(self._batch_size, self._n_kv_heads, seq_len)
            compressed_bytes += stats["compressed_bytes"]
            fp16_baseline += stats["fp16_bytes"]
            ct = stats["compressed_tokens"]
            ft = stats["fp16_tokens"]
            n_compressed_blocks += max(1, ct // 16) if ct else 0
            n_fp16_blocks += max(1, ft // 16) if ft else 0
            if ct > 0:
                bit_histogram[bit_key] = bit_histogram.get(bit_key, 0) + max(1, ct // 16)

        return {
            "compressed_bytes": compressed_bytes,
            "fp16_baseline_bytes": fp16_baseline,
            "compression_ratio": (
                fp16_baseline / compressed_bytes if compressed_bytes > 0 else 0.0
            ),
            "n_compressed_blocks": n_compressed_blocks,
            "n_fp16_blocks": n_fp16_blocks,
            "n_layers": len(self._total_seq),
            "bit_histogram": bit_histogram,
            "precision_histogram": {},
            "residual_window": self.residual_window,
        }
