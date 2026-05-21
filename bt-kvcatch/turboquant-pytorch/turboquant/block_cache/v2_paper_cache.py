"""TurboQuant paper V2 KV cache (compressors.py): MSE+QJL keys, MSE values.

No residual_window — each update compresses the full KV history seen so far.
Attention uses ``k_mse`` reconstruction for keys (stored QJL is not wired into
HF softmax; same limitation as offline validate.py which compares scores separately).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from turboquant.compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2


class V2PaperCache(DynamicCache):
    """HF DynamicCache backed by TurboQuantCompressorV2 / TurboQuantCompressorMSE."""

    def __init__(
        self,
        key_bits: int = 2,
        value_bits: int = 2,
        n_layers: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.key_bits = int(key_bits)
        self.value_bits = int(value_bits)
        self.n_layers = int(n_layers)
        self.seed = int(seed)
        self._key_comp: dict[int, TurboQuantCompressorV2] = {}
        self._val_comp: dict[int, TurboQuantCompressorMSE] = {}
        self._chunks_k: dict[int, list[dict]] = {}
        self._chunks_v: dict[int, list[dict]] = {}
        self._pending_k: dict[int, list[torch.Tensor]] = {}
        self._pending_v: dict[int, list[torch.Tensor]] = {}
        self._total_seq: dict[int, int] = {}
        self._batch_size = 1
        self._n_kv_heads = 0
        self._head_dim = 0

    def _get_key_comp(self, layer_idx: int, head_dim: int, device: torch.device) -> TurboQuantCompressorV2:
        if layer_idx not in self._key_comp:
            self._key_comp[layer_idx] = TurboQuantCompressorV2(
                head_dim=head_dim,
                bits=self.key_bits,
                seed=self.seed + layer_idx * 1000,
                device=str(device),
            )
        return self._key_comp[layer_idx]

    def _get_val_comp(self, layer_idx: int, head_dim: int, device: torch.device) -> TurboQuantCompressorMSE:
        if layer_idx not in self._val_comp:
            self._val_comp[layer_idx] = TurboQuantCompressorMSE(
                head_dim=head_dim,
                bits=self.value_bits,
                seed=self.seed + layer_idx * 1000 + 500,
                device=str(device),
            )
        return self._val_comp[layer_idx]

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

        if layer_idx not in self._pending_k:
            self._pending_k[layer_idx] = []
            self._pending_v[layer_idx] = []
            self._chunks_k[layer_idx] = []
            self._chunks_v[layer_idx] = []
            self._total_seq[layer_idx] = 0

        self._total_seq[layer_idx] += S_new
        self._pending_k[layer_idx].append(key_states)
        self._pending_v[layer_idx].append(value_states)

        recent_k = torch.cat(self._pending_k[layer_idx], dim=2)
        recent_v = torch.cat(self._pending_v[layer_idx], dim=2)

        key_comp = self._get_key_comp(layer_idx, D, device)
        val_comp = self._get_val_comp(layer_idx, D, device)

        if recent_k.shape[2] > 0:
            self._chunks_k[layer_idx] = [key_comp.compress(recent_k)]
            self._chunks_v[layer_idx] = [val_comp.compress(recent_v)]
            self._pending_k[layer_idx] = []
            self._pending_v[layer_idx] = []

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        for ck in self._chunks_k[layer_idx]:
            parts_k.append(ck["k_mse"].to(key_states.dtype))
        for cv in self._chunks_v[layer_idx]:
            parts_v.append(val_comp.decompress(cv).to(value_states.dtype))

        full_k = torch.cat(parts_k, dim=2) if parts_k else key_states
        full_v = torch.cat(parts_v, dim=2) if parts_v else value_states

        while len(self.layers) <= layer_idx:
            from transformers.cache_utils import DynamicLayer

            self.layers.append(DynamicLayer())

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._total_seq.get(layer_idx, 0)

    def memory_report(self) -> dict[str, Any]:
        """Byte accounting aligned with validate.py V2 estimates."""
        compressed_bytes = 0.0
        fp16_baseline = 0.0
        n_compressed_blocks = 0
        mse_bits = max(self.key_bits - 1, 1)

        for layer_idx, seq_len in self._total_seq.items():
            if seq_len <= 0:
                continue
            n_vecs = self._batch_size * self._n_kv_heads * seq_len
            d = self._head_dim

            k_bits = n_vecs * d * mse_bits
            k_bits += n_vecs * d * 1
            k_bits += n_vecs * 16 * 2

            v_bits = n_vecs * d * self.value_bits
            v_bits += n_vecs * 16

            compressed_bytes += (k_bits + v_bits) / 8.0
            fp16_baseline += n_vecs * d * 2 * 2 * 2

            n_compressed_blocks += max(1, seq_len // 16)

        return {
            "compressed_bytes": compressed_bytes,
            "fp16_baseline_bytes": fp16_baseline,
            "compression_ratio": (
                fp16_baseline / compressed_bytes if compressed_bytes > 0 else 0.0
            ),
            "n_compressed_blocks": n_compressed_blocks,
            "n_fp16_blocks": 0,
            "n_layers": len(self._total_seq),
            "bit_histogram": {f"K{self.key_bits}/V{self.value_bits}": n_compressed_blocks},
            "precision_histogram": {"v2_paper": n_compressed_blocks},
        }
