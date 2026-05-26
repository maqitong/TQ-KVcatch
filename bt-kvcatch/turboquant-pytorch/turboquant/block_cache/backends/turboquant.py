"""TurboQuant page backend for ``BlockKVCache``."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..quantizer import BlockMSECompressor
from .base import PageQuantBackend


class TurboQuantPageBackend(PageQuantBackend):
    """Page compressor backed by TurboQuant's random-rotation MSE quantizer."""

    name = "turboquant"

    def __init__(
        self,
        *,
        head_dim: int,
        layer_idx: int,
        granularity: str,
        seed_base: int,
        device: torch.device,
        reorder_idx: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.granularity = granularity
        self.seed_base = seed_base
        self.device = device
        self.reorder_idx = reorder_idx
        self._k_compressors: dict[float, BlockMSECompressor] = {}
        self._v_compressors: dict[float, BlockMSECompressor] = {}

    @classmethod
    def from_runtime(cls, **kwargs: Any) -> "TurboQuantPageBackend":
        config = kwargs["config"]
        layer_idx = int(kwargs["layer_idx"])
        reorder_idx, _group_st_idx = kwargs["reorder"]
        return cls(
            head_dim=int(kwargs["head_dim"]),
            layer_idx=layer_idx,
            granularity=config.granularity,
            seed_base=int(config.seed) + layer_idx * 1000,
            device=kwargs["device"],
            reorder_idx=reorder_idx,
        )

    def get_compressor(self, ttype: str, bits: float) -> BlockMSECompressor:
        bits_f = float(bits)
        if bits_f != round(bits_f):
            raise ValueError("TurboQuant backend only supports integer bit-widths")
        bits_i = int(bits_f)
        cache = self._k_compressors if ttype == "k" else self._v_compressors
        if bits_f in cache:
            return cache[bits_f]

        seed = self.seed_base if ttype == "k" else self.seed_base + 500
        cache[bits_f] = BlockMSECompressor(
            head_dim=self.head_dim,
            bits=bits_i,
            seed=seed,
            granularity=self.granularity,
            device=str(self.device),
        )
        return cache[bits_f]

    def _reorder_states(
        self, states: torch.Tensor, ttype: str
    ) -> tuple[torch.Tensor, bool]:
        if self.reorder_idx is None or ttype not in self.reorder_idx:
            return states, False

        B, H, S, D = states.shape
        hidden = H * D
        idx = self.reorder_idx[ttype].long().to(states.device)
        if idx.numel() != hidden:
            raise ValueError(
                f"TurboQuant reorder index length {idx.numel()} != hidden {hidden}"
            )

        flat = states.transpose(1, 2).reshape(B, S, hidden)
        reordered = flat.index_select(-1, idx)
        return reordered.reshape(B, S, H, D).transpose(1, 2).contiguous(), True

    def _inverse_reorder_states(self, states: torch.Tensor, ttype: str) -> torch.Tensor:
        if self.reorder_idx is None or ttype not in self.reorder_idx:
            raise ValueError(
                "TurboQuant compressed page was reordered, but metadata is missing"
            )

        B, H, S, D = states.shape
        hidden = H * D
        idx = self.reorder_idx[ttype].long().to(states.device)
        if idx.numel() != hidden:
            raise ValueError(
                f"TurboQuant reorder index length {idx.numel()} != hidden {hidden}"
            )

        inv_idx = idx.argsort()
        flat = states.transpose(1, 2).reshape(B, S, hidden)
        restored = flat.index_select(-1, inv_idx)
        return restored.reshape(B, S, H, D).transpose(1, 2).contiguous()

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        key_bits: float,
        value_bits: float,
        layer_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        work_k, reordered_k = self._reorder_states(key_states, "k")
        work_v, reordered_v = self._reorder_states(value_states, "v")

        ck = self.get_compressor("k", key_bits).compress(work_k)
        cv = self.get_compressor("v", value_bits).compress(work_v)
        ck.update(
            {
                "backend": self.name,
                "tq_reordered": reordered_k,
                "ttype": "k",
                "layer_idx": layer_idx,
            }
        )
        cv.update(
            {
                "backend": self.name,
                "tq_reordered": reordered_v,
                "ttype": "v",
                "layer_idx": layer_idx,
            }
        )
        return ck, cv

    def decompress(
        self,
        compressed_k: dict[str, Any],
        compressed_v: dict[str, Any],
        *,
        key_bits: float,
        value_bits: float,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.get_compressor("k", key_bits).decompress(compressed_k)
        v = self.get_compressor("v", value_bits).decompress(compressed_v)
        if compressed_k.get("tq_reordered", False):
            k = self._inverse_reorder_states(k, "k")
        if compressed_v.get("tq_reordered", False):
            v = self._inverse_reorder_states(v, "v")
        return k.to(dtype), v.to(dtype)
