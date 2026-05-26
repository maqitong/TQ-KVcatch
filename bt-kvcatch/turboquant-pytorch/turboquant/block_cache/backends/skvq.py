"""SKVQ-style page backend for ``BlockKVCache``."""

from __future__ import annotations

from typing import Any

import torch

from ..skvq_quantizer import SKVQPageCompressor
from .base import PageQuantBackend


class SKVQPageBackend(PageQuantBackend):
    """Page compressor using SKVQ-style group min/max quantization."""

    name = "skvq"

    def __init__(
        self,
        *,
        head_dim: int,
        n_kv_heads: int,
        group_size: int,
        key_group_size: int | None,
        value_group_size: int | None,
        clipping: float,
        reorder_idx: dict[str, torch.Tensor] | None = None,
        group_st_idx: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.compressor = SKVQPageCompressor(
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            group_size=group_size,
            key_group_size=key_group_size,
            value_group_size=value_group_size,
            clipping=clipping,
            reorder_idx=reorder_idx,
            group_st_idx=group_st_idx,
        )

    @classmethod
    def from_runtime(cls, **kwargs: Any) -> "SKVQPageBackend":
        config = kwargs["config"]
        reorder_idx, group_st_idx = kwargs["reorder"]
        return cls(
            head_dim=int(kwargs["head_dim"]),
            n_kv_heads=int(kwargs["n_kv_heads"]),
            group_size=int(config.group_size),
            key_group_size=config.key_group_size,
            value_group_size=config.value_group_size,
            clipping=float(config.clipping),
            reorder_idx=reorder_idx,
            group_st_idx=group_st_idx,
        )

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        key_bits: float,
        value_bits: float,
        layer_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ck = self.compressor.compress(
            key_states,
            bits=key_bits,
            ttype="k",
            layer_idx=layer_idx,
        )
        cv = self.compressor.compress(
            value_states,
            bits=value_bits,
            ttype="v",
            layer_idx=layer_idx,
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
        k = self.compressor.decompress(compressed_k)
        v = self.compressor.decompress(compressed_v)
        return k.to(dtype), v.to(dtype)
