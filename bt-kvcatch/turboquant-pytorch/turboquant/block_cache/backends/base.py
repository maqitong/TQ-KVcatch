"""Backend interface for page-level KV-cache compression.

The cache layer owns page/block scheduling. Backends own only the
algorithm-specific compression payload and reconstruction logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class PageQuantBackend(ABC):
    """Compress and decompress one sealed KV page.

    A page is represented by two tensors shaped ``(B, H, S, D)``. The cache
    framework decides when a page should leave fp16 storage and which K/V bit
    widths it should receive; the backend decides how to encode those tensors.
    """

    name = "base"

    @classmethod
    def from_runtime(cls, **kwargs: Any) -> "PageQuantBackend":
        """Construct a backend from ``BlockCacheLayer`` runtime context."""
        return cls(**kwargs)

    @abstractmethod
    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        key_bits: float,
        value_bits: float,
        layer_idx: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compress one page of K/V states."""
        ...

    @abstractmethod
    def decompress(
        self,
        compressed_k: dict[str, Any],
        compressed_v: dict[str, Any],
        *,
        key_bits: float,
        value_bits: float,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct one compressed page to dense K/V tensors."""
        ...
