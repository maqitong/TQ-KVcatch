"""KVBlock and BlockTable: per-layer block-structured KV storage.

A `KVBlock` is a fixed-capacity slot holding FP16 K/V tensors while filling,
then optionally swapped to a compressed (TurboQuant-quantized) representation
once a `GroupingPolicy` decides it should leave the FP16 working set.

A `BlockTable` is the ordered list of blocks for a single transformer layer.
Cross-layer organisation lives in `BlockKVCache` (see `hf_cache.py`).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BlockState(Enum):
    FILLING = "filling"        # accepting new tokens, FP16
    SEALED = "sealed"          # full, FP16, awaiting policy decision
    COMPRESSED = "compressed"  # quantized via TurboQuant


@dataclass
class KVBlock:
    """A fixed-capacity slot of KV pairs for one layer.

    Tensor shape conventions (B = batch, H = n_kv_heads, D = head_dim):
        fp16_k / fp16_v: (B, H, current_len, D)
        compressed_k / compressed_v: dict produced by BlockMSECompressor
    """

    block_idx: int
    block_size: int
    head_dim: int
    n_kv_heads: int
    batch_size: int

    state: BlockState = BlockState.FILLING
    current_len: int = 0

    fp16_k: Optional[torch.Tensor] = None
    fp16_v: Optional[torch.Tensor] = None
    compressed_k: Optional[dict] = None
    compressed_v: Optional[dict] = None

    importance: float = 0.0  # page-level score used by mixed precision
    key_bits: Optional[float] = None
    value_bits: Optional[float] = None
    page_meta: Optional[dict] = None

    @property
    def is_full(self) -> bool:
        return self.current_len >= self.block_size

    def append_fp16(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Append k, v of shape (B, H, n_new, D).

        Returns any overflow that did not fit (caller routes it to the next
        block). Returns (None, None) if everything fit.
        """
        if self.state != BlockState.FILLING:
            raise RuntimeError(f"cannot append to block in state {self.state}")
        n_new = k.shape[2]
        capacity = self.block_size - self.current_len
        if n_new <= capacity:
            take_k, take_v = k, v
            overflow_k = overflow_v = None
        else:
            take_k = k[:, :, :capacity, :]
            take_v = v[:, :, :capacity, :]
            overflow_k = k[:, :, capacity:, :]
            overflow_v = v[:, :, capacity:, :]

        if self.fp16_k is None:
            self.fp16_k = take_k.contiguous()
            self.fp16_v = take_v.contiguous()
        else:
            self.fp16_k = torch.cat([self.fp16_k, take_k], dim=2)
            self.fp16_v = torch.cat([self.fp16_v, take_v], dim=2)
        self.current_len = self.fp16_k.shape[2]
        if self.is_full:
            self.state = BlockState.SEALED
        return overflow_k, overflow_v

    def to_compressed(self, compressed_k: dict, compressed_v: dict) -> None:
        """Move from SEALED -> COMPRESSED, drop FP16 storage."""
        if self.state != BlockState.SEALED:
            raise RuntimeError(
                f"can only compress SEALED blocks, got {self.state}"
            )
        self.compressed_k = compressed_k
        self.compressed_v = compressed_v
        self.fp16_k = None
        self.fp16_v = None
        self.state = BlockState.COMPRESSED

    def memory_bytes(self) -> int:
        """Actual memory footprint of this block (storage only)."""
        if self.state in (BlockState.FILLING, BlockState.SEALED):
            n = 0
            if self.fp16_k is not None:
                n += self.fp16_k.numel() * self.fp16_k.element_size()
            if self.fp16_v is not None:
                n += self.fp16_v.numel() * self.fp16_v.element_size()
            return n
        n = 0
        for d in (self.compressed_k, self.compressed_v):
            if d is None:
                continue
            for val in d.values():
                if torch.is_tensor(val):
                    n += val.numel() * val.element_size()
        return n


class BlockTable:
    """Ordered list of KVBlocks for a single transformer layer."""

    def __init__(
        self,
        block_size: int,
        head_dim: int,
        n_kv_heads: int,
        batch_size: int,
    ):
        self.block_size = block_size
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.batch_size = batch_size
        self.blocks: list[KVBlock] = []
        self._total_len = 0

    @property
    def total_len(self) -> int:
        return self._total_len

    def _new_block(self) -> KVBlock:
        blk = KVBlock(
            block_idx=len(self.blocks),
            block_size=self.block_size,
            head_dim=self.head_dim,
            n_kv_heads=self.n_kv_heads,
            batch_size=self.batch_size,
        )
        self.blocks.append(blk)
        return blk

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> list[KVBlock]:
        """Append (B, H, n, D). Returns blocks newly SEALED by this call."""
        sealed: list[KVBlock] = []
        n_appended = int(k.shape[2])
        cur: Optional[KVBlock] = (
            self.blocks[-1]
            if self.blocks and self.blocks[-1].state == BlockState.FILLING
            else None
        )
        rest_k: Optional[torch.Tensor] = k
        rest_v: Optional[torch.Tensor] = v
        while rest_k is not None and rest_k.shape[2] > 0:
            if cur is None:
                cur = self._new_block()
            rest_k, rest_v = cur.append_fp16(rest_k, rest_v)
            if cur.state == BlockState.SEALED:
                sealed.append(cur)
                cur = None
        self._total_len += n_appended
        return sealed

    def memory_bytes(self) -> int:
        return sum(b.memory_bytes() for b in self.blocks)
