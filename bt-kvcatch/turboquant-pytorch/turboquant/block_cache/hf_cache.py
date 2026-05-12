"""HuggingFace `Cache`-compatible wrapper around the block-structured KV store.

Targets the **transformers v5** Cache architecture: a `Cache` is a thin
orchestrator that holds one `CacheLayerMixin` per transformer layer. Per-layer
state (block table + TurboQuant compressors) lives in `BlockCacheLayer`;
`BlockKVCache` is a `Cache` subclass that lazily creates these layers as the
model fires `update()` for new layer indices.

The old-API path (transformers <5) is supported via a thin fallback: if the
`CacheLayerMixin` symbol is missing, both classes degrade to plain `object`
and only the unit tests using direct `update()` calls work — they don't need
HF generate integration.

Lazy initialisation: shapes (head_dim, n_kv_heads, batch_size) are discovered
on the first `update()` call to a layer. There is no need to specify them up
front -- pass the cache straight into `generate()`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from transformers.cache_utils import Cache as _HFCache  # type: ignore
    from transformers.cache_utils import CacheLayerMixin as _HFCacheLayerMixin  # type: ignore

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HFCache = object
    _HFCacheLayerMixin = object
    _HF_AVAILABLE = False

from .blocks import BlockState, BlockTable, KVBlock
from .policies import GroupingPolicy, TokenBlockPolicy
from .quantizer import BlockMSECompressor


@dataclass
class BlockCacheConfig:
    """Configuration shared across all layers of a `BlockKVCache`."""

    block_size: int = 16
    key_bits: int = 6
    value_bits: int = 4
    granularity: str = "per-vector"  # 'per-vector' | 'per-block'
    seed: int = 42
    policy: GroupingPolicy = field(default_factory=TokenBlockPolicy)


# ---------------------------------------------------------------------------
# Per-layer cache: holds one BlockTable + compressors
# ---------------------------------------------------------------------------


class BlockCacheLayer(_HFCacheLayerMixin):
    """A single transformer layer's block-structured KV cache.

    Conforms to `transformers.cache_utils.CacheLayerMixin`:
      - `lazy_initialization(k, v)` discovers shapes on first call.
      - `update(k, v)` appends and returns the materialised dense tensors.
      - `get_seq_length()` returns the logical token count.
      - `get_max_cache_shape()` returns -1 (dynamic / unbounded).
      - `get_mask_sizes(query_length)` returns the shape needed for the
        causal-mask path inside attention (matches DynamicLayer behaviour).
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, config: BlockCacheConfig, layer_idx: int = 0):
        if _HF_AVAILABLE:
            super().__init__()
        else:
            self.is_initialized = False
        self.cfg = config
        self.layer_idx = layer_idx
        self.table: Optional[BlockTable] = None
        self.k_compressor: Optional[BlockMSECompressor] = None
        self.v_compressor: Optional[BlockMSECompressor] = None
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

    # ----- abstract method implementations -----

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        B, H, _S, D = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.table = BlockTable(
            block_size=self.cfg.block_size,
            head_dim=D,
            n_kv_heads=H,
            batch_size=B,
        )
        seed_base = self.cfg.seed + self.layer_idx * 1000
        self.k_compressor = BlockMSECompressor(
            head_dim=D,
            bits=self.cfg.key_bits,
            seed=seed_base,
            granularity=self.cfg.granularity,
            device=str(self.device),
        )
        self.v_compressor = BlockMSECompressor(
            head_dim=D,
            bits=self.cfg.value_bits,
            seed=seed_base + 500,
            granularity=self.cfg.granularity,
            device=str(self.device),
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        sealed = self.table.append(key_states, value_states)
        if sealed:
            to_compress = self.cfg.policy.on_seal(sealed, self.table)
            for blk in to_compress:
                ck = self.k_compressor.compress(blk.fp16_k)
                cv = self.v_compressor.compress(blk.fp16_v)
                blk.to_compressed(ck, cv)

        return self._materialize(key_states.dtype)

    def _materialize(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for blk in self.table.blocks:
            if blk.state == BlockState.COMPRESSED:
                k = self.k_compressor.decompress(blk.compressed_k).to(dtype)
                v = self.v_compressor.decompress(blk.compressed_v).to(dtype)
            else:
                k = blk.fp16_k.to(dtype)
                v = blk.fp16_v.to(dtype)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.table.total_len if self.table is not None else 0

    def get_max_cache_shape(self) -> int:
        return -1

    # ----- HF v5 hooks (optional but useful) -----

    def reset(self) -> None:
        if self.table is not None:
            self.table.blocks.clear()
        self.is_initialized = False

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self.table is None:
            return
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k.index_select(0, beam_idx.to(blk.fp16_k.device))
                blk.fp16_v = blk.fp16_v.index_select(0, beam_idx.to(blk.fp16_v.device))
            for d in (blk.compressed_k, blk.compressed_v):
                if d is None:
                    continue
                for k_, val in list(d.items()):
                    if torch.is_tensor(val) and val.shape[:1] == (
                        self.table.batch_size,
                    ):
                        d[k_] = val.index_select(0, beam_idx.to(val.device))

    def crop(self, max_length: int) -> None:
        if self.table is None:
            return
        if max_length < 0:
            max_length = self.get_seq_length() + max_length
        if self.get_seq_length() <= max_length:
            return

        kept: list[KVBlock] = []
        cursor = 0
        for blk in self.table.blocks:
            if cursor >= max_length:
                break
            blk_end = cursor + blk.current_len
            if blk_end <= max_length:
                kept.append(blk)
                cursor = blk_end
                continue
            keep_n = max_length - cursor
            if blk.state == BlockState.COMPRESSED:
                warnings.warn(
                    "crop() truncating into a COMPRESSED block; "
                    "decompressing on the fly."
                )
                blk.fp16_k = self.k_compressor.decompress(blk.compressed_k)[
                    :, :, :keep_n, :
                ]
                blk.fp16_v = self.v_compressor.decompress(blk.compressed_v)[
                    :, :, :keep_n, :
                ]
                blk.compressed_k = None
                blk.compressed_v = None
            else:
                blk.fp16_k = blk.fp16_k[:, :, :keep_n, :]
                blk.fp16_v = blk.fp16_v[:, :, :keep_n, :]
            blk.current_len = keep_n
            blk.state = (
                BlockState.SEALED if keep_n >= blk.block_size else BlockState.FILLING
            )
            kept.append(blk)
            break
        self.table.blocks = kept

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.table is None:
            return
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k.repeat_interleave(repeats, dim=0)
                blk.fp16_v = blk.fp16_v.repeat_interleave(repeats, dim=0)
            for d in (blk.compressed_k, blk.compressed_v):
                if d is None:
                    continue
                for k_, val in list(d.items()):
                    if torch.is_tensor(val) and val.shape[:1] == (
                        self.table.batch_size,
                    ):
                        d[k_] = val.repeat_interleave(repeats, dim=0)
        self.table.batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.table is None:
            return
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k[indices]
                blk.fp16_v = blk.fp16_v[indices]
            for d in (blk.compressed_k, blk.compressed_v):
                if d is None:
                    continue
                for k_, val in list(d.items()):
                    if torch.is_tensor(val) and val.shape[:1] == (
                        self.table.batch_size,
                    ):
                        d[k_] = val[indices]
        self.table.batch_size = int(indices.shape[0])

    def memory_bytes(self) -> int:
        return self.table.memory_bytes() if self.table is not None else 0


# ---------------------------------------------------------------------------
# Top-level Cache: thin orchestrator
# ---------------------------------------------------------------------------


class BlockKVCache(_HFCache):
    """Block-structured KV cache compatible with HuggingFace `generate()`.

    Usage:
        from turboquant.block_cache import (
            BlockKVCache, BlockCacheConfig, WindowBlockPolicy,
        )
        cache = BlockKVCache(BlockCacheConfig(
            block_size=16, key_bits=6, value_bits=4,
            policy=WindowBlockPolicy(window_size=128),
        ))
        out = model.generate(input_ids, past_key_values=cache, max_new_tokens=64)
    """

    def __init__(self, config: Optional[BlockCacheConfig] = None):
        if _HF_AVAILABLE:
            super().__init__(layers=[])
        else:
            self.layers = []
        self.config = config or BlockCacheConfig()
        self._seen_tokens: int = 0

    # ----- core update path -----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Auto-extend layer list as new layer indices appear.
        while len(self.layers) <= layer_idx:
            self.layers.append(
                BlockCacheLayer(self.config, layer_idx=len(self.layers))
            )
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(
            key_states, value_states, *args, **kwargs
        )

    # ----- HF Cache surface -----

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self) -> int:
        return -1

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.layers = []
        self._seen_tokens = 0

    # ----- diagnostics -----

    def memory_report(self) -> dict[str, Any]:
        compressed_bytes = 0
        fp16_baseline = 0
        n_compressed_blocks = 0
        n_fp16_blocks = 0

        for layer in self.layers:
            if layer.table is None:
                continue
            for blk in layer.table.blocks:
                if blk.state == BlockState.COMPRESSED:
                    n_compressed_blocks += 1
                else:
                    n_fp16_blocks += 1
                compressed_bytes += blk.memory_bytes()
                fp16_baseline += (
                    2  # bytes per fp16
                    * 2  # K + V
                    * layer.table.batch_size
                    * layer.table.n_kv_heads
                    * blk.current_len
                    * layer.table.head_dim
                )

        return {
            "compressed_bytes": compressed_bytes,
            "fp16_baseline_bytes": fp16_baseline,
            "compression_ratio": (
                fp16_baseline / compressed_bytes if compressed_bytes > 0 else 0.0
            ),
            "n_compressed_blocks": n_compressed_blocks,
            "n_fp16_blocks": n_fp16_blocks,
            "n_layers": len(self.layers),
        }
