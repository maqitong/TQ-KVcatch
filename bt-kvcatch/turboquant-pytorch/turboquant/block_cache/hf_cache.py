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
from .bit_allocator import (
    FixedPageBitAllocator,
    PageBitAllocator,
    TopRatioPageBitAllocator,
)
from .page_importance import build_page_importance_scorer
from .skvq_quantizer import SKVQPageCompressor


@dataclass
class BlockCacheConfig:
    """Configuration shared across all layers of a `BlockKVCache`."""

    block_size: int = 16
    key_bits: int = 6
    value_bits: int = 4
    granularity: str = "per-vector"  # 'per-vector' | 'per-block'
    seed: int = 42
    policy: GroupingPolicy = field(default_factory=TokenBlockPolicy)
    quant_backend: str = "turboquant"  # 'turboquant' | 'skvq'
    mixed_precision: bool = False
    importance_metric: str = "k_norm"
    important_ratio: float = 0.2
    high_key_bits: float = 4
    high_value_bits: float = 2
    low_key_bits: float = 2
    low_value_bits: float = 2
    group_size: int = 128
    clipping: float = 0.92
    reorder_file: Optional[str] = None
    reorder_meta: Optional[dict[str, Any]] = None


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

    def __init__(
        self,
        config: BlockCacheConfig,
        layer_idx: int = 0,
        reorder_meta: Optional[dict[str, Any]] = None,
    ):
        if _HF_AVAILABLE:
            super().__init__()
        else:
            self.is_initialized = False
        self.cfg = config
        self.layer_idx = layer_idx
        self.reorder_meta = reorder_meta
        self.table: Optional[BlockTable] = None
        self.k_compressor: Optional[BlockMSECompressor] = None
        self.v_compressor: Optional[BlockMSECompressor] = None
        self.skvq_compressor: Optional[SKVQPageCompressor] = None
        self.bit_allocator: Optional[PageBitAllocator] = None
        self._k_compressors: dict[float, BlockMSECompressor] = {}
        self._v_compressors: dict[float, BlockMSECompressor] = {}
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

    # ----- abstract method implementations -----

    def _layer_reorder_meta(
        self,
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]:
        if self.reorder_meta is None:
            return None, None

        try:
            reorder_pair = self.reorder_meta["reorder_indices"][self.layer_idx]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"invalid reorder metadata for layer {self.layer_idx}"
            ) from exc

        group_pair = None
        if "cluster_st_inds" in self.reorder_meta:
            try:
                group_pair = self.reorder_meta["cluster_st_inds"][self.layer_idx]
            except (IndexError, TypeError) as exc:
                raise ValueError(
                    f"invalid group-start metadata for layer {self.layer_idx}"
                ) from exc

        reorder_idx = {"k": reorder_pair[0], "v": reorder_pair[1]}
        group_st_idx = (
            {"k": group_pair[0], "v": group_pair[1]} if group_pair is not None else None
        )
        return reorder_idx, group_st_idx

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
        if self.cfg.quant_backend not in ("turboquant", "skvq"):
            raise ValueError(f"unknown quant_backend: {self.cfg.quant_backend}")

        if self.cfg.quant_backend == "skvq":
            reorder_idx, group_st_idx = self._layer_reorder_meta()
            self.skvq_compressor = SKVQPageCompressor(
                head_dim=D,
                n_kv_heads=H,
                group_size=self.cfg.group_size,
                clipping=self.cfg.clipping,
                reorder_idx=reorder_idx,
                group_st_idx=group_st_idx,
            )
        else:
            self.k_compressor = self._get_turboquant_compressor(
                "k", self.cfg.key_bits, seed_base
            )
            self.v_compressor = self._get_turboquant_compressor(
                "v", self.cfg.value_bits, seed_base
            )

        if self.cfg.mixed_precision:
            scorer = build_page_importance_scorer(self.cfg.importance_metric)
            self.bit_allocator = TopRatioPageBitAllocator(
                scorer=scorer,
                important_ratio=self.cfg.important_ratio,
                high_key_bits=self.cfg.high_key_bits,
                high_value_bits=self.cfg.high_value_bits,
                low_key_bits=self.cfg.low_key_bits,
                low_value_bits=self.cfg.low_value_bits,
            )
        else:
            self.bit_allocator = FixedPageBitAllocator(
                self.cfg.key_bits, self.cfg.value_bits
            )
        self.is_initialized = True

    def _get_turboquant_compressor(
        self, ttype: str, bits: float, seed_base: Optional[int] = None
    ) -> BlockMSECompressor:
        bits_f = float(bits)
        if bits_f != round(bits_f):
            raise ValueError("TurboQuant backend only supports integer bit-widths")
        bits_i = int(bits_f)
        cache = self._k_compressors if ttype == "k" else self._v_compressors
        if bits_f in cache:
            return cache[bits_f]
        if seed_base is None:
            seed_base = self.cfg.seed + self.layer_idx * 1000
        seed = seed_base if ttype == "k" else seed_base + 500
        cache[bits_f] = BlockMSECompressor(
            head_dim=self.table.head_dim,
            bits=bits_i,
            seed=seed,
            granularity=self.cfg.granularity,
            device=str(self.device),
        )
        return cache[bits_f]

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
            bit_assignments = self.bit_allocator.assign_many(
                to_compress, self.table, self.layer_idx
            )
            for blk in to_compress:
                k_bits, v_bits = bit_assignments[blk.block_idx]
                blk.key_bits = k_bits
                blk.value_bits = v_bits
                if self.cfg.quant_backend == "skvq":
                    ck = self.skvq_compressor.compress(
                        blk.fp16_k,
                        bits=k_bits,
                        ttype="k",
                        layer_idx=self.layer_idx,
                    )
                    cv = self.skvq_compressor.compress(
                        blk.fp16_v,
                        bits=v_bits,
                        ttype="v",
                        layer_idx=self.layer_idx,
                    )
                else:
                    ck = self._get_turboquant_compressor("k", k_bits).compress(
                        blk.fp16_k
                    )
                    cv = self._get_turboquant_compressor("v", v_bits).compress(
                        blk.fp16_v
                    )
                blk.to_compressed(ck, cv)

        return self._materialize(key_states.dtype)

    def _decompress_block(
        self, blk: KVBlock, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if blk.compressed_k.get("backend") == "skvq":
            k = self.skvq_compressor.decompress(blk.compressed_k).to(dtype)
            v = self.skvq_compressor.decompress(blk.compressed_v).to(dtype)
            return k, v

        k_bits = blk.key_bits if blk.key_bits is not None else self.cfg.key_bits
        v_bits = blk.value_bits if blk.value_bits is not None else self.cfg.value_bits
        k = self._get_turboquant_compressor("k", k_bits).decompress(blk.compressed_k)
        v = self._get_turboquant_compressor("v", v_bits).decompress(blk.compressed_v)
        return k.to(dtype), v.to(dtype)

    def _materialize(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for blk in self.table.blocks:
            if blk.state == BlockState.COMPRESSED:
                k, v = self._decompress_block(blk, dtype)
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
                dk, dv = self._decompress_block(blk, self.dtype)
                blk.fp16_k = dk[:, :, :keep_n, :]
                blk.fp16_v = dv[:, :, :keep_n, :]
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
        if self.config.reorder_meta is not None and self.config.reorder_file is not None:
            raise ValueError("set only one of reorder_meta or reorder_file")
        self._reorder_meta = self.config.reorder_meta
        if self._reorder_meta is None and self.config.reorder_file is not None:
            self._reorder_meta = torch.load(self.config.reorder_file, map_location="cpu")
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
                BlockCacheLayer(
                    self.config,
                    layer_idx=len(self.layers),
                    reorder_meta=self._reorder_meta,
                )
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
        bit_histogram: dict[str, int] = {}
        precision_histogram: dict[str, int] = {}

        for layer in self.layers:
            if layer.table is None:
                continue
            for blk in layer.table.blocks:
                if blk.state == BlockState.COMPRESSED:
                    n_compressed_blocks += 1
                    k_bits = blk.key_bits if blk.key_bits is not None else "?"
                    v_bits = blk.value_bits if blk.value_bits is not None else "?"
                    bit_key = f"K{k_bits}/V{v_bits}"
                    bit_histogram[bit_key] = bit_histogram.get(bit_key, 0) + 1
                    precision = (
                        blk.page_meta.get("precision")
                        if isinstance(blk.page_meta, dict)
                        else None
                    )
                    if precision is not None:
                        precision_histogram[precision] = (
                            precision_histogram.get(precision, 0) + 1
                        )
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
            "bit_histogram": bit_histogram,
            "precision_histogram": precision_histogram,
        }
