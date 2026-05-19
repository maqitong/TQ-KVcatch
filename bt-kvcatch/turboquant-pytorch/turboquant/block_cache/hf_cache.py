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
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
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
    num_layers: Optional[int] = None
    protected_layers: int = 0
    protected_key_bits: Optional[float] = 8
    protected_value_bits: Optional[float] = 8
    group_size: int = 128
    key_group_size: Optional[int] = None
    value_group_size: Optional[int] = None
    clipping: float = 0.92
    reorder_file: Optional[str] = None
    reorder_meta: Optional[dict[str, Any]] = None
    max_cached_decompressed_blocks: int = 0


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
        self.tq_reorder_idx: Optional[dict[str, torch.Tensor]] = None
        self.bit_allocator: Optional[PageBitAllocator] = None
        self._k_compressors: dict[float, BlockMSECompressor] = {}
        self._v_compressors: dict[float, BlockMSECompressor] = {}
        self._decompressed_cache: OrderedDict[
            tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
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

    def _init_runtime(
        self,
        *,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.table = BlockTable(
            block_size=self.cfg.block_size,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            batch_size=batch_size,
        )
        seed_base = self.cfg.seed + self.layer_idx * 1000
        if self.cfg.quant_backend not in ("turboquant", "skvq"):
            raise ValueError(f"unknown quant_backend: {self.cfg.quant_backend}")

        if self.cfg.quant_backend == "skvq":
            self.tq_reorder_idx = None
            reorder_idx, group_st_idx = self._layer_reorder_meta()
            self.skvq_compressor = SKVQPageCompressor(
                head_dim=head_dim,
                n_kv_heads=n_kv_heads,
                group_size=self.cfg.group_size,
                key_group_size=self.cfg.key_group_size,
                value_group_size=self.cfg.value_group_size,
                clipping=self.cfg.clipping,
                reorder_idx=reorder_idx,
                group_st_idx=group_st_idx,
            )
        else:
            self.tq_reorder_idx, _ = self._layer_reorder_meta()
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

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        B, H, _S, D = key_states.shape
        self._init_runtime(
            batch_size=B,
            n_kv_heads=H,
            head_dim=D,
            dtype=key_states.dtype,
            device=key_states.device,
        )

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

    def _tq_reorder_states(self, states: torch.Tensor, ttype: str) -> tuple[torch.Tensor, bool]:
        if self.tq_reorder_idx is None or ttype not in self.tq_reorder_idx:
            return states, False

        B, H, S, D = states.shape
        hidden = H * D
        idx = self.tq_reorder_idx[ttype].long().to(states.device)
        if idx.numel() != hidden:
            raise ValueError(f"TurboQuant reorder index length {idx.numel()} != hidden {hidden}")

        flat = states.transpose(1, 2).reshape(B, S, hidden)
        reordered = flat.index_select(-1, idx)
        return reordered.reshape(B, S, H, D).transpose(1, 2).contiguous(), True

    def _tq_inverse_reorder_states(self, states: torch.Tensor, ttype: str) -> torch.Tensor:
        if self.tq_reorder_idx is None or ttype not in self.tq_reorder_idx:
            raise ValueError(
                "TurboQuant compressed block was reordered, but reorder metadata is missing"
            )

        B, H, S, D = states.shape
        hidden = H * D
        idx = self.tq_reorder_idx[ttype].long().to(states.device)
        if idx.numel() != hidden:
            raise ValueError(f"TurboQuant reorder index length {idx.numel()} != hidden {hidden}")

        inv_idx = idx.argsort()
        flat = states.transpose(1, 2).reshape(B, S, hidden)
        restored = flat.index_select(-1, inv_idx)
        return restored.reshape(B, S, H, D).transpose(1, 2).contiguous()

    def _apply_layer_protection(self, blk: KVBlock, k_bits: float, v_bits: float) -> tuple[float, float]:
        if self.cfg.protected_layers <= 0:
            return k_bits, v_bits

        in_first = self.layer_idx < self.cfg.protected_layers
        in_last = (
            self.cfg.num_layers is not None
            and self.layer_idx >= self.cfg.num_layers - self.cfg.protected_layers
        )
        if not (in_first or in_last):
            return k_bits, v_bits

        protected_k = (
            self.cfg.protected_key_bits
            if self.cfg.protected_key_bits is not None
            else k_bits
        )
        protected_v = (
            self.cfg.protected_value_bits
            if self.cfg.protected_value_bits is not None
            else v_bits
        )
        if blk.page_meta is None:
            blk.page_meta = {}
        else:
            blk.page_meta = dict(blk.page_meta)
        blk.page_meta.update(
            {
                "protected_layer": True,
                "pre_protection_bits": (k_bits, v_bits),
                "precision": "protected",
            }
        )
        return float(protected_k), float(protected_v)

    def _invalidate_decompressed_cache(self, block_idx: Optional[int] = None) -> None:
        if block_idx is None:
            self._decompressed_cache.clear()
            return
        for key in list(self._decompressed_cache):
            if key[0] == block_idx:
                del self._decompressed_cache[key]

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
                k_bits, v_bits = self._apply_layer_protection(blk, k_bits, v_bits)
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
                    work_k, reordered_k = self._tq_reorder_states(blk.fp16_k, "k")
                    work_v, reordered_v = self._tq_reorder_states(blk.fp16_v, "v")
                    ck = self._get_turboquant_compressor("k", k_bits).compress(
                        work_k
                    )
                    cv = self._get_turboquant_compressor("v", v_bits).compress(
                        work_v
                    )
                    ck["backend"] = "turboquant"
                    cv["backend"] = "turboquant"
                    ck["tq_reordered"] = reordered_k
                    cv["tq_reordered"] = reordered_v
                    ck["ttype"] = "k"
                    cv["ttype"] = "v"
                blk.to_compressed(ck, cv)
                self._invalidate_decompressed_cache(blk.block_idx)

        return self._materialize(key_states.dtype)

    def _decompress_block(
        self, blk: KVBlock, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            blk.block_idx,
            str(dtype),
            str(self.device if self.device is not None else "cpu"),
        )
        if self.cfg.max_cached_decompressed_blocks > 0 and cache_key in self._decompressed_cache:
            cached = self._decompressed_cache.pop(cache_key)
            self._decompressed_cache[cache_key] = cached
            return cached

        if blk.compressed_k.get("backend") == "skvq":
            k = self.skvq_compressor.decompress(blk.compressed_k).to(dtype)
            v = self.skvq_compressor.decompress(blk.compressed_v).to(dtype)
        else:
            k_bits = blk.key_bits if blk.key_bits is not None else self.cfg.key_bits
            v_bits = blk.value_bits if blk.value_bits is not None else self.cfg.value_bits
            k = self._get_turboquant_compressor("k", k_bits).decompress(blk.compressed_k)
            v = self._get_turboquant_compressor("v", v_bits).decompress(blk.compressed_v)
            if blk.compressed_k.get("tq_reordered", False):
                k = self._tq_inverse_reorder_states(k, "k")
            if blk.compressed_v.get("tq_reordered", False):
                v = self._tq_inverse_reorder_states(v, "v")
            k = k.to(dtype)
            v = v.to(dtype)

        if self.cfg.max_cached_decompressed_blocks > 0:
            self._decompressed_cache[cache_key] = (k, v)
            while len(self._decompressed_cache) > self.cfg.max_cached_decompressed_blocks:
                self._decompressed_cache.popitem(last=False)
        return k, v

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

    def record_attention(self, attn_weights: torch.Tensor) -> None:
        """Accumulate attention mass per page.

        Args:
            attn_weights: attention probabilities with key length on the last
                dimension, commonly shaped (B, H, Q, S) or (B, H, S).
        """
        if self.table is None or attn_weights is None:
            return
        if attn_weights.ndim < 2:
            raise ValueError("attention weights must have key length on the last dim")

        total_len = self.table.total_len
        key_len = int(attn_weights.shape[-1])
        if key_len <= 0 or total_len <= 0:
            return

        # If the supplied attention window is shorter than the cache, align it
        # to the cache tail. This matches sliding-window attention outputs.
        offset = max(0, total_len - key_len)
        reduce_dims = tuple(range(attn_weights.ndim - 1))
        token_scores = attn_weights.detach().float().sum(dim=reduce_dims).cpu()

        cursor = 0
        for blk in self.table.blocks:
            blk_start = cursor
            blk_end = cursor + blk.current_len
            cursor = blk_end

            overlap_start = max(blk_start, offset)
            overlap_end = min(blk_end, offset + key_len)
            if overlap_start >= overlap_end:
                continue

            local_start = overlap_start - offset
            local_end = overlap_end - offset
            mass = float(token_scores[local_start:local_end].sum().item())
            count = float(local_end - local_start)

            meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
            meta["attention_score"] = float(meta.get("attention_score", 0.0)) + mass
            meta["attention_count"] = float(meta.get("attention_count", 0.0)) + count
            blk.page_meta = meta

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if torch.is_tensor(query_length):
            query_length = int(query_length.item())
        return self.get_seq_length() + int(query_length), 0

    def get_seq_length(self) -> int:
        return self.table.total_len if self.table is not None else 0

    def get_max_cache_shape(self) -> int:
        return -1

    # ----- HF v5 hooks (optional but useful) -----

    def reset(self) -> None:
        if self.table is not None:
            self.table.blocks.clear()
        self._invalidate_decompressed_cache()
        self.is_initialized = False

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self.table is None:
            return
        self._invalidate_decompressed_cache()
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
        self._invalidate_decompressed_cache()

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
        self._invalidate_decompressed_cache()
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
        self._invalidate_decompressed_cache()
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

    def state_dict(self) -> dict[str, Any]:
        """Serialize one layer's block table and compressed payloads."""
        if self.table is None:
            return {
                "is_initialized": False,
                "layer_idx": self.layer_idx,
            }

        blocks = []
        for blk in self.table.blocks:
            blocks.append(
                {
                    "block_idx": blk.block_idx,
                    "state": blk.state.value,
                    "current_len": blk.current_len,
                    "fp16_k": blk.fp16_k,
                    "fp16_v": blk.fp16_v,
                    "compressed_k": blk.compressed_k,
                    "compressed_v": blk.compressed_v,
                    "importance": blk.importance,
                    "key_bits": blk.key_bits,
                    "value_bits": blk.value_bits,
                    "page_meta": blk.page_meta,
                }
            )

        return {
            "is_initialized": self.is_initialized,
            "layer_idx": self.layer_idx,
            "dtype": self.dtype,
            "device": str(self.device),
            "decompressed_cache_entries": len(self._decompressed_cache),
            "table": {
                "block_size": self.table.block_size,
                "head_dim": self.table.head_dim,
                "n_kv_heads": self.table.n_kv_heads,
                "batch_size": self.table.batch_size,
                "blocks": blocks,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore one layer from `state_dict()` output."""
        self.layer_idx = int(state.get("layer_idx", self.layer_idx))
        if not state.get("is_initialized", False):
            self.table = None
            self.is_initialized = False
            return

        table_state = state["table"]
        dtype = state.get("dtype") or torch.float16
        first_device = torch.device("cpu")
        for block_state in table_state.get("blocks", []):
            for key in ("fp16_k", "compressed_k", "compressed_v"):
                value = block_state.get(key)
                if torch.is_tensor(value):
                    first_device = value.device
                    break
                if isinstance(value, dict):
                    tensor = next((v for v in value.values() if torch.is_tensor(v)), None)
                    if tensor is not None:
                        first_device = tensor.device
                        break

        self._k_compressors.clear()
        self._v_compressors.clear()
        self._invalidate_decompressed_cache()
        self._init_runtime(
            batch_size=int(table_state["batch_size"]),
            n_kv_heads=int(table_state["n_kv_heads"]),
            head_dim=int(table_state["head_dim"]),
            dtype=dtype,
            device=first_device,
        )
        self.table.blocks = []

        for block_state in table_state.get("blocks", []):
            blk = KVBlock(
                block_idx=int(block_state["block_idx"]),
                block_size=int(table_state["block_size"]),
                head_dim=int(table_state["head_dim"]),
                n_kv_heads=int(table_state["n_kv_heads"]),
                batch_size=int(table_state["batch_size"]),
                state=BlockState(block_state["state"]),
                current_len=int(block_state["current_len"]),
                fp16_k=block_state.get("fp16_k"),
                fp16_v=block_state.get("fp16_v"),
                compressed_k=block_state.get("compressed_k"),
                compressed_v=block_state.get("compressed_v"),
                importance=float(block_state.get("importance", 0.0)),
                key_bits=block_state.get("key_bits"),
                value_bits=block_state.get("value_bits"),
                page_meta=block_state.get("page_meta"),
            )
            self.table.blocks.append(blk)


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

    def record_attention(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        """Record attention probabilities for later page-importance scoring.

        This explicit API is intentionally conservative: evaluation scripts or
        future model hooks can pass attention probabilities here without
        monkey-patching attention modules.
        """
        if layer_idx >= len(self.layers):
            return
        self.layers[layer_idx].record_attention(attn_weights)

    def record_attentions(self, attentions: Any) -> None:
        """Record a HuggingFace attentions object.

        Handles both ordinary forward outputs shaped like
        `tuple[layer](B, H, Q, S)` and generation outputs shaped like
        `tuple[step][layer](B, H, Q, S)`.
        """
        if attentions is None:
            return
        if torch.is_tensor(attentions):
            self.record_attention(0, attentions)
            return
        if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
            return

        first = attentions[0]
        if isinstance(first, (list, tuple)):
            for step_attentions in attentions:
                self.record_attentions(step_attentions)
            return

        for layer_idx, attn_weights in enumerate(attentions):
            if torch.is_tensor(attn_weights):
                self.record_attention(layer_idx, attn_weights)

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

    def state_dict(self) -> dict[str, Any]:
        """Serialize the cache payload.

        Restore into a `BlockKVCache` constructed with a compatible
        `BlockCacheConfig`. The config snapshot is included for diagnostics,
        but `load_state_dict()` intentionally does not replace `self.config`.
        """
        config_snapshot = asdict(self.config)
        config_snapshot["policy"] = type(self.config.policy).__name__
        return {
            "format": "BlockKVCache.v1",
            "seen_tokens": self._seen_tokens,
            "config": config_snapshot,
            "layers": [layer.state_dict() for layer in self.layers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") not in (None, "BlockKVCache.v1"):
            raise ValueError(f"unsupported BlockKVCache state format: {state.get('format')}")

        self.layers = []
        for layer_state in state.get("layers", []):
            layer = BlockCacheLayer(
                self.config,
                layer_idx=int(layer_state.get("layer_idx", len(self.layers))),
                reorder_meta=self._reorder_meta,
            )
            layer.load_state_dict(layer_state)
            self.layers.append(layer)
        self._seen_tokens = int(state.get("seen_tokens", 0))
