"""Per-layer block cache used by ``BlockKVCache``."""

from __future__ import annotations

import warnings
from collections import OrderedDict, deque
from typing import Any, Optional

import torch

from .backends import PageQuantBackend, TurboQuantPageBackend, build_page_backend
from .bit_allocator import (
    FixedPageBitAllocator,
    PageBitAllocator,
    TopRatioPageBitAllocator,
)
from .blocks import BlockState, BlockTable, KVBlock
from .config import BlockCacheConfig
from .hf_compat import HF_AVAILABLE, HFCacheLayerMixin
from .page_importance import build_page_importance_scorer


_ATTENTION_IMPORTANCE_NAMES = {"attention", "attention_score", "attn", "attn_score"}


class BlockCacheLayer(HFCacheLayerMixin):
    """A single transformer layer's block-structured KV cache.

    Conforms to ``transformers.cache_utils.CacheLayerMixin``:
      - ``lazy_initialization(k, v)`` discovers shapes on first call.
      - ``update(k, v)`` appends and returns the materialised dense tensors.
      - ``get_seq_length()`` returns the logical token count.
      - ``get_max_cache_shape()`` returns -1 (dynamic / unbounded).
      - ``get_mask_sizes(query_length)`` returns the shape needed for the
        causal-mask path inside attention.
    """

    is_sliding = False
    is_compileable = False

    def __init__(
        self,
        config: BlockCacheConfig,
        layer_idx: int = 0,
        reorder_meta: Optional[dict[str, Any]] = None,
    ):
        if HF_AVAILABLE:
            super().__init__()
        else:
            self.is_initialized = False
        self.cfg = config
        self.layer_idx = layer_idx
        self.reorder_meta = reorder_meta
        self.table: Optional[BlockTable] = None
        self.page_backend: Optional[PageQuantBackend] = None
        self.skvq_compressor: Optional[Any] = None
        self.tq_reorder_idx: Optional[dict[str, torch.Tensor]] = None
        self.bit_allocator: Optional[PageBitAllocator] = None
        self._decompressed_cache: OrderedDict[
            tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._pending_quant_blocks: deque[int] = deque()
        self._pending_quant_block_ids: set[int] = set()
        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None

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
        reorder = self._layer_reorder_meta()
        self.page_backend = build_page_backend(
            self.cfg.quant_backend,
            config=self.cfg,
            layer_idx=self.layer_idx,
            batch_size=batch_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            reorder=reorder,
        )
        self.tq_reorder_idx = (
            self.page_backend.reorder_idx
            if isinstance(self.page_backend, TurboQuantPageBackend)
            else None
        )
        self.skvq_compressor = getattr(self.page_backend, "compressor", None)

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

    def _uses_attention_importance(self) -> bool:
        return (
            self.cfg.mixed_precision
            and self.cfg.importance_metric.replace("-", "_") in _ATTENTION_IMPORTANCE_NAMES
        )

    def _has_attention_score(self, blk: KVBlock) -> bool:
        return (
            isinstance(blk.page_meta, dict)
            and float(blk.page_meta.get("attention_count", 0.0)) > 0.0
        )

    def _uses_budgeted_quantization(self) -> bool:
        return self.cfg.quant_budget_per_update is not None

    def _attention_ready_blocks(self, blocks: list[KVBlock]) -> list[KVBlock]:
        if not self._uses_attention_importance():
            return blocks

        ready: list[KVBlock] = []
        for blk in blocks:
            if self._has_attention_score(blk):
                ready.append(blk)
                continue
            meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
            meta.update(
                {
                    "allocator": "top_ratio",
                    "importance_metric": "attention_score",
                    "precision": "deferred",
                    "defer_reason": "waiting_for_attention_score",
                }
            )
            blk.page_meta = meta
        return ready

    def _mark_pending_quant(self, blk: KVBlock) -> None:
        meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
        meta["quant_status"] = "pending"
        if self._uses_attention_importance() and not self._has_attention_score(blk):
            meta.update(
                {
                    "allocator": "top_ratio",
                    "importance_metric": "attention_score",
                    "precision": "deferred",
                    "defer_reason": "waiting_for_attention_score",
                }
            )
        blk.page_meta = meta

    def _enqueue_quant_blocks(self, blocks: list[KVBlock]) -> None:
        for blk in blocks:
            if blk.state != BlockState.SEALED:
                continue
            if blk.block_idx in self._pending_quant_block_ids:
                continue
            self._mark_pending_quant(blk)
            self._pending_quant_blocks.append(blk.block_idx)
            self._pending_quant_block_ids.add(blk.block_idx)

    def _ready_pending_quant_blocks(self) -> list[KVBlock]:
        if self.table is None:
            return []
        ready: list[KVBlock] = []
        valid_queue: deque[int] = deque()
        valid_ids: set[int] = set()

        for block_idx in self._pending_quant_blocks:
            if block_idx >= len(self.table.blocks):
                continue
            blk = self.table.blocks[block_idx]
            if blk.state != BlockState.SEALED:
                continue

            valid_queue.append(block_idx)
            valid_ids.add(block_idx)
            if self._uses_attention_importance() and not self._has_attention_score(blk):
                self._mark_pending_quant(blk)
                continue
            ready.append(blk)

        self._pending_quant_blocks = valid_queue
        self._pending_quant_block_ids = valid_ids
        return ready

    def _compress_blocks(
        self,
        blocks: list[KVBlock],
        bit_assignments: Optional[dict[int, tuple[float, float]]] = None,
    ) -> None:
        if not blocks:
            return
        if self.bit_allocator is None or self.page_backend is None or self.table is None:
            raise RuntimeError("cache layer was not initialized")

        if bit_assignments is None:
            bit_assignments = self.bit_allocator.assign_many(
                blocks, self.table, self.layer_idx
            )
        for blk in blocks:
            if blk.state != BlockState.SEALED:
                continue
            k_bits, v_bits = bit_assignments[blk.block_idx]
            k_bits, v_bits = self._apply_layer_protection(blk, k_bits, v_bits)
            blk.key_bits = k_bits
            blk.value_bits = v_bits
            ck, cv = self.page_backend.compress(
                blk.fp16_k,
                blk.fp16_v,
                key_bits=k_bits,
                value_bits=v_bits,
                layer_idx=self.layer_idx,
            )
            blk.to_compressed(ck, cv)
            meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
            meta["quant_status"] = "compressed"
            meta.pop("defer_reason", None)
            blk.page_meta = meta
            self._invalidate_decompressed_cache(blk.block_idx)

    def _step_pending_quantization(self) -> None:
        budget = self.cfg.quant_budget_per_update
        if budget is None or budget <= 0 or self.table is None:
            return
        if self.bit_allocator is None:
            raise RuntimeError("cache layer was not initialized")

        ready_pool = self._ready_pending_quant_blocks()
        if not ready_pool:
            return

        # Assign precision across all currently-ready queued pages, then let the
        # cursor compress only the oldest `budget` pages. This keeps PageMix's
        # high/low decision tied to the pending pool instead of the tiny batch
        # that happens to fit this step's budget.
        bit_assignments = self.bit_allocator.assign_many(
            ready_pool, self.table, self.layer_idx
        )
        ready_ids = {blk.block_idx for blk in ready_pool}
        chosen_ids: set[int] = set()
        for block_idx in self._pending_quant_blocks:
            if block_idx in ready_ids:
                chosen_ids.add(block_idx)
                if len(chosen_ids) >= budget:
                    break

        to_compress = [
            self.table.blocks[block_idx]
            for block_idx in self._pending_quant_blocks
            if block_idx in chosen_ids
        ]
        self._compress_blocks(to_compress, bit_assignments=bit_assignments)

        self._pending_quant_blocks = deque(
            block_idx
            for block_idx in self._pending_quant_blocks
            if block_idx not in chosen_ids
        )
        self._pending_quant_block_ids.difference_update(chosen_ids)

    def _policy_candidates_for_sealed_blocks(self) -> list[KVBlock]:
        if self.table is None:
            return []
        sealed = [
            blk for blk in self.table.blocks if blk.state == BlockState.SEALED
        ]
        if not sealed:
            return []
        return self.cfg.policy.on_seal(sealed, self.table)

    def _schedule_or_compress_blocks(self, blocks: list[KVBlock]) -> None:
        if self._uses_budgeted_quantization():
            self._enqueue_quant_blocks(blocks)
            self._step_pending_quantization()
            return

        ready = self._attention_ready_blocks(blocks)
        self._compress_blocks(ready)

    def _compress_attention_ready_sealed_blocks(self) -> None:
        if not self._uses_attention_importance() or self.table is None:
            return
        self._schedule_or_compress_blocks(self._policy_candidates_for_sealed_blocks())

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
    ):
        """Compatibility helper for older tests/extensions."""
        if not isinstance(self.page_backend, TurboQuantPageBackend):
            raise RuntimeError("current page backend is not TurboQuant")
        return self.page_backend.get_compressor(ttype, bits)

    def _apply_layer_protection(
        self, blk: KVBlock, k_bits: float, v_bits: float
    ) -> tuple[float, float]:
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
        blk.page_meta = dict(blk.page_meta) if blk.page_meta is not None else {}
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
        if self.page_backend is None:
            raise RuntimeError("page backend was not initialized")

        sealed = self.table.append(key_states, value_states)
        if sealed:
            to_compress = self.cfg.policy.on_seal(sealed, self.table)
            self._schedule_or_compress_blocks(to_compress)
        elif self._uses_budgeted_quantization():
            self._step_pending_quantization()

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
        if self.page_backend is None:
            raise RuntimeError("page backend was not initialized")

        k_bits = blk.key_bits if blk.key_bits is not None else self.cfg.key_bits
        v_bits = blk.value_bits if blk.value_bits is not None else self.cfg.value_bits
        k, v = self.page_backend.decompress(
            blk.compressed_k,
            blk.compressed_v,
            key_bits=k_bits,
            value_bits=v_bits,
            dtype=dtype,
        )

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
        """Accumulate attention mass per page."""
        if self.table is None or attn_weights is None:
            return
        if attn_weights.ndim < 2:
            raise ValueError("attention weights must have key length on the last dim")

        total_len = self.table.total_len
        key_len = int(attn_weights.shape[-1])
        if key_len <= 0 or total_len <= 0:
            return

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
        self._compress_attention_ready_sealed_blocks()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if torch.is_tensor(query_length):
            query_length = int(query_length.item())
        return self.get_seq_length() + int(query_length), 0

    def get_seq_length(self) -> int:
        return self.table.total_len if self.table is not None else 0

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        if self.table is not None:
            self.table.blocks.clear()
        self._invalidate_decompressed_cache()
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
        self.page_backend = None
        self.skvq_compressor = None
        self.tq_reorder_idx = None
        self.is_initialized = False

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self.table is None:
            return
        self._invalidate_decompressed_cache()
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
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
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()

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
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
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
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
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
            "pending_quant_blocks": list(self._pending_quant_blocks),
            "table": {
                "block_size": self.table.block_size,
                "head_dim": self.table.head_dim,
                "n_kv_heads": self.table.n_kv_heads,
                "batch_size": self.table.batch_size,
                "blocks": blocks,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore one layer from ``state_dict()`` output."""
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

        pending = []
        for block_idx in state.get("pending_quant_blocks", []):
            idx = int(block_idx)
            if 0 <= idx < len(self.table.blocks):
                blk = self.table.blocks[idx]
                if blk.state == BlockState.SEALED:
                    pending.append(idx)
        self._pending_quant_blocks = deque(pending)
        self._pending_quant_block_ids = set(pending)
