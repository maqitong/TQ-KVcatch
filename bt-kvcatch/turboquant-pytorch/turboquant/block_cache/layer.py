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
        self._mat_k: Optional[torch.Tensor] = None
        self._mat_v: Optional[torch.Tensor] = None
        self._mat_sig: list[tuple[Any, ...]] = []
        self._mat_dtype: Optional[torch.dtype] = None
        self._mat_device: Optional[torch.device] = None
        self._tq_compressed_runs: list[dict[str, Any]] = []
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
                run_aware=self.cfg.pagemix_run_aware,
            )
        else:
            self.bit_allocator = FixedPageBitAllocator(
                self.cfg.key_bits, self.cfg.value_bits
            )
        self._invalidate_materialized_cache()
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

    def _uses_base_residual_mixed_precision(self) -> bool:
        return (
            self.cfg.mixed_precision
            and self.cfg.mixed_precision_mode == "base_residual"
            and isinstance(self.page_backend, TurboQuantPageBackend)
            and self.cfg.granularity == "per-vector"
        )

    def _is_high_precision_assignment(self, k_bits: float, v_bits: float) -> bool:
        return (
            float(k_bits) == float(self.cfg.high_key_bits)
            and float(v_bits) == float(self.cfg.high_value_bits)
        )

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
        if self._uses_base_residual_mixed_precision():
            self._compress_blocks_base_residual(blocks, bit_assignments)
            return

        prepared: list[tuple[KVBlock, float, float]] = []
        for blk in blocks:
            if blk.state != BlockState.SEALED:
                continue
            k_bits, v_bits = bit_assignments[blk.block_idx]
            k_bits, v_bits = self._apply_layer_protection(blk, k_bits, v_bits)
            blk.key_bits = k_bits
            blk.value_bits = v_bits
            prepared.append((blk, k_bits, v_bits))

        if self._try_compress_turboquant_batched(prepared):
            return

        for blk, k_bits, v_bits in prepared:
            ck, cv = self.page_backend.compress(
                blk.fp16_k,
                blk.fp16_v,
                key_bits=k_bits,
                value_bits=v_bits,
                layer_idx=self.layer_idx,
            )
            self._finalize_compressed_block(blk, ck, cv)

    def _compress_blocks_base_residual(
        self,
        blocks: list[KVBlock],
        bit_assignments: dict[int, tuple[float, float]],
    ) -> None:
        if not isinstance(self.page_backend, TurboQuantPageBackend):
            raise RuntimeError("base_residual mode requires TurboQuant backend")

        direct: list[tuple[KVBlock, float, float]] = []
        base: list[tuple[KVBlock, bool]] = []
        base_k_bits = float(self.cfg.low_key_bits)
        base_v_bits = float(self.cfg.low_value_bits)

        for blk in blocks:
            if blk.state != BlockState.SEALED:
                continue
            assigned_k, assigned_v = bit_assignments[blk.block_idx]
            protected_k, protected_v = self._apply_layer_protection(
                blk, assigned_k, assigned_v
            )
            is_protected = isinstance(blk.page_meta, dict) and blk.page_meta.get(
                "protected_layer", False
            )
            if is_protected:
                blk.key_bits = protected_k
                blk.value_bits = protected_v
                direct.append((blk, protected_k, protected_v))
                continue

            has_residual = self._is_high_precision_assignment(assigned_k, assigned_v)
            blk.key_bits = base_k_bits
            blk.value_bits = base_v_bits
            meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
            meta.update(
                {
                    "mixed_precision_mode": "base_residual",
                    "base_bits": (base_k_bits, base_v_bits),
                    "precision": "base_residual" if has_residual else "base",
                }
            )
            if has_residual:
                meta["residual_bits"] = (
                    float(self.cfg.residual_key_bits),
                    float(self.cfg.residual_value_bits),
                )
            blk.page_meta = meta
            base.append((blk, has_residual))

        if direct:
            self._try_compress_turboquant_batched(direct)
        if base:
            self._compress_base_residual_group(base, base_k_bits, base_v_bits)

    def _compress_base_residual_group(
        self,
        prepared: list[tuple[KVBlock, bool]],
        base_k_bits: float,
        base_v_bits: float,
    ) -> None:
        if self.page_backend is None:
            raise RuntimeError("page backend was not initialized")
        blocks = [blk for blk, _has_residual in prepared]
        lengths = [blk.current_len for blk in blocks]
        batched_k = torch.cat([blk.fp16_k for blk in blocks], dim=2)
        batched_v = torch.cat([blk.fp16_v for blk in blocks], dim=2)
        ck_all, cv_all = self.page_backend.compress(
            batched_k,
            batched_v,
            key_bits=base_k_bits,
            value_bits=base_v_bits,
            layer_idx=self.layer_idx,
        )
        if self._can_store_turboquant_runs():
            split_k, split_v = self._register_turboquant_run(
                blocks, ck_all, cv_all, base_k_bits, base_v_bits
            )
        else:
            split_k = self._split_turboquant_compressed(ck_all, lengths)
            split_v = self._split_turboquant_compressed(cv_all, lengths)

        self._attach_batched_residual_payloads(
            prepared,
            split_k,
            split_v,
            ck_all,
            cv_all,
            base_k_bits=base_k_bits,
            base_v_bits=base_v_bits,
            source_k=batched_k,
            source_v=batched_v,
        )

        for blk, ck, cv in zip(blocks, split_k, split_v):
            self._finalize_compressed_block(blk, ck, cv)

    def _attach_batched_residual_payloads(
        self,
        prepared: list[tuple[KVBlock, bool]],
        split_k: list[dict[str, Any]],
        split_v: list[dict[str, Any]],
        base_compressed_k: dict[str, Any],
        base_compressed_v: dict[str, Any],
        *,
        base_k_bits: float,
        base_v_bits: float,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
    ) -> None:
        if self.page_backend is None or not isinstance(
            self.page_backend, TurboQuantPageBackend
        ):
            raise RuntimeError("base_residual mode requires TurboQuant backend")
        residual_entries: list[tuple[int, KVBlock, int, int]] = []
        start = 0
        for idx, (blk, has_residual) in enumerate(prepared):
            end = start + blk.current_len
            if has_residual:
                residual_entries.append((idx, blk, start, end))
            start = end
        if not residual_entries:
            return

        base_k, base_v = self.page_backend.decompress(
            base_compressed_k,
            base_compressed_v,
            key_bits=base_k_bits,
            value_bits=base_v_bits,
            dtype=source_k.dtype,
        )

        residual_k_bits = float(self.cfg.residual_key_bits)
        residual_v_bits = float(self.cfg.residual_value_bits)
        if residual_k_bits > 0:
            residual_parts = [
                (source_k[:, :, start:end, :] - base_k[:, :, start:end, :]).contiguous()
                for _idx, _blk, start, end in residual_entries
            ]
            residual_k = torch.cat(residual_parts, dim=2)
            rk_all = self.page_backend.get_compressor("k", residual_k_bits).compress(
                residual_k
            )
            rk_all.update(
                {
                    "backend": "turboquant_residual",
                    "ttype": "k_residual",
                    "layer_idx": self.layer_idx,
                }
            )
            split_rk = self._split_turboquant_compressed(
                rk_all, [blk.current_len for _idx, blk, _start, _end in residual_entries]
            )
            for (idx, _blk, _start, _end), rk in zip(residual_entries, split_rk):
                split_k[idx]["__residual"] = rk
                split_k[idx]["__residual_bits"] = residual_k_bits

        if residual_v_bits > 0:
            residual_parts = [
                (source_v[:, :, start:end, :] - base_v[:, :, start:end, :]).contiguous()
                for _idx, _blk, start, end in residual_entries
            ]
            residual_v = torch.cat(residual_parts, dim=2)
            rv_all = self.page_backend.get_compressor("v", residual_v_bits).compress(
                residual_v
            )
            rv_all.update(
                {
                    "backend": "turboquant_residual",
                    "ttype": "v_residual",
                    "layer_idx": self.layer_idx,
                }
            )
            split_rv = self._split_turboquant_compressed(
                rv_all, [blk.current_len for _idx, blk, _start, _end in residual_entries]
            )
            for (idx, _blk, _start, _end), rv in zip(residual_entries, split_rv):
                split_v[idx]["__residual"] = rv
                split_v[idx]["__residual_bits"] = residual_v_bits

    def _finalize_compressed_block(
        self,
        blk: KVBlock,
        compressed_k: dict[str, Any],
        compressed_v: dict[str, Any],
    ) -> None:
        blk.to_compressed(compressed_k, compressed_v)
        meta = dict(blk.page_meta) if isinstance(blk.page_meta, dict) else {}
        meta["quant_status"] = "compressed"
        meta.pop("defer_reason", None)
        blk.page_meta = meta
        self._invalidate_decompressed_cache(blk.block_idx)

    def _try_compress_turboquant_batched(
        self, prepared: list[tuple[KVBlock, float, float]]
    ) -> bool:
        if not prepared:
            return True
        if not isinstance(self.page_backend, TurboQuantPageBackend):
            return False
        if self.cfg.granularity != "per-vector":
            return False

        groups: dict[tuple[float, float], list[KVBlock]] = {}
        for blk, k_bits, v_bits in prepared:
            groups.setdefault((float(k_bits), float(v_bits)), []).append(blk)

        for (k_bits, v_bits), group in groups.items():
            if len(group) == 1:
                blk = group[0]
                ck, cv = self.page_backend.compress(
                    blk.fp16_k,
                    blk.fp16_v,
                    key_bits=k_bits,
                    value_bits=v_bits,
                    layer_idx=self.layer_idx,
                )
                self._finalize_compressed_block(blk, ck, cv)
                continue

            lengths = [blk.current_len for blk in group]
            batched_k = torch.cat([blk.fp16_k for blk in group], dim=2)
            batched_v = torch.cat([blk.fp16_v for blk in group], dim=2)
            ck_all, cv_all = self.page_backend.compress(
                batched_k,
                batched_v,
                key_bits=k_bits,
                value_bits=v_bits,
                layer_idx=self.layer_idx,
            )
            if self._can_store_turboquant_runs():
                split_k, split_v = self._register_turboquant_run(
                    group, ck_all, cv_all, k_bits, v_bits
                )
            else:
                split_k = self._split_turboquant_compressed(ck_all, lengths)
                split_v = self._split_turboquant_compressed(cv_all, lengths)
            for blk, ck, cv in zip(group, split_k, split_v):
                self._finalize_compressed_block(blk, ck, cv)
        return True

    def _can_store_turboquant_runs(self) -> bool:
        return (
            isinstance(self.page_backend, TurboQuantPageBackend)
            and self.cfg.granularity == "per-vector"
        )

    def _register_turboquant_run(
        self,
        blocks: list[KVBlock],
        compressed_k: dict[str, Any],
        compressed_v: dict[str, Any],
        key_bits: float,
        value_bits: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        run_id = len(self._tq_compressed_runs)
        lengths = [blk.current_len for blk in blocks]
        self._tq_compressed_runs.append(
            {
                "compressed_k": compressed_k,
                "compressed_v": compressed_v,
                "key_bits": float(key_bits),
                "value_bits": float(value_bits),
                "block_indices": [blk.block_idx for blk in blocks],
                "lengths": lengths,
                "active": True,
            }
        )

        split_k: list[dict[str, Any]] = []
        split_v: list[dict[str, Any]] = []
        B, H, _S, D = compressed_k["shape"]
        start = 0
        for length in lengths:
            split_k.append(
                self._turboquant_run_proxy(compressed_k, run_id, start, length, B, H, D)
            )
            split_v.append(
                self._turboquant_run_proxy(compressed_v, run_id, start, length, B, H, D)
            )
            start += length
        return split_k, split_v

    @staticmethod
    def _turboquant_run_proxy(
        compressed: dict[str, Any],
        run_id: int,
        start: int,
        length: int,
        B: int,
        H: int,
        D: int,
    ) -> dict[str, Any]:
        return {
            "backend": compressed.get("backend"),
            "tq_reordered": compressed.get("tq_reordered", False),
            "ttype": compressed.get("ttype"),
            "layer_idx": compressed.get("layer_idx"),
            "granularity": compressed.get("granularity", "per-vector"),
            "shape": (B, H, length, D),
            "__run_id": run_id,
            "__run_start": start,
        }

    @staticmethod
    def _split_turboquant_compressed(
        compressed: dict[str, Any], lengths: list[int]
    ) -> list[dict[str, Any]]:
        B, H, _S, D = compressed["shape"]
        out: list[dict[str, Any]] = []
        start = 0
        for length in lengths:
            end = start + length
            part: dict[str, Any] = {}
            for key, value in compressed.items():
                if key == "shape":
                    part[key] = (B, H, length, D)
                elif torch.is_tensor(value) and value.ndim >= 3 and value.shape[2] == _S:
                    part[key] = value[:, :, start:end, ...]
                else:
                    part[key] = value
            out.append(part)
            start = end
        return out

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

    def _invalidate_materialized_cache(self) -> None:
        self._mat_k = None
        self._mat_v = None
        self._mat_sig = []
        self._mat_dtype = None
        self._mat_device = None

    def _block_signature(self, blk: KVBlock) -> tuple[Any, ...]:
        if blk.state == BlockState.COMPRESSED:
            return (
                blk.block_idx,
                blk.state.value,
                blk.current_len,
                blk.key_bits,
                blk.value_bits,
                id(blk.compressed_k),
                id(blk.compressed_v),
            )
        return (
            blk.block_idx,
            blk.state.value,
            blk.current_len,
            id(blk.fp16_k),
            id(blk.fp16_v),
        )

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
        compressed_k, compressed_v = self._compressed_payload_for_block(blk)
        k, v = self.page_backend.decompress(
            compressed_k,
            compressed_v,
            key_bits=k_bits,
            value_bits=v_bits,
            dtype=dtype,
        )
        k, v = self._apply_residual_payloads(blk, k, v, dtype)

        if self.cfg.max_cached_decompressed_blocks > 0:
            self._decompressed_cache[cache_key] = (k, v)
            while len(self._decompressed_cache) > self.cfg.max_cached_decompressed_blocks:
                self._decompressed_cache.popitem(last=False)
        return k, v

    def _compressed_payload_for_block(
        self, blk: KVBlock
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            self._resolve_turboquant_proxy(blk.compressed_k),
            self._resolve_turboquant_proxy(blk.compressed_v),
        )

    def _resolve_turboquant_proxy(self, compressed: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(compressed, dict) or "__run_id" not in compressed:
            return compressed

        run_id = int(compressed["__run_id"])
        start = int(compressed.get("__run_start", 0))
        B, H, length, D = compressed["shape"]
        run = self._tq_compressed_runs[run_id]
        if not run.get("active", True):
            raise RuntimeError(f"compressed run {run_id} is inactive")
        run_key = "compressed_k" if compressed.get("ttype") == "k" else "compressed_v"
        return self._slice_turboquant_compressed(run[run_key], start, int(length))

    def _apply_residual_payloads(
        self,
        blk: KVBlock,
        k: torch.Tensor,
        v: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.page_backend, TurboQuantPageBackend):
            return k, v
        if isinstance(blk.compressed_k, dict) and "__residual" in blk.compressed_k:
            bits = float(blk.compressed_k["__residual_bits"])
            residual = self.page_backend.get_compressor("k", bits).decompress(
                blk.compressed_k["__residual"]
            )
            k = k + residual.to(dtype)
        if isinstance(blk.compressed_v, dict) and "__residual" in blk.compressed_v:
            bits = float(blk.compressed_v["__residual_bits"])
            residual = self.page_backend.get_compressor("v", bits).decompress(
                blk.compressed_v["__residual"]
            )
            v = v + residual.to(dtype)
        return k, v

    def _apply_batched_residual_payloads(
        self,
        blocks: list[KVBlock],
        k: torch.Tensor,
        v: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.page_backend, TurboQuantPageBackend):
            return k, v

        positions: list[tuple[KVBlock, int, int]] = []
        start = 0
        for blk in blocks:
            end = start + blk.current_len
            positions.append((blk, start, end))
            start = end

        k_entries = [
            (blk, start, end)
            for blk, start, end in positions
            if isinstance(blk.compressed_k, dict) and "__residual" in blk.compressed_k
        ]
        if k_entries:
            bits = float(k_entries[0][0].compressed_k["__residual_bits"])
            merged = self._merge_turboquant_compressed(
                [blk.compressed_k["__residual"] for blk, _start, _end in k_entries]
            )
            residual = (
                self.page_backend.get_compressor("k", bits)
                .decompress(merged)
                .to(dtype)
            )
            k = k.clone()
            offset = 0
            for blk, start, end in k_entries:
                next_offset = offset + blk.current_len
                k[:, :, start:end, :] += residual[:, :, offset:next_offset, :]
                offset = next_offset

        v_entries = [
            (blk, start, end)
            for blk, start, end in positions
            if isinstance(blk.compressed_v, dict) and "__residual" in blk.compressed_v
        ]
        if v_entries:
            bits = float(v_entries[0][0].compressed_v["__residual_bits"])
            merged = self._merge_turboquant_compressed(
                [blk.compressed_v["__residual"] for blk, _start, _end in v_entries]
            )
            residual = (
                self.page_backend.get_compressor("v", bits)
                .decompress(merged)
                .to(dtype)
            )
            v = v.clone()
            offset = 0
            for blk, start, end in v_entries:
                next_offset = offset + blk.current_len
                v[:, :, start:end, :] += residual[:, :, offset:next_offset, :]
                offset = next_offset

        return k, v

    @staticmethod
    def _slice_turboquant_compressed(
        compressed: dict[str, Any], start: int, length: int
    ) -> dict[str, Any]:
        B, H, _S, D = compressed["shape"]
        end = start + length
        part: dict[str, Any] = {}
        for key, value in compressed.items():
            if key == "shape":
                part[key] = (B, H, length, D)
            elif torch.is_tensor(value) and value.ndim >= 3 and value.shape[2] == _S:
                part[key] = value[:, :, start:end, ...]
            else:
                part[key] = value
        return part

    def _can_batch_turboquant_materialize(self) -> bool:
        return (
            isinstance(self.page_backend, TurboQuantPageBackend)
            and self.cfg.granularity == "per-vector"
            and self.cfg.max_cached_decompressed_blocks <= 0
        )

    def _is_turboquant_compressed_block(self, blk: KVBlock) -> bool:
        return (
            blk.state == BlockState.COMPRESSED
            and isinstance(blk.compressed_k, dict)
            and isinstance(blk.compressed_v, dict)
            and blk.compressed_k.get("backend") == "turboquant"
            and blk.compressed_v.get("backend") == "turboquant"
        )

    @staticmethod
    def _merge_turboquant_compressed(
        compressed_parts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not compressed_parts:
            raise ValueError("cannot merge an empty compressed page list")

        B, H, _S0, D = compressed_parts[0]["shape"]
        total_s = sum(int(part["shape"][2]) for part in compressed_parts)
        merged: dict[str, Any] = {}
        for key, first in compressed_parts[0].items():
            if key == "shape":
                merged[key] = (B, H, total_s, D)
                continue
            if (
                torch.is_tensor(first)
                and first.ndim >= 3
                and first.shape[2] == _S0
            ):
                merged[key] = torch.cat([part[key] for part in compressed_parts], dim=2)
            else:
                merged[key] = first
        return merged

    def _decompress_turboquant_group(
        self,
        blocks: list[KVBlock],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.page_backend is None:
            raise RuntimeError("page backend was not initialized")
        if len(blocks) == 1:
            return self._decompress_block(blocks[0], dtype)

        first = blocks[0]
        k_bits = first.key_bits if first.key_bits is not None else self.cfg.key_bits
        v_bits = first.value_bits if first.value_bits is not None else self.cfg.value_bits
        run = self._turboquant_run_for_group(blocks)
        if run is not None:
            merged_k = run["compressed_k"]
            merged_v = run["compressed_v"]
        else:
            payloads = [self._compressed_payload_for_block(blk) for blk in blocks]
            merged_k = self._merge_turboquant_compressed([p[0] for p in payloads])
            merged_v = self._merge_turboquant_compressed([p[1] for p in payloads])
        k, v = self.page_backend.decompress(
            merged_k,
            merged_v,
            key_bits=k_bits,
            value_bits=v_bits,
            dtype=dtype,
        )
        if any(
            (isinstance(blk.compressed_k, dict) and "__residual" in blk.compressed_k)
            or (isinstance(blk.compressed_v, dict) and "__residual" in blk.compressed_v)
            for blk in blocks
        ):
            return self._apply_batched_residual_payloads(blocks, k, v, dtype)
        return k, v

    def _turboquant_run_for_group(
        self, blocks: list[KVBlock]
    ) -> Optional[dict[str, Any]]:
        if not blocks:
            return None
        first_meta = blocks[0].compressed_k
        if not isinstance(first_meta, dict) or "__run_id" not in first_meta:
            return None
        run_id = int(first_meta["__run_id"])
        expected_start = int(first_meta.get("__run_start", 0))
        total_len = 0
        for blk in blocks:
            meta = blk.compressed_k
            if not isinstance(meta, dict) or int(meta.get("__run_id", -1)) != run_id:
                return None
            if int(meta.get("__run_start", -1)) != expected_start + total_len:
                return None
            total_len += blk.current_len
        run = self._tq_compressed_runs[run_id]
        if not run.get("active", True):
            return None
        if expected_start != 0 or total_len != int(run["compressed_k"]["shape"][2]):
            return None
        return run

    def _materialize_blocks(
        self, blocks: list[KVBlock], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not blocks:
            raise RuntimeError("cannot materialize an empty block list")
        if not self._can_batch_turboquant_materialize():
            return self._materialize_blocks_legacy(blocks, dtype)

        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        group: list[KVBlock] = []
        group_key: Optional[tuple[float, float, Optional[int]]] = None

        def flush_group() -> None:
            nonlocal group, group_key
            if not group:
                return
            k, v = self._decompress_turboquant_group(group, dtype)
            ks.append(k)
            vs.append(v)
            group = []
            group_key = None

        for blk in blocks:
            if self._is_turboquant_compressed_block(blk):
                key = (
                    float(blk.key_bits if blk.key_bits is not None else self.cfg.key_bits),
                    float(blk.value_bits if blk.value_bits is not None else self.cfg.value_bits),
                    (
                        int(blk.compressed_k["__run_id"])
                        if isinstance(blk.compressed_k, dict)
                        and "__run_id" in blk.compressed_k
                        else None
                    ),
                )
                if group and key != group_key:
                    flush_group()
                group.append(blk)
                group_key = key
                continue

            flush_group()
            k, v = self._materialize_block(blk, dtype)
            ks.append(k)
            vs.append(v)

        flush_group()
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def _materialize_blocks_legacy(
        self, blocks: list[KVBlock], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for blk in blocks:
            k, v = self._materialize_block(blk, dtype)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def _materialize(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.incremental_materialize:
            return self._materialize_legacy(dtype)

        if self.table is None:
            raise RuntimeError("cache layer was not initialized")

        device = self.device
        if (
            self._mat_k is not None
            and self._mat_v is not None
            and self._mat_dtype == dtype
            and self._mat_device == device
        ):
            new_sig = [self._block_signature(blk) for blk in self.table.blocks]
            if new_sig == self._mat_sig:
                return self._mat_k, self._mat_v
        else:
            new_sig = [self._block_signature(blk) for blk in self.table.blocks]
            self._invalidate_materialized_cache()

        if not self.table.blocks:
            raise RuntimeError("cannot materialize an empty block table")

        p = 0
        if self._mat_k is not None and self._mat_v is not None:
            common = min(len(new_sig), len(self._mat_sig))
            while p < common and new_sig[p] == self._mat_sig[p]:
                p += 1

        prefix_tokens = sum(blk.current_len for blk in self.table.blocks[:p])
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        if prefix_tokens and self._mat_k is not None and self._mat_v is not None:
            parts_k.append(self._mat_k[:, :, :prefix_tokens, :])
            parts_v.append(self._mat_v[:, :, :prefix_tokens, :])

        suffix = self.table.blocks[p:]
        if suffix:
            k, v = self._materialize_blocks(suffix, dtype)
            parts_k.append(k)
            parts_v.append(v)

        self._mat_k = torch.cat(parts_k, dim=2)
        self._mat_v = torch.cat(parts_v, dim=2)
        self._mat_sig = new_sig
        self._mat_dtype = dtype
        self._mat_device = device
        return self._mat_k, self._mat_v

    def _materialize_block(
        self, blk: KVBlock, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if blk.state == BlockState.COMPRESSED:
            return self._decompress_block(blk, dtype)
        return blk.fp16_k.to(dtype), blk.fp16_v.to(dtype)

    def _materialize_legacy(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._materialize_blocks(self.table.blocks, dtype)

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

    def _compressed_payload_dicts(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []

        def add_payload_dicts(value: Any) -> None:
            if not isinstance(value, dict):
                return
            payloads.append(value)
            for child in value.values():
                if isinstance(child, dict):
                    add_payload_dicts(child)

        for run in self._tq_compressed_runs:
            if run.get("active", True):
                add_payload_dicts(run["compressed_k"])
                add_payload_dicts(run["compressed_v"])
        if self.table is not None:
            for blk in self.table.blocks:
                for d in (blk.compressed_k, blk.compressed_v):
                    if isinstance(d, dict):
                        if "__run_id" not in d:
                            add_payload_dicts(d)
                        else:
                            for child in d.values():
                                if isinstance(child, dict):
                                    add_payload_dicts(child)
        return payloads

    def reset(self) -> None:
        if self.table is not None:
            self.table.blocks.clear()
            self.table._total_len = 0
        self._invalidate_decompressed_cache()
        self._invalidate_materialized_cache()
        self._tq_compressed_runs.clear()
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
        self._invalidate_materialized_cache()
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k.index_select(0, beam_idx.to(blk.fp16_k.device))
                blk.fp16_v = blk.fp16_v.index_select(0, beam_idx.to(blk.fp16_v.device))
        for d in self._compressed_payload_dicts():
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
        self._invalidate_materialized_cache()
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
        self.table._total_len = sum(blk.current_len for blk in kept)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.table is None:
            return
        self._invalidate_decompressed_cache()
        self._invalidate_materialized_cache()
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k.repeat_interleave(repeats, dim=0)
                blk.fp16_v = blk.fp16_v.repeat_interleave(repeats, dim=0)
        for d in self._compressed_payload_dicts():
            for k_, val in list(d.items()):
                if torch.is_tensor(val) and val.shape[:1] == (
                    self.table.batch_size,
                ):
                    d[k_] = val.repeat_interleave(repeats, dim=0)
            if "shape" in d:
                B, H, S, D = d["shape"]
                d["shape"] = (int(B) * repeats, H, S, D)
        self.table.batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.table is None:
            return
        self._invalidate_decompressed_cache()
        self._invalidate_materialized_cache()
        self._pending_quant_blocks.clear()
        self._pending_quant_block_ids.clear()
        for blk in self.table.blocks:
            if blk.fp16_k is not None:
                blk.fp16_k = blk.fp16_k[indices]
                blk.fp16_v = blk.fp16_v[indices]
        for d in self._compressed_payload_dicts():
            for k_, val in list(d.items()):
                if torch.is_tensor(val) and val.shape[:1] == (
                    self.table.batch_size,
                ):
                    d[k_] = val[indices]
            if "shape" in d:
                _B, H, S, D = d["shape"]
                d["shape"] = (int(indices.shape[0]), H, S, D)
        self.table.batch_size = int(indices.shape[0])

    def memory_bytes(self) -> int:
        if self.table is None:
            return 0
        return self.table.memory_bytes() + self.compressed_run_memory_bytes()

    def compressed_run_memory_bytes(self) -> int:
        n = 0
        for run in self._tq_compressed_runs:
            if not run.get("active", True):
                continue
            for d in (run["compressed_k"], run["compressed_v"]):
                for value in d.values():
                    if torch.is_tensor(value):
                        n += value.numel() * value.element_size()
        return n

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
                    "compressed_run_id": blk.compressed_run_id,
                    "compressed_run_start": blk.compressed_run_start,
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
            "tq_compressed_runs": self._tq_compressed_runs,
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
            self._invalidate_decompressed_cache()
            self._invalidate_materialized_cache()
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
        for run in state.get("tq_compressed_runs", []):
            for d in (run.get("compressed_k"), run.get("compressed_v")):
                if not isinstance(d, dict):
                    continue
                tensor = next((v for v in d.values() if torch.is_tensor(v)), None)
                if tensor is not None:
                    first_device = tensor.device
                    break

        self._invalidate_decompressed_cache()
        self._invalidate_materialized_cache()
        self._init_runtime(
            batch_size=int(table_state["batch_size"]),
            n_kv_heads=int(table_state["n_kv_heads"]),
            head_dim=int(table_state["head_dim"]),
            dtype=dtype,
            device=first_device,
        )
        self.table.blocks = []
        self._tq_compressed_runs = list(state.get("tq_compressed_runs", []))

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
                compressed_run_id=block_state.get("compressed_run_id"),
                compressed_run_start=int(block_state.get("compressed_run_start", 0)),
                importance=float(block_state.get("importance", 0.0)),
                key_bits=block_state.get("key_bits"),
                value_bits=block_state.get("value_bits"),
                page_meta=block_state.get("page_meta"),
            )
            self.table.blocks.append(blk)
        self.table._total_len = sum(blk.current_len for blk in self.table.blocks)

        pending = []
        for block_idx in state.get("pending_quant_blocks", []):
            idx = int(block_idx)
            if 0 <= idx < len(self.table.blocks):
                blk = self.table.blocks[idx]
                if blk.state == BlockState.SEALED:
                    pending.append(idx)
        self._pending_quant_blocks = deque(pending)
        self._pending_quant_block_ids = set(pending)
