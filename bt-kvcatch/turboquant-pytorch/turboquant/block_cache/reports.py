"""Diagnostics for block-structured KV caches."""

from __future__ import annotations

from typing import Any, Iterable

from .blocks import BlockState


def build_memory_report(layers: Iterable[Any]) -> dict[str, Any]:
    """Aggregate storage diagnostics across cache layers."""
    layers = list(layers)
    compressed_bytes = 0
    fp16_baseline = 0
    n_compressed_blocks = 0
    n_fp16_blocks = 0
    n_pending_quant_blocks = 0
    bit_histogram: dict[str, int] = {}
    precision_histogram: dict[str, int] = {}
    quant_status_histogram: dict[str, int] = {}

    for layer in layers:
        if layer.table is None:
            continue
        n_pending_quant_blocks += len(getattr(layer, "_pending_quant_blocks", []))
        for blk in layer.table.blocks:
            quant_status = (
                blk.page_meta.get("quant_status")
                if isinstance(blk.page_meta, dict)
                else None
            )
            if quant_status is not None:
                quant_status_histogram[quant_status] = (
                    quant_status_histogram.get(quant_status, 0) + 1
                )
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
        compressed_bytes += getattr(layer, "compressed_run_memory_bytes", lambda: 0)()

    return {
        "compressed_bytes": compressed_bytes,
        "fp16_baseline_bytes": fp16_baseline,
        "compression_ratio": (
            fp16_baseline / compressed_bytes if compressed_bytes > 0 else 0.0
        ),
        "n_compressed_blocks": n_compressed_blocks,
        "n_fp16_blocks": n_fp16_blocks,
        "n_pending_quant_blocks": n_pending_quant_blocks,
        "n_layers": len(layers),
        "bit_histogram": bit_histogram,
        "precision_histogram": precision_histogram,
        "quant_status_histogram": quant_status_histogram,
    }
