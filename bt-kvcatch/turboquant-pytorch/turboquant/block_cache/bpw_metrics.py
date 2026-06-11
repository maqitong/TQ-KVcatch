"""BPW helpers for KV-cache experiment tables (see EXPERIMENTS_4090_README.md)."""
from __future__ import annotations

import re
from typing import Any

FP16_BITS = 16.0
KV_PAIR_FP16_BITS = 32.0  # K16 + V16

_BIT_HIST_RE = re.compile(r"K([^/]+)/V(.+)")


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def theoretical_bpw_from_config(
    backend: str,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    """Theoretical page bpw on compressed pages (from scheme config)."""
    if backend in ("dynamic", "") or config.get("policy") in ("fp16", "v3_flat"):
        if backend == "v3_flat" or config.get("policy") == "v3_flat":
            k = _as_float(config.get("key_bits"), 2)
            v = _as_float(config.get("value_bits"), 2)
            return k, v, (k + v) / 2.0
        return FP16_BITS, FP16_BITS, FP16_BITS

    if config.get("mixed"):
        ratio = _as_float(config.get("important_ratio"), 0.3)
        k = ratio * _as_float(config.get("high_key_bits"), 4) + (1 - ratio) * _as_float(
            config.get("low_key_bits"), 2
        )
        v = ratio * _as_float(config.get("high_value_bits"), 4) + (1 - ratio) * _as_float(
            config.get("low_value_bits"), 2
        )
        return k, v, (k + v) / 2.0

    k = _as_float(config.get("key_bits"), FP16_BITS)
    v = _as_float(config.get("value_bits"), FP16_BITS)
    return k, v, (k + v) / 2.0


def theoretical_bpw_from_histogram(
    bit_histogram: dict[str, int],
    residual_histogram: dict[str, int] | None = None,
) -> tuple[float, float, float]:
    """Weighted K/V/Avg bpw from memory_report bit_histogram."""
    total = 0
    k_acc = 0.0
    v_acc = 0.0
    for key, count in bit_histogram.items():
        m = _BIT_HIST_RE.fullmatch(str(key))
        if not m or count <= 0:
            continue
        k_bits = float(m.group(1))
        v_bits = float(m.group(2))
        total += count
        k_acc += k_bits * count
        v_acc += v_bits * count
    if total <= 0:
        return 0.0, 0.0, 0.0
    for key, count in (residual_histogram or {}).items():
        m = _BIT_HIST_RE.fullmatch(str(key))
        if not m or count <= 0:
            continue
        k_acc += float(m.group(1)) * count
        v_acc += float(m.group(2)) * count
    k_bpw = k_acc / total
    v_bpw = v_acc / total
    return k_bpw, v_bpw, (k_bpw + v_bpw) / 2.0


def effective_bpw_kv_pair(compression_ratio: float | None) -> float | None:
    """Whole-cache effective bpw: 32 / compression_ratio (K16+V16 baseline)."""
    if compression_ratio is None or compression_ratio <= 0:
        return None
    return KV_PAIR_FP16_BITS / float(compression_ratio)


def attach_bpw_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Add k_bpw, v_bpw, avg_bpw, effective_bpw to a PPL / result row."""
    cfg = row.get("config") or {}
    backend = row.get("backend", "")

    if backend == "dynamic":
        k_bpw = v_bpw = avg_bpw = FP16_BITS
        effective = FP16_BITS
    elif row.get("bit_histogram"):
        k_bpw, v_bpw, avg_bpw = theoretical_bpw_from_histogram(
            row["bit_histogram"], row.get("residual_histogram")
        )
        effective = effective_bpw_kv_pair(row.get("avg_compression_ratio"))
    else:
        k_bpw, v_bpw, avg_bpw = theoretical_bpw_from_config(backend, cfg)
        effective = effective_bpw_kv_pair(row.get("avg_compression_ratio"))
        if effective is None and backend == "skvq_native":
            # Native SKVQ path: no BlockKVCache ratio in export; use scheme nominal only.
            effective = None

    row["k_bpw"] = round(k_bpw, 4)
    row["v_bpw"] = round(v_bpw, 4)
    row["avg_bpw"] = round(avg_bpw, 4)
    row["effective_bpw"] = round(effective, 4) if effective is not None else None
    return row
