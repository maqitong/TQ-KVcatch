"""Configuration for block-structured KV-cache compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .policies import GroupingPolicy, TokenBlockPolicy


@dataclass
class BlockCacheConfig:
    """Configuration shared across all layers of a ``BlockKVCache``."""

    block_size: int = 16
    key_bits: int = 6
    value_bits: int = 4
    granularity: str = "per-vector"  # 'per-vector' | 'per-block'
    seed: int = 42
    policy: GroupingPolicy = field(default_factory=TokenBlockPolicy)
    quant_backend: str = "turboquant"  # 'turboquant' | 'skvq' | registered backend
    mixed_precision: bool = False
    mixed_precision_mode: str = "direct"  # 'direct' | 'base_residual'
    importance_metric: str = "k_norm"
    important_ratio: float = 0.2
    pagemix_run_aware: bool = True
    pagemix_max_high_runs: int = 1
    high_key_bits: float = 4
    high_value_bits: float = 2
    low_key_bits: float = 2
    low_value_bits: float = 2
    residual_key_bits: float = 2
    residual_value_bits: float = 0
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
    incremental_materialize: bool = False
    # None keeps the original synchronous behavior. An integer enables a
    # budgeted "quant cursor": ready pages are queued and at most this many
    # pages are compressed per cache update / attention feedback call.
    quant_budget_per_update: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mixed_precision_mode not in ("direct", "base_residual"):
            raise ValueError("mixed_precision_mode must be 'direct' or 'base_residual'")
        if self.quant_budget_per_update is not None and self.quant_budget_per_update < 0:
            raise ValueError("quant_budget_per_update must be non-negative or None")
        if self.residual_key_bits < 0 or self.residual_value_bits < 0:
            raise ValueError("residual bit-widths must be non-negative")
        if self.pagemix_max_high_runs < 1:
            raise ValueError("pagemix_max_high_runs must be >= 1")
