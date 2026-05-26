"""HuggingFace ``Cache`` entry point for block-structured KV storage.

``BlockKVCache`` is intentionally a thin orchestrator. Per-layer page storage
and compression live in ``layer.py``; page quantization algorithms live in
``backends/``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch

from .config import BlockCacheConfig
from .hf_compat import HF_AVAILABLE, HFCache
from .layer import BlockCacheLayer
from .reports import build_memory_report


class BlockKVCache(HFCache):
    """Block-structured KV cache compatible with HuggingFace ``generate()``.

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
        if HF_AVAILABLE:
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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        """Record attention probabilities for later page-importance scoring."""
        if layer_idx >= len(self.layers):
            return
        self.layers[layer_idx].record_attention(attn_weights)

    def record_attentions(self, attentions: Any) -> None:
        """Record a HuggingFace attentions object.

        Handles both ordinary forward outputs shaped like
        ``tuple[layer](B, H, Q, S)`` and generation outputs shaped like
        ``tuple[step][layer](B, H, Q, S)``.
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

    def memory_report(self) -> dict[str, Any]:
        return build_memory_report(self.layers)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the cache payload.

        Restore into a ``BlockKVCache`` constructed with a compatible
        ``BlockCacheConfig``. The config snapshot is included for diagnostics,
        but ``load_state_dict()`` intentionally does not replace ``self.config``.
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
            raise ValueError(
                f"unsupported BlockKVCache state format: {state.get('format')}"
            )

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


__all__ = ["BlockKVCache", "BlockCacheConfig", "BlockCacheLayer"]
