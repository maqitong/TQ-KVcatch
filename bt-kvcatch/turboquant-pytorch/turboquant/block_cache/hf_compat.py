"""Small compatibility layer for HuggingFace cache classes."""

from __future__ import annotations

try:
    from transformers.cache_utils import Cache as HFCache  # type: ignore
    from transformers.cache_utils import CacheLayerMixin as HFCacheLayerMixin  # type: ignore

    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HFCache = object
    HFCacheLayerMixin = object
    HF_AVAILABLE = False


__all__ = ["HFCache", "HFCacheLayerMixin", "HF_AVAILABLE"]
