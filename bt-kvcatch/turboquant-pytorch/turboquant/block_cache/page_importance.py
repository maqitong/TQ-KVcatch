"""Page-level importance scoring for block-structured KV cache.

The first implementation deliberately uses block-local statistics so it can
run inside the current HuggingFace Cache path without patching attention.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import BlockTable, KVBlock


class PageImportanceScorer(ABC):
    """Assign a scalar importance score to a sealed KV page."""

    name = "base"

    @abstractmethod
    def score(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> float:
        ...


class NormPageImportanceScorer(PageImportanceScorer):
    """Use block-level K/V L2 norms as an attention-free importance proxy."""

    def __init__(self, mode: str = "k_norm"):
        if mode not in ("k_norm", "v_norm", "kv_norm"):
            raise ValueError(f"unknown norm importance mode: {mode}")
        self.mode = mode
        self.name = mode

    def score(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> float:
        if block.fp16_k is None or block.fp16_v is None:
            return float(block.importance)

        scores: list[float] = []
        if self.mode in ("k_norm", "kv_norm"):
            scores.append(float(block.fp16_k.float().norm(dim=-1).mean().item()))
        if self.mode in ("v_norm", "kv_norm"):
            scores.append(float(block.fp16_v.float().norm(dim=-1).mean().item()))
        return sum(scores) / max(len(scores), 1)


class RandomPageImportanceScorer(PageImportanceScorer):
    """Random baseline for ablations."""

    name = "random"

    def score(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> float:
        return random.random()


def build_page_importance_scorer(name: str) -> PageImportanceScorer:
    normalized = name.replace("-", "_")
    if normalized in ("k_norm", "v_norm", "kv_norm"):
        return NormPageImportanceScorer(normalized)
    if normalized == "random":
        return RandomPageImportanceScorer()
    raise ValueError(f"unknown page importance metric: {name}")
