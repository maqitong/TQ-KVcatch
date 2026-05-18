"""Bit allocation policies for page-level mixed precision."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .page_importance import PageImportanceScorer

if TYPE_CHECKING:
    from .blocks import BlockTable, KVBlock


BitPair = tuple[float, float]


class PageBitAllocator(ABC):
    """Choose K/V bit-widths for pages leaving the fp16 working set."""

    @abstractmethod
    def assign(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> BitPair:
        ...

    def assign_many(
        self, blocks: list["KVBlock"], table: "BlockTable", layer_idx: int
    ) -> dict[int, BitPair]:
        return {block.block_idx: self.assign(block, table, layer_idx) for block in blocks}


class FixedPageBitAllocator(PageBitAllocator):
    """Use one K/V bit-width pair for every compressed page."""

    def __init__(self, key_bits: float, value_bits: float):
        self.key_bits = float(key_bits)
        self.value_bits = float(value_bits)

    def assign(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> BitPair:
        block.key_bits = self.key_bits
        block.value_bits = self.value_bits
        block.page_meta = {
            "allocator": "fixed",
            "importance": block.importance,
        }
        return self.key_bits, self.value_bits


class TopRatioPageBitAllocator(PageBitAllocator):
    """Give the top-scoring fraction of pages a higher precision budget."""

    def __init__(
        self,
        scorer: PageImportanceScorer,
        important_ratio: float,
        high_key_bits: float,
        high_value_bits: float,
        low_key_bits: float,
        low_value_bits: float,
    ):
        if not 0.0 <= important_ratio <= 1.0:
            raise ValueError("important_ratio must be in [0, 1]")
        self.scorer = scorer
        self.important_ratio = important_ratio
        self.high_bits = (float(high_key_bits), float(high_value_bits))
        self.low_bits = (float(low_key_bits), float(low_value_bits))

    def assign(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> BitPair:
        block.importance = self.scorer.score(block, table, layer_idx)
        block.key_bits, block.value_bits = self.low_bits
        block.page_meta = {
            "allocator": "top_ratio",
            "importance_metric": self.scorer.name,
            "importance": block.importance,
            "precision": "low",
        }
        return self.low_bits

    def assign_many(
        self, blocks: list["KVBlock"], table: "BlockTable", layer_idx: int
    ) -> dict[int, BitPair]:
        if not blocks:
            return {}

        scored = []
        for block in blocks:
            block.importance = self.scorer.score(block, table, layer_idx)
            scored.append((block.importance, block))

        n_high = int(math.ceil(len(scored) * self.important_ratio))
        high_ids = {
            block.block_idx
            for _score, block in sorted(scored, key=lambda item: item[0], reverse=True)[
                :n_high
            ]
        }
        threshold = min((score for score, block in scored if block.block_idx in high_ids), default=None)

        out: dict[int, BitPair] = {}
        for rank, (score, block) in enumerate(
            sorted(scored, key=lambda item: item[0], reverse=True)
        ):
            is_high = block.block_idx in high_ids
            bits = self.high_bits if is_high else self.low_bits
            block.key_bits, block.value_bits = bits
            block.page_meta = {
                "allocator": "top_ratio",
                "importance_metric": self.scorer.name,
                "importance": score,
                "rank": rank,
                "threshold": threshold,
                "precision": "high" if is_high else "low",
            }
            out[block.block_idx] = bits
        return out
