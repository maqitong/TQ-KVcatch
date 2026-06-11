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
        meta = dict(block.page_meta) if isinstance(block.page_meta, dict) else {}
        meta.update({
            "allocator": "fixed",
            "importance": block.importance,
        })
        block.page_meta = meta
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
        run_aware: bool = True,
        max_high_runs: int = 1,
    ):
        if not 0.0 <= important_ratio <= 1.0:
            raise ValueError("important_ratio must be in [0, 1]")
        if max_high_runs < 1:
            raise ValueError("max_high_runs must be >= 1")
        self.scorer = scorer
        self.important_ratio = important_ratio
        self.high_bits = (float(high_key_bits), float(high_value_bits))
        self.low_bits = (float(low_key_bits), float(low_value_bits))
        self.run_aware = bool(run_aware)
        self.max_high_runs = int(max_high_runs)

    def assign(self, block: "KVBlock", table: "BlockTable", layer_idx: int) -> BitPair:
        block.importance = self.scorer.score(block, table, layer_idx)
        block.key_bits, block.value_bits = self.low_bits
        meta = dict(block.page_meta) if isinstance(block.page_meta, dict) else {}
        meta.update({
            "allocator": "top_ratio",
            "importance_metric": self.scorer.name,
            "importance": block.importance,
            "precision": "low",
        })
        block.page_meta = meta
        return self.low_bits

    def assign_many(
        self, blocks: list["KVBlock"], table: "BlockTable", layer_idx: int
    ) -> dict[int, BitPair]:
        if not blocks:
            return {}

        scores = self.scorer.score_many(blocks, table, layer_idx)
        scored = []
        for score, block in zip(scores, blocks):
            block.importance = float(score)
            scored.append((block.importance, block))

        n_high = int(math.ceil(len(scored) * self.important_ratio))
        high_ids = self._select_high_block_ids(scored, n_high)
        threshold = min((score for score, block in scored if block.block_idx in high_ids), default=None)

        out: dict[int, BitPair] = {}
        for rank, (score, block) in enumerate(
            sorted(scored, key=lambda item: item[0], reverse=True)
        ):
            is_high = block.block_idx in high_ids
            bits = self.high_bits if is_high else self.low_bits
            block.key_bits, block.value_bits = bits
            meta = dict(block.page_meta) if isinstance(block.page_meta, dict) else {}
            meta.update({
                "allocator": "top_ratio",
                "importance_metric": self.scorer.name,
                "importance": score,
                "rank": rank,
                "threshold": threshold,
                "precision": "high" if is_high else "low",
                "max_high_runs": self.max_high_runs if self.run_aware else None,
            })
            block.page_meta = meta
            out[block.block_idx] = bits
        return out

    def _select_high_block_ids(
        self, scored: list[tuple[float, "KVBlock"]], n_high: int
    ) -> set[int]:
        if n_high <= 0:
            return set()
        if n_high >= len(scored):
            return {block.block_idx for _score, block in scored}
        if not self.run_aware:
            return {
                block.block_idx
                for _score, block in sorted(scored, key=lambda item: item[0], reverse=True)[
                    :n_high
                ]
            }

        ordered = sorted(scored, key=lambda item: item[1].block_idx)
        if self.max_high_runs > 1:
            return self._select_segmented_high_block_ids(ordered, n_high)

        window_sum = sum(score for score, _block in ordered[:n_high])
        best_sum = window_sum
        best_start = 0
        for start in range(1, len(ordered) - n_high + 1):
            window_sum += ordered[start + n_high - 1][0]
            window_sum -= ordered[start - 1][0]
            if window_sum > best_sum:
                best_sum = window_sum
                best_start = start
        return {
            block.block_idx
            for _score, block in ordered[best_start : best_start + n_high]
        }

    def _select_segmented_high_block_ids(
        self, ordered: list[tuple[float, "KVBlock"]], n_high: int
    ) -> set[int]:
        states: dict[
            tuple[int, int, bool], tuple[float, tuple[int, ...]]
        ] = {(0, 0, False): (0.0, ())}
        max_runs = min(self.max_high_runs, n_high)

        for score, block in ordered:
            next_states: dict[
                tuple[int, int, bool], tuple[float, tuple[int, ...]]
            ] = {}
            for (count, runs, in_run), (total, ids) in states.items():
                skip_key = (count, runs, False)
                current = next_states.get(skip_key)
                if current is None or total > current[0]:
                    next_states[skip_key] = (total, ids)

                if count >= n_high:
                    continue
                next_runs = runs if in_run else runs + 1
                if next_runs > max_runs:
                    continue
                key = (count + 1, next_runs, True)
                value = (total + score, ids + (block.block_idx,))
                current = next_states.get(key)
                if current is None or value[0] > current[0]:
                    next_states[key] = value
            states = next_states

        candidates = [
            value for (count, _runs, _in_run), value in states.items() if count == n_high
        ]
        if not candidates:
            return set()
        _score, ids = max(candidates, key=lambda item: item[0])
        return set(ids)
