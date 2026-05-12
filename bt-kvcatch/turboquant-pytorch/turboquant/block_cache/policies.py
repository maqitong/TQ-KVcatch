"""GroupingPolicy: pluggable strategy deciding when blocks get compressed.

Three concrete policies cover the report's grouping axes:

  * TokenBlockPolicy   - fixed-position grouping (PagedAttention-style).
                         Every sealed block is compressed immediately.
  * WindowBlockPolicy  - temporal grouping (SKVQ-style sliding window).
                         Recent `window_size` tokens stay FP16; blocks fully
                         outside the window get compressed.
  * HybridPolicy       - sink + window + token blocks combined.

Add new policies by subclassing `GroupingPolicy` and overriding `on_seal`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import BlockTable, KVBlock


class GroupingPolicy(ABC):
    """Strategy for SEALED -> COMPRESSED transitions."""

    @abstractmethod
    def on_seal(
        self, sealed: list["KVBlock"], table: "BlockTable"
    ) -> list["KVBlock"]:
        """Return the subset of blocks that should be compressed now.

        Called every time `BlockTable.append` produces newly-sealed blocks.
        Implementations are free to also re-evaluate previously-sealed blocks
        (e.g. window policy re-checks whether old blocks have rolled out).
        """
        ...


class TokenBlockPolicy(GroupingPolicy):
    """Compress every block as soon as it seals.

    Pure fixed-position grouping. Highest compression ratio; recent tokens
    suffer the most quantization error.
    """

    def on_seal(self, sealed, table):
        return list(sealed)


class WindowBlockPolicy(GroupingPolicy):
    """Keep the most-recent `window_size` tokens in FP16; compress the rest.

    A block is compressed only when *all* of its tokens have rolled out of
    the recent window. Optionally the first `sink_size` tokens are also
    permanently kept in FP16 (the "attention sink" pattern).

    For clean semantics prefer `window_size % block_size == 0` so that a
    block boundary aligns with the window edge, but the policy works
    correctly for arbitrary alignments — a partially-in-window block stays
    FP16 until it has fully rolled out.
    """

    def __init__(self, window_size: int, sink_size: int = 0):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if sink_size < 0:
            raise ValueError("sink_size must be non-negative")
        self.window_size = window_size
        self.sink_size = sink_size

    def on_seal(self, sealed, table):
        from .blocks import BlockState

        to_compress: list = []
        total = table.total_len
        cursor = 0
        for blk in table.blocks:
            blk_start = cursor
            blk_end = cursor + blk.current_len
            cursor = blk_end
            if blk.state != BlockState.SEALED:
                continue
            in_sink = blk_end <= self.sink_size
            in_window = blk_end > total - self.window_size
            if not in_sink and not in_window:
                to_compress.append(blk)
        return to_compress


class HybridPolicy(GroupingPolicy):
    """Sink + window + token-block compression.

    Equivalent to `WindowBlockPolicy` with a non-zero `sink_size`. Provided
    as a separate name so that demo scripts can talk about the three-segment
    layout (sink / window / quantized blocks) explicitly.
    """

    def __init__(self, sink_size: int, window_size: int):
        self._inner = WindowBlockPolicy(
            window_size=window_size, sink_size=sink_size
        )

    @property
    def window_size(self) -> int:
        return self._inner.window_size

    @property
    def sink_size(self) -> int:
        return self._inner.sink_size

    def on_seal(self, sealed, table):
        return self._inner.on_seal(sealed, table)
