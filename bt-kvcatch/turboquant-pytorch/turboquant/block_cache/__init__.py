"""Block-structured KV cache with TurboQuant per-block quantization.

See `README_block_cache.md` for the design overview and paper mapping.
"""

from .blocks import BlockState, KVBlock, BlockTable
from .policies import (
    GroupingPolicy,
    TokenBlockPolicy,
    WindowBlockPolicy,
    HybridPolicy,
)
from .quantizer import BlockMSECompressor
from .page_importance import (
    PageImportanceScorer,
    NormPageImportanceScorer,
    RandomPageImportanceScorer,
)
from .bit_allocator import (
    PageBitAllocator,
    FixedPageBitAllocator,
    TopRatioPageBitAllocator,
)
from .skvq_quantizer import SKVQPageCompressor
from .hf_cache import BlockKVCache, BlockCacheConfig

__all__ = [
    "BlockState",
    "KVBlock",
    "BlockTable",
    "GroupingPolicy",
    "TokenBlockPolicy",
    "WindowBlockPolicy",
    "HybridPolicy",
    "BlockMSECompressor",
    "PageImportanceScorer",
    "NormPageImportanceScorer",
    "RandomPageImportanceScorer",
    "PageBitAllocator",
    "FixedPageBitAllocator",
    "TopRatioPageBitAllocator",
    "SKVQPageCompressor",
    "BlockKVCache",
    "BlockCacheConfig",
]
