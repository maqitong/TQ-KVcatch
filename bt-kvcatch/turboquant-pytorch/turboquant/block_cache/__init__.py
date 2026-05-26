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
    AttentionScorePageImportanceScorer,
    VKRatioPageImportanceScorer,
)
from .bit_allocator import (
    PageBitAllocator,
    FixedPageBitAllocator,
    TopRatioPageBitAllocator,
)
from .skvq_quantizer import SKVQPageCompressor
from .config import BlockCacheConfig
from .hf_cache import BlockKVCache
from .layer import BlockCacheLayer
from .reports import build_memory_report
from .backends import (
    PageQuantBackend,
    SKVQPageBackend,
    TurboQuantPageBackend,
    available_page_backends,
    build_page_backend,
    get_page_backend_class,
    register_page_backend,
)

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
    "AttentionScorePageImportanceScorer",
    "VKRatioPageImportanceScorer",
    "PageBitAllocator",
    "FixedPageBitAllocator",
    "TopRatioPageBitAllocator",
    "SKVQPageCompressor",
    "BlockCacheLayer",
    "build_memory_report",
    "PageQuantBackend",
    "TurboQuantPageBackend",
    "SKVQPageBackend",
    "available_page_backends",
    "build_page_backend",
    "get_page_backend_class",
    "register_page_backend",
    "BlockKVCache",
    "BlockCacheConfig",
]
