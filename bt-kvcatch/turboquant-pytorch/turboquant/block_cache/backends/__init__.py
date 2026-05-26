"""Page quantization backends for block-structured KV cache."""

from .base import PageQuantBackend
from .registry import (
    available_page_backends,
    build_page_backend,
    get_page_backend_class,
    register_page_backend,
)
from .skvq import SKVQPageBackend
from .turboquant import TurboQuantPageBackend


register_page_backend(TurboQuantPageBackend.name, TurboQuantPageBackend)
register_page_backend("tq", TurboQuantPageBackend)
register_page_backend(SKVQPageBackend.name, SKVQPageBackend)


__all__ = [
    "PageQuantBackend",
    "TurboQuantPageBackend",
    "SKVQPageBackend",
    "available_page_backends",
    "build_page_backend",
    "get_page_backend_class",
    "register_page_backend",
]
