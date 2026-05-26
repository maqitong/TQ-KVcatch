"""Registry for page-level quantization backends."""

from __future__ import annotations

from typing import Any

from .base import PageQuantBackend


_BACKENDS: dict[str, type[PageQuantBackend]] = {}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def register_page_backend(
    name: str, backend_cls: type[PageQuantBackend]
) -> type[PageQuantBackend]:
    """Register a backend class under ``name``.

    External experiments can call this before constructing ``BlockKVCache`` to
    add a new page compressor without editing ``hf_cache.py``.
    """
    normalized = _normalize_name(name)
    if not issubclass(backend_cls, PageQuantBackend):
        raise TypeError("backend_cls must inherit PageQuantBackend")
    _BACKENDS[normalized] = backend_cls
    return backend_cls


def get_page_backend_class(name: str) -> type[PageQuantBackend]:
    normalized = _normalize_name(name)
    try:
        return _BACKENDS[normalized]
    except KeyError as exc:
        available = ", ".join(available_page_backends()) or "<none>"
        raise ValueError(
            f"unknown quant_backend: {name}. Available backends: {available}"
        ) from exc


def build_page_backend(name: str, **runtime: Any) -> PageQuantBackend:
    backend_cls = get_page_backend_class(name)
    return backend_cls.from_runtime(**runtime)


def available_page_backends() -> list[str]:
    return sorted(_BACKENDS)
