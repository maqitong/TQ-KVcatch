# KVcatch Page-Quantization Architecture

This project is organized around one rule:

```text
The cache framework owns pages. Quantization backends own algorithms.
```

The main path is `turboquant.block_cache`. It provides a HuggingFace-compatible
KV cache that splits each layer's `(B, H, S, D)` K/V tensors into fixed-size
sequence pages, protects sink/window pages in fp16, and compresses old sealed
pages with a pluggable backend.

## Runtime Flow

```text
model.forward / generate
  -> BlockKVCache.update(layer_idx)
  -> BlockCacheLayer.update()
  -> BlockTable.append()
  -> GroupingPolicy.on_seal()
  -> PageImportanceScorer + PageBitAllocator
  -> PageQuantBackend.compress()
  -> PageQuantBackend.decompress()
  -> dense K/V returned to attention
```

## Core Modules

- `turboquant/block_cache/hf_cache.py`
  HuggingFace cache entry point. It coordinates layer creation, attention
  recording, state restore, and report calls.
- `turboquant/block_cache/layer.py`
  Per-layer block table, page scheduling, mixed-precision assignment, and
  backend calls.
- `turboquant/block_cache/config.py`
  Shared cache configuration.
- `turboquant/block_cache/blocks.py`
  Page data structures: `KVBlock`, `BlockTable`, and `BlockState`.
- `turboquant/block_cache/policies.py`
  Page scheduling policies: token-block, sliding-window, and sink+window.
- `turboquant/block_cache/page_importance.py`
  Page scoring methods such as `k_norm`, `kv_norm`, `vk_ratio`, and
  `attention_score`.
- `turboquant/block_cache/bit_allocator.py`
  Fixed-bit and top-ratio mixed-precision allocation.
- `turboquant/block_cache/backends/`
  The algorithm plugin layer. Built-ins are TurboQuant and SKVQ-style page
  quantization.
- `turboquant/block_cache/reports.py`
  Memory and compression diagnostics.
- `turboquant/block_cache/methods.py`
  Shared experiment method selection, backend configuration, and cache factory
  construction for PPL, NIAH, LongBench, and the formal main experiment.

## Backend Boundary

New quantization methods should implement `PageQuantBackend`:

```python
from turboquant.block_cache import PageQuantBackend, register_page_backend


class MyBackend(PageQuantBackend):
    name = "my_backend"

    @classmethod
    def from_runtime(cls, **kwargs):
        return cls(...)

    def compress(self, key_states, value_states, *, key_bits, value_bits, layer_idx):
        return compressed_k, compressed_v

    def decompress(self, compressed_k, compressed_v, *, key_bits, value_bits, dtype):
        return key_states, value_states


register_page_backend(MyBackend.name, MyBackend)
```

After registration, use it with:

```python
cfg = BlockCacheConfig(quant_backend="my_backend")
cache = BlockKVCache(cfg)
```

This keeps `BlockKVCache` stable while allowing TurboQuant, SKVQ, KIVI-like,
RTN, or future methods to be compared under the same page scheduling and
mixed-precision framework.

## SKVQ Repository Role

The sibling `SKVQ` repository is an optional paper-baseline dependency. The
main TurboQuant PageMix path does not require it. Use it only when running a
native SKVQ paper baseline through `skvq_native_integration.py`.

The in-repo `block_cache` SKVQ backend is a page-level compressor used for
framework comparisons; it is not the same as the external SKVQ model-patching
implementation.
