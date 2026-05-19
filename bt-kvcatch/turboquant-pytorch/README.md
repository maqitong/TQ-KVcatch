# KVcatch

Block-structured mixed-precision KV cache quantization for HuggingFace
generation. This fork combines three ideas:

- KVcatch-style page/block cache management.
- SKVQ-style sink/window protection, reorder metadata, and group-wise quantization.
- TurboQuant vector quantization for compressed old pages.

The main research path is:

```text
[prefix sink: FP16] [old sealed pages: compressed] [recent window: FP16]
                                      |
                                      +-- page-level mixed precision
                                          high-importance page: more bits
                                          low-importance page: fewer bits
```

## Current Status

Implemented:

- HuggingFace-compatible `BlockKVCache`.
- Page/block splitting for K/V cache tensors shaped `(B, H, S, D)`.
- `TokenBlockPolicy`, `WindowBlockPolicy`, and `HybridPolicy`.
- TurboQuant page compression backend with asymmetric K/V bits.
- SKVQ page compression backend with reorder metadata and 1.5-bit support.
- TurboQuant reorder metadata support.
- Page-level mixed precision using page importance.
- Importance metrics: `k_norm`, `v_norm`, `kv_norm`, `vk_ratio`, `attention_score`, `random`.
- First/last layer protection.
- K/V asymmetric SKVQ group size.
- PPL, NIAH, LongBench-style, main comparison, calibration, and ablation scripts.

Important caveats:

- TurboQuant currently supports integer bit widths only, for example K4/V4 or K2/V2.
- SKVQ supports 1.5-bit storage, for example K2/V1.5.
- Reorder metadata is wired into both SKVQ and TurboQuant backends.
- Attention-score importance is available, but in ordinary `generate()` the returned attentions usually arrive after prefill compression. For now, `k_norm` is the safest default.

## Quick Start

Install dependencies:

```powershell
cd D:\KVcatch\bt-kvcatch\turboquant-pytorch
pip install -r requirements.txt
pip install -e .
```

Run unit tests:

```powershell
python -m turboquant.block_cache.test_block_cache
```

Run a small local NIAH smoke test:

```powershell
python -m turboquant.block_cache.eval_niah `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_tq_mix `
  --context-lengths 96 `
  --positions 0.5 `
  --seeds 0 `
  --max-new-tokens 8 `
  --block-size 8 `
  --sink 8 `
  --window 32
```

Run the 5-method main comparison:

```powershell
python -m turboquant.block_cache.experiment_main `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --context-lengths 512,1024 `
  --positions 0.1,0.5,0.9 `
  --seeds 0 `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --max-cached-decompressed-blocks 128
```

Run a parameter sweep:

```powershell
python -m turboquant.block_cache.ablation `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_tq_mix `
  --context-lengths 2048,4096 `
  --positions 0.1,0.5,0.9 `
  --seeds 0,1,2 `
  --sweep block_size=8,16,32 `
  --sweep important_ratio=0.2,0.3,0.5 `
  --sweep high_key_bits=4,6 `
  --sweep high_value_bits=4 `
  --sweep low_key_bits=2 `
  --sweep low_value_bits=2
```

## Core Backends

Use `BlockKVCache` directly:

```python
from turboquant.block_cache import BlockCacheConfig, BlockKVCache, HybridPolicy

cfg = BlockCacheConfig(
    block_size=16,
    policy=HybridPolicy(sink_size=16, window_size=128),
    quant_backend="turboquant",
    key_bits=2,
    value_bits=2,
    mixed_precision=True,
    importance_metric="k_norm",
    important_ratio=0.3,
    high_key_bits=4,
    high_value_bits=4,
    low_key_bits=2,
    low_value_bits=2,
    protected_layers=1,
    protected_key_bits=8,
    protected_value_bits=8,
)
cache = BlockKVCache(cfg)
```

Then pass it to HuggingFace generation:

```python
output = model.generate(
    input_ids=input_ids,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=64,
)
print(cache.memory_report())
```

Save and restore a cache:

```python
import torch

torch.save(cache.state_dict(), "runs/cache_state.pt")

restored = BlockKVCache(cfg)
restored.load_state_dict(torch.load("runs/cache_state.pt", map_location="cpu"))
print(restored.memory_report())
```

## Evaluation Scripts

PPL:

```powershell
python -m turboquant.block_cache.eval_ppl `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --inline-text "A short evaluation text for cache smoke testing." `
  --backend all `
  --seq-len 64 `
  --stride 32
```

NIAH:

```powershell
python -m turboquant.block_cache.eval_niah `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend all `
  --context-lengths 2048,4096,8192 `
  --positions 0.1,0.5,0.9 `
  --output-dir runs\niah
```

LongBench-style QA:

```powershell
python -m turboquant.block_cache.eval_longbench `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --toy-sample `
  --backend all `
  --output-dir runs\longbench_toy
```

Calibration for reorder metadata:

```powershell
python -m turboquant.block_cache.calibration `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --inline-text "Calibration text for K and V projection statistics." `
  --n-samples 1 `
  --seq-len 32 `
  --metric absmax_sort `
  --key-group-size 128 `
  --value-group-size 64 `
  --output runs\llama32_reorder_meta.pt
```

Use reorder metadata with TurboQuant:

```powershell
python -m turboquant.block_cache.eval_niah `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_tq_mix `
  --reorder-file runs\llama32_reorder_meta.pt `
  --context-lengths 2048 `
  --positions 0.5
```

Use reorder metadata with SKVQ:

```powershell
python -m turboquant.block_cache.eval_niah `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --backend block_skvq_mix `
  --reorder-file runs\llama32_reorder_meta.pt `
  --key-group-size 128 `
  --value-group-size 64
```

## Architecture

Main files:

- `turboquant/block_cache/hf_cache.py`: HuggingFace cache wrapper and per-layer block cache.
- `turboquant/block_cache/blocks.py`: `KVBlock` and `BlockTable`.
- `turboquant/block_cache/policies.py`: sink/window/grouping policies.
- `turboquant/block_cache/bit_allocator.py`: fixed and top-ratio mixed precision bit allocation.
- `turboquant/block_cache/page_importance.py`: page importance scorers.
- `turboquant/block_cache/quantizer.py`: TurboQuant page compressor wrapper.
- `turboquant/block_cache/skvq_quantizer.py`: SKVQ group-wise compressor with reorder and 1.5-bit support.
- `turboquant/block_cache/calibration.py`: K/V projection stats to reorder metadata.
- `turboquant/block_cache/experiment_main.py`: 5-method core comparison.
- `turboquant/block_cache/ablation.py`: parameter sweep runner.
- `turboquant/block_cache/profile_memory.py`: CUDA memory and latency profiler.
- `BlockKVCache.state_dict/load_state_dict`: cache save/load for reproducibility.
- `max_cached_decompressed_blocks`: optional LRU cache for dequantized old pages.

Runtime flow:

```text
1. Model forward produces key_states/value_states for one layer.
2. BlockKVCache.update routes them to the layer's BlockCacheLayer.
3. BlockTable appends tokens and seals full pages.
4. HybridPolicy keeps sink and recent window pages in FP16.
5. Compressible old pages are scored by a page-importance metric.
6. Bit allocator assigns high/low precision per page.
7. Optional first/last layer protection overrides page bits.
8. TurboQuant or SKVQ compresses the selected pages.
9. Before attention, compressed pages are materialized back to dense K/V.
10. memory_report records compression ratio, FP16 pages, compressed pages, and bit histograms.
```

## Recommended Experiments

On a two-RTX-4090 server, start with:

```powershell
python -m turboquant.block_cache.experiment_main `
  --model D:\model\Llama3.2_3B `
  --local-files-only `
  --context-lengths 2048,4096,8192 `
  --positions 0.1,0.5,0.9 `
  --seeds 0,1,2 `
  --block-size 16 `
  --sink 16 `
  --window 128 `
  --important-ratio 0.3 `
  --high-key-bits 4 `
  --high-value-bits 4 `
  --low-key-bits 2 `
  --low-value-bits 2 `
  --protected-layers 1
```

Then run ablations over:

- `block_size`: 8, 16, 32
- `important_ratio`: 0.2, 0.3, 0.5
- `importance_metric`: `k_norm`, `kv_norm`, `vk_ratio`, `random`
- `high_key_bits/high_value_bits`: K4/V4, K6/V4
- `low_key_bits/low_value_bits`: K2/V2
- `sink/window`: sink 16/32, window 128/256

## Legacy TurboQuant README

The original TurboQuant README was preserved as
`README_turboquant_v3.md`.

## License

MIT
