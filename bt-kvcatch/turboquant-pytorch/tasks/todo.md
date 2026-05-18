# SKVQ + KVcatch + Page-Level Mixed Precision Todo

## Plan

- [x] Add page-level metadata to `KVBlock`.
- [x] Add page importance scoring with a first `k_norm` scorer.
- [x] Add page bit allocation with a top-ratio mixed-precision allocator.
- [x] Add a natural-group SKVQ page compressor with packed storage and dequantization.
- [x] Wire the SKVQ mixed-precision path into `BlockKVCache`.
- [x] Add focused block-cache tests for metadata, allocation, SKVQ roundtrip, and cache integration.
- [x] Run verification and record the result.
- [x] Add SKVQ reorder metadata support.

## Review

- Added page-level metadata to `KVBlock`: importance, K/V bit-widths, and debug metadata.
- Added `page_importance.py` with `k_norm` / `v_norm` / `kv_norm` scoring plus a random baseline.
- Added `bit_allocator.py` with fixed-bit and top-ratio mixed-precision allocation.
- Added `skvq_quantizer.py`, a natural-group SKVQ page compressor with packed uint8 storage, dequantization, clipping, optional reorder hooks, and 1.5-bit as 3-level quantization stored in a 2-bit container.
- Extended `BlockCacheConfig` and `BlockCacheLayer` so `quant_backend="skvq"` and `mixed_precision=True` route sealed pages through SKVQ page compression with per-page K/V bits.
- Extended `memory_report()` with `bit_histogram` and `precision_histogram`.
- Added block-cache tests for SKVQ compressor roundtrip and SKVQ mixed-precision page integration.
- Added a no-Scipy Gaussian Lloyd-Max fallback in `lloyd_max.py` so local tests can run in the current environment.
- Verification passed: `python -m py_compile ...` and `python -m turboquant.block_cache.test_block_cache`.
- Added SKVQ reorder metadata support through `BlockCacheConfig.reorder_file` or `BlockCacheConfig.reorder_meta`.
- `BlockKVCache` now loads shared reorder metadata once and passes per-layer `reorder_idx` / `cluster_st_inds` into `SKVQPageCompressor`.
- `SKVQPageCompressor` now uses reorder during quantization and inverse reorder during dequantization without charging reorder tensors as per-page storage.
- Added reorder tests for direct compressor roundtrip and full `BlockKVCache` integration.
- Re-verification passed: `python -m py_compile ...` and `python -m turboquant.block_cache.test_block_cache`.
