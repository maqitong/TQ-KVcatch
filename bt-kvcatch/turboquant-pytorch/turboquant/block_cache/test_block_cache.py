"""Self-contained tests for the block-structured KV cache (no model needed).

Run with:
    python -m turboquant.block_cache.test_block_cache
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    BlockState,
    BlockTable,
    GroupingPolicy,
    HybridPolicy,
    NormPageImportanceScorer,
    PageQuantBackend,
    PageImportanceScorer,
    SKVQPageCompressor,
    TokenBlockPolicy,
    TopRatioPageBitAllocator,
    WindowBlockPolicy,
    available_page_backends,
    register_page_backend,
)
from turboquant.block_cache.methods import (
    NIAH_ALL_BACKENDS,
    cache_factory_for_backend,
    parse_backend_selection,
)
from turboquant.block_cache.quantizer import BlockMSECompressor


def _kv(B: int, H: int, S: int, D: int, dtype=torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    k = torch.randn(B, H, S, D, generator=g).to(dtype)
    v = torch.randn(B, H, S, D, generator=g).to(dtype)
    return k, v


def test_block_table_splits_into_blocks():
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 10, 8)
    sealed = table.append(k, v)
    assert table.total_len == 10
    assert len(table.blocks) == 3  # 4 + 4 + 2
    assert [b.current_len for b in table.blocks] == [4, 4, 2]
    assert [b.state for b in table.blocks] == [
        BlockState.SEALED, BlockState.SEALED, BlockState.FILLING
    ]
    assert len(sealed) == 2
    print("ok: test_block_table_splits_into_blocks")


def test_token_block_policy_compresses_all_sealed():
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 10, 8)
    sealed = table.append(k, v)
    policy = TokenBlockPolicy()
    chosen = policy.on_seal(sealed, table)
    assert {b.block_idx for b in chosen} == {0, 1}
    print("ok: test_token_block_policy_compresses_all_sealed")


def test_window_policy_keeps_recent_fp16():
    # block_size=4, window=4. After 10 tokens: blocks 0-2 ([0,4) [4,8) [8,10)).
    # window covers last 4 tokens = positions 6..10.
    # block 0 ends at 4 (<= 6) -> compress.
    # block 1 ends at 8 (> 6) -> still inside window, keep fp16.
    # block 2 is FILLING -> never compressed.
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 10, 8)
    sealed = table.append(k, v)
    policy = WindowBlockPolicy(window_size=4)
    chosen = policy.on_seal(sealed, table)
    assert [b.block_idx for b in chosen] == [0]
    print("ok: test_window_policy_keeps_recent_fp16")


def test_hybrid_policy_sink_and_window():
    # block_size=4, sink=2, window=4. After 10 tokens:
    # block 0 ends at 4: not in sink (4 > 2), not in window (4 <= 6) -> compress
    # block 1 ends at 8: in window (8 > 6) -> keep
    # but sink_size=2 < block_size=4, so block 0 isn't entirely sink.
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 10, 8)
    sealed = table.append(k, v)
    chosen = HybridPolicy(sink_size=2, window_size=4).on_seal(sealed, table)
    assert [b.block_idx for b in chosen] == [0]
    # Now with sink=4 covering the whole first block: nothing to compress.
    table2 = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    sealed2 = table2.append(k, v)
    chosen2 = HybridPolicy(sink_size=4, window_size=4).on_seal(sealed2, table2)
    assert chosen2 == []
    print("ok: test_hybrid_policy_sink_and_window")


def test_per_vector_compress_decompress_roundtrip():
    cmp = BlockMSECompressor(head_dim=64, bits=8, seed=1, granularity="per-vector")
    k, _ = _kv(1, 2, 16, 64, dtype=torch.float32)
    d = cmp.compress(k)
    out = cmp.decompress(d)
    err = (out - k).norm() / k.norm()
    assert err.item() < 0.15, f"per-vector 8-bit error too high: {err.item()}"
    print(f"ok: test_per_vector_compress_decompress_roundtrip  err={err.item():.4f}")


def test_per_block_compress_decompress_roundtrip():
    cmp = BlockMSECompressor(head_dim=64, bits=8, seed=1, granularity="per-block")
    k, _ = _kv(1, 2, 16, 64, dtype=torch.float32)
    d = cmp.compress(k)
    out = cmp.decompress(d)
    err = (out - k).norm() / k.norm()
    # per-block is lossier than per-vector, allow more headroom
    assert err.item() < 0.4, f"per-block 8-bit error too high: {err.item()}"
    print(f"ok: test_per_block_compress_decompress_roundtrip  err={err.item():.4f}")


def test_norm_importance_score_many_matches_single_block_scores():
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 16, 8)
    blocks = table.append(k, v)
    scorer = NormPageImportanceScorer("k_norm")

    batched = scorer.score_many(blocks, table, layer_idx=0)
    single = [scorer.score(block, table, layer_idx=0) for block in blocks]

    assert torch.allclose(torch.tensor(batched), torch.tensor(single), atol=1e-6)
    print("ok: test_norm_importance_score_many_matches_single_block_scores")


class StaticScoreScorer(PageImportanceScorer):
    name = "static"

    def __init__(self, scores):
        self.scores = list(scores)

    def score(self, block, table, layer_idx):
        return float(self.scores[block.block_idx])

    def score_many(self, blocks, table, layer_idx):
        return [self.score(block, table, layer_idx) for block in blocks]


def test_top_ratio_allocator_run_aware_selects_contiguous_segment():
    table = BlockTable(block_size=4, head_dim=8, n_kv_heads=2, batch_size=1)
    k, v = _kv(1, 2, 16, 8)
    blocks = table.append(k, v)
    scorer = StaticScoreScorer([10.0, 1.0, 9.0, 1.0])

    run_aware = TopRatioPageBitAllocator(
        scorer=scorer,
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
        run_aware=True,
    )
    assignments = run_aware.assign_many(blocks, table, layer_idx=0)
    high_ids = {
        block_idx for block_idx, bits in assignments.items() if bits == (4.0, 4.0)
    }
    assert high_ids == {0, 1}

    scattered = TopRatioPageBitAllocator(
        scorer=scorer,
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
        run_aware=False,
    )
    assignments = scattered.assign_many(blocks, table, layer_idx=0)
    high_ids = {
        block_idx for block_idx, bits in assignments.items() if bits == (4.0, 4.0)
    }
    assert high_ids == {0, 2}
    print("ok: test_top_ratio_allocator_run_aware_selects_contiguous_segment")


def test_block_kv_cache_update_returns_full_history():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4, key_bits=8, value_bits=8,
        policy=TokenBlockPolicy(),
    ))
    # Simulate prefill of 10 tokens at layer 0
    k, v = _kv(1, 2, 10, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    assert full_k.shape == (1, 2, 10, 8)
    assert full_v.shape == (1, 2, 10, 8)
    assert cache.get_seq_length(0) == 10
    # Now decode step: append 1 token
    k1, v1 = _kv(1, 2, 1, 8)
    full_k, full_v = cache.update(k1, v1, layer_idx=0)
    assert full_k.shape == (1, 2, 11, 8)
    assert cache.get_seq_length(0) == 11
    print("ok: test_block_kv_cache_update_returns_full_history")


def test_incremental_materialize_matches_legacy_path():
    common = dict(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=HybridPolicy(sink_size=4, window_size=4),
        quant_backend="turboquant",
    )
    inc = BlockKVCache(BlockCacheConfig(**common, incremental_materialize=True))
    legacy = BlockKVCache(BlockCacheConfig(**common, incremental_materialize=False))

    g = torch.Generator().manual_seed(123)
    for n_new in [3, 3, 2, 1, 4, 1, 1]:
        k = torch.randn(1, 2, n_new, 8, generator=g).half()
        v = torch.randn(1, 2, n_new, 8, generator=g).half()
        inc_k, inc_v = inc.update(k, v, layer_idx=0)
        legacy_k, legacy_v = legacy.update(k, v, layer_idx=0)
        assert torch.allclose(inc_k, legacy_k, atol=0, rtol=0)
        assert torch.allclose(inc_v, legacy_v, atol=0, rtol=0)

    layer = inc.layers[0]
    assert layer._mat_k is not None
    same_k, same_v = layer._materialize(dtype=torch.float16)
    assert same_k is layer._mat_k
    assert same_v is layer._mat_v
    print("ok: test_incremental_materialize_matches_legacy_path")


def test_turboquant_batched_compression_matches_single_page_path():
    cfg = BlockCacheConfig(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        incremental_materialize=False,
    )
    batched = BlockKVCache(cfg)
    single = BlockKVCache(cfg)

    k, v = _kv(1, 2, 12, 8)
    batched_k, batched_v = batched.update(k, v, layer_idx=0)
    single_parts_k = []
    single_parts_v = []
    for start in range(0, 12, 4):
        out_k, out_v = single.update(
            k[:, :, start : start + 4, :],
            v[:, :, start : start + 4, :],
            layer_idx=0,
        )
        single_parts_k.append(out_k)
        single_parts_v.append(out_v)

    assert torch.allclose(batched_k, single_parts_k[-1], atol=0, rtol=0)
    assert torch.allclose(batched_v, single_parts_v[-1], atol=0, rtol=0)
    assert all(b.state == BlockState.COMPRESSED for b in batched.layers[0].table.blocks)
    assert all(b.state == BlockState.COMPRESSED for b in single.layers[0].table.blocks)
    assert len(batched.layers[0]._tq_compressed_runs) == 1
    assert all("__run_id" in b.compressed_k for b in batched.layers[0].table.blocks)
    assert not any(
        torch.is_tensor(value)
        for b in batched.layers[0].table.blocks
        for value in b.compressed_k.values()
    )
    print("ok: test_turboquant_batched_compression_matches_single_page_path")


def test_turboquant_batched_materialize_matches_blockwise_path():
    common = dict(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        incremental_materialize=False,
    )
    batched = BlockKVCache(
        BlockCacheConfig(**common, max_cached_decompressed_blocks=0)
    )
    blockwise = BlockKVCache(
        BlockCacheConfig(**common, max_cached_decompressed_blocks=1)
    )

    k, v = _kv(1, 2, 12, 8)
    batched_k, batched_v = batched.update(k, v, layer_idx=0)
    blockwise_k, blockwise_v = blockwise.update(k, v, layer_idx=0)

    assert torch.allclose(batched_k, blockwise_k, atol=0, rtol=0)
    assert torch.allclose(batched_v, blockwise_v, atol=0, rtol=0)
    assert len(batched.layers[0]._decompressed_cache) == 0
    assert len(blockwise.layers[0]._decompressed_cache) == 1
    print("ok: test_turboquant_batched_materialize_matches_blockwise_path")


def test_live_fp16_blocks_are_compacted_after_prefill():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=WindowBlockPolicy(window_size=4),
        quant_backend="turboquant",
        incremental_materialize=False,
        max_cached_decompressed_blocks=0,
    ))
    k, v = _kv(1, 2, 12, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    layer = cache.layers[0]
    blocks = layer.table.blocks

    assert full_k.shape == (1, 2, 12, 8)
    assert full_v.shape == (1, 2, 12, 8)
    assert [blk.state for blk in blocks] == [
        BlockState.COMPRESSED,
        BlockState.COMPRESSED,
        BlockState.SEALED,
    ]
    assert all(blk.fp16_k is None for blk in blocks[:2])
    assert blocks[2].fp16_k.is_contiguous()
    assert blocks[2].fp16_v.is_contiguous()
    print("ok: test_live_fp16_blocks_are_compacted_after_prefill")


def test_block_table_total_len_tracks_crop_reset_and_restore():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
    ))
    k, v = _kv(1, 2, 12, 8)
    cache.update(k, v, layer_idx=0)
    layer = cache.layers[0]
    assert layer.table.total_len == 12

    layer.crop(8)
    assert layer.table.total_len == 8

    restored = BlockKVCache(cache.config)
    restored.load_state_dict(cache.state_dict())
    assert restored.layers[0].table.total_len == 8

    restored.layers[0].reset()
    assert restored.layers[0].table.total_len == 0
    print("ok: test_block_table_total_len_tracks_crop_reset_and_restore")



def test_block_kv_cache_window_policy_memory_drops():
    # Long-ish sequence so most tokens roll out of window
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4, key_bits=4, value_bits=4,
        policy=WindowBlockPolicy(window_size=8),
    ))
    k, v = _kv(1, 2, 64, 8)
    cache.update(k, v, layer_idx=0)
    report = cache.memory_report()
    assert report["compression_ratio"] > 1.5, report
    print(
        f"ok: test_block_kv_cache_window_policy_memory_drops  "
        f"ratio={report['compression_ratio']:.2f}x"
    )


def test_block_kv_cache_per_layer_independence():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4, key_bits=8, value_bits=8, policy=TokenBlockPolicy(),
    ))
    k, v = _kv(1, 2, 10, 8)
    cache.update(k, v, layer_idx=0)
    cache.update(k * 2, v * 2, layer_idx=1)
    assert len(cache) == 2
    fk0, _ = cache.layers[0]._materialize(dtype=torch.float16)
    fk1, _ = cache.layers[1]._materialize(dtype=torch.float16)
    # Layer 1 fed 2x scale -> expect ~2x norm (allow quantization drift)
    ratio = fk1.float().norm() / fk0.float().norm()
    assert 1.7 < ratio.item() < 2.3, ratio.item()
    print(f"ok: test_block_kv_cache_per_layer_independence  ratio={ratio.item():.2f}")


def test_reorder_cache_permutes_batch():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4, key_bits=8, value_bits=8, policy=TokenBlockPolicy(),
    ))
    B, H, S, D = 2, 2, 10, 8
    g = torch.Generator().manual_seed(7)
    k = torch.randn(B, H, S, D, generator=g).half()
    v = torch.randn(B, H, S, D, generator=g).half()
    cache.update(k, v, layer_idx=0)
    cache.layers[0].reorder_cache(torch.tensor([1, 0]))
    fk, _ = cache.layers[0]._materialize(dtype=torch.float16)
    err = (fk[0] - k[1].float()).norm() / k[1].float().norm()
    # quantization tolerance — index 0 should now correspond to original batch 1
    assert err.item() < 0.2, err.item()
    print(f"ok: test_reorder_cache_permutes_batch  err={err.item():.4f}")


def test_skvq_page_compressor_roundtrip():
    cmp = SKVQPageCompressor(
        head_dim=16,
        n_kv_heads=2,
        group_size=8,
        clipping=1.0,
    )
    k, _ = _kv(1, 2, 8, 16, dtype=torch.float32)
    compressed = cmp.compress(k, bits=4, ttype="k", layer_idx=0)
    out = cmp.decompress(compressed)
    err = (out - k).norm() / k.norm()
    assert compressed["backend"] == "skvq"
    assert compressed["qdata"].dtype == torch.uint8
    assert out.shape == k.shape
    assert out.dtype == k.dtype
    assert err.item() < 0.25, f"SKVQ 4-bit error too high: {err.item()}"

    compressed_15 = cmp.compress(k, bits=1.5, ttype="k", layer_idx=0)
    out_15 = cmp.decompress(compressed_15)
    assert compressed_15["bits"] == 1.5
    assert compressed_15["container_bits"] == 2
    assert out_15.shape == k.shape
    print(f"ok: test_skvq_page_compressor_roundtrip  err={err.item():.4f}")


def test_skvq_page_compressor_reorder_roundtrip():
    hidden = 32
    idx = torch.arange(hidden - 1, -1, -1)
    gst = torch.tensor([0, 8, 16, 24, 32])
    cmp = SKVQPageCompressor(
        head_dim=16,
        n_kv_heads=2,
        group_size=8,
        clipping=1.0,
        reorder_idx={"k": idx, "v": idx},
        group_st_idx={"k": gst, "v": gst},
    )
    k, _ = _kv(1, 2, 8, 16, dtype=torch.float32)
    compressed = cmp.compress(k, bits=4, ttype="k", layer_idx=0)
    out = cmp.decompress(compressed)
    err = (out - k).norm() / k.norm()
    assert compressed["reordered"] is True
    assert out.shape == k.shape
    assert err.item() < 0.3, f"SKVQ reorder 4-bit error too high: {err.item()}"
    print(f"ok: test_skvq_page_compressor_reorder_roundtrip  err={err.item():.4f}")


def test_skvq_page_compressor_asymmetric_group_size():
    cmp = SKVQPageCompressor(
        head_dim=16,
        n_kv_heads=2,
        group_size=16,
        key_group_size=8,
        value_group_size=32,
        clipping=1.0,
    )
    k, v = _kv(1, 2, 8, 16, dtype=torch.float32)
    ck = cmp.compress(k, bits=4, ttype="k", layer_idx=0)
    cv = cmp.compress(v, bits=4, ttype="v", layer_idx=0)
    assert ck["group_widths"] == [8, 8, 8, 8]
    assert cv["group_widths"] == [32]
    assert cmp.decompress(ck).shape == k.shape
    assert cmp.decompress(cv).shape == v.shape
    print("ok: test_skvq_page_compressor_asymmetric_group_size")


def test_block_kv_cache_skvq_mixed_precision_pages():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        policy=TokenBlockPolicy(),
        quant_backend="skvq",
        mixed_precision=True,
        importance_metric="k_norm",
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=1.5,
        group_size=8,
        clipping=1.0,
    ))
    k, v = _kv(1, 2, 12, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    assert full_k.shape == (1, 2, 12, 8)
    assert full_v.shape == (1, 2, 12, 8)
    blocks = cache.layers[0].table.blocks
    assert all(b.state == BlockState.COMPRESSED for b in blocks)
    assert {b.page_meta["precision"] for b in blocks} == {"high", "low"}
    assert all(b.key_bits is not None and b.value_bits is not None for b in blocks)

    report = cache.memory_report()
    assert report["precision_histogram"]["high"] == 2
    assert report["precision_histogram"]["low"] == 1
    assert report["bit_histogram"]["K4.0/V4.0"] == 2
    assert report["bit_histogram"]["K2.0/V1.5"] == 1
    print("ok: test_block_kv_cache_skvq_mixed_precision_pages")


def test_block_kv_cache_skvq_reorder_metadata():
    hidden = 16
    idx = torch.arange(hidden - 1, -1, -1)
    gst = torch.tensor([0, 8, 16])
    reorder_meta = {
        "reorder_indices": [(idx, idx)],
        "cluster_st_inds": [(gst, gst)],
    }
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=4,
        value_bits=4,
        policy=TokenBlockPolicy(),
        quant_backend="skvq",
        group_size=8,
        clipping=1.0,
        reorder_meta=reorder_meta,
    ))
    k, v = _kv(1, 2, 8, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    err = (full_k.float() - k.float()).norm() / k.float().norm()
    blocks = cache.layers[0].table.blocks
    assert full_k.shape == k.shape
    assert full_v.shape == v.shape
    assert all(b.compressed_k["reordered"] for b in blocks)
    assert cache.layers[0].skvq_compressor.reorder_idx["k"].equal(idx)
    assert err.item() < 0.3, f"SKVQ cache reorder error too high: {err.item()}"
    print(f"ok: test_block_kv_cache_skvq_reorder_metadata  err={err.item():.4f}")


def test_block_kv_cache_turboquant_reorder_metadata():
    hidden = 16
    idx = torch.arange(hidden - 1, -1, -1)
    reorder_meta = {
        "reorder_indices": [(idx, idx)],
    }
    cfg = BlockCacheConfig(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        reorder_meta=reorder_meta,
    )
    cache = BlockKVCache(cfg)
    k, v = _kv(1, 2, 8, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    blocks = cache.layers[0].table.blocks
    err = (full_k.float() - k.float()).norm() / k.float().norm()

    assert full_k.shape == k.shape
    assert full_v.shape == v.shape
    assert all(b.compressed_k["backend"] == "turboquant" for b in blocks)
    assert all(b.compressed_k["tq_reordered"] for b in blocks)
    assert cache.layers[0].tq_reorder_idx["k"].equal(idx)
    assert err.item() < 0.2, f"TurboQuant cache reorder error too high: {err.item()}"

    restored = BlockKVCache(cfg)
    restored.load_state_dict(cache.state_dict())
    restored_k, _ = restored.layers[0]._materialize(dtype=torch.float16)
    assert torch.allclose(restored_k, full_k, atol=0, rtol=0)
    print(f"ok: test_block_kv_cache_turboquant_reorder_metadata  err={err.item():.4f}")


def test_custom_page_backend_registry():
    class IdentityPageBackend(PageQuantBackend):
        name = "identity_test"

        @classmethod
        def from_runtime(cls, **kwargs):
            return cls()

        def compress(
            self,
            key_states,
            value_states,
            *,
            key_bits,
            value_bits,
            layer_idx,
        ):
            return (
                {"backend": self.name, "payload": key_states.clone()},
                {"backend": self.name, "payload": value_states.clone()},
            )

        def decompress(
            self,
            compressed_k,
            compressed_v,
            *,
            key_bits,
            value_bits,
            dtype,
        ):
            return compressed_k["payload"].to(dtype), compressed_v["payload"].to(dtype)

    register_page_backend(IdentityPageBackend.name, IdentityPageBackend)
    assert IdentityPageBackend.name in available_page_backends()

    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=TokenBlockPolicy(),
        quant_backend=IdentityPageBackend.name,
    ))
    k, v = _kv(1, 2, 8, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    blocks = cache.layers[0].table.blocks

    assert torch.allclose(full_k, k, atol=0, rtol=0)
    assert torch.allclose(full_v, v, atol=0, rtol=0)
    assert all(b.compressed_k["backend"] == IdentityPageBackend.name for b in blocks)
    print("ok: test_custom_page_backend_registry")


def test_shared_method_cache_factory():
    args = SimpleNamespace(
        policy="hybrid",
        block_size=4,
        sink=4,
        window=8,
        key_bits=2,
        value_bits=2,
        granularity="per-vector",
        importance_metric="k_norm",
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
        num_layers=2,
        protected_layers=0,
        protected_key_bits=8,
        protected_value_bits=8,
        group_size=8,
        key_group_size=None,
        value_group_size=None,
        clipping=1.0,
        reorder_file=None,
        max_cached_decompressed_blocks=0,
    )
    assert parse_backend_selection("all", all_backends=NIAH_ALL_BACKENDS) == NIAH_ALL_BACKENDS

    cache = cache_factory_for_backend(args, "block_tq_mix")()
    assert isinstance(cache, BlockKVCache)
    assert cache.config.quant_backend == "turboquant"
    assert cache.config.mixed_precision is True

    skvq_cache = cache_factory_for_backend(args, "block_skvq")()
    assert skvq_cache.config.quant_backend == "skvq"
    assert skvq_cache.config.mixed_precision is False
    print("ok: test_shared_method_cache_factory")


def test_paper_pure_mix_protection_defaults_are_latency_safe():
    from turboquant.block_cache.skvq_native_integration import (
        paper_pure_layer_protection,
    )

    default_args = SimpleNamespace(
        protected_layers=0,
        protected_key_bits=8,
        protected_value_bits=8,
    )
    assert paper_pure_layer_protection("tq_pure_mix", default_args) == (1, 4.0, 4.0)

    explicit_args = SimpleNamespace(
        protected_layers=1,
        protected_key_bits=8,
        protected_value_bits=8,
    )
    assert paper_pure_layer_protection("tq_pure_mix", explicit_args) == (1, 8.0, 8.0)

    disabled_args = SimpleNamespace(
        protected_layers=-1,
        protected_key_bits=8,
        protected_value_bits=8,
    )
    assert paper_pure_layer_protection("tq_pure_mix", disabled_args) == (0, 4.0, 4.0)
    assert paper_pure_layer_protection("tq_pure", explicit_args)[0] == 0
    print("ok: test_paper_pure_mix_protection_defaults_are_latency_safe")


def test_block_kv_cache_protected_layers_override_bits():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        num_layers=4,
        protected_layers=1,
        protected_key_bits=8,
        protected_value_bits=4,
    ))
    k, v = _kv(1, 2, 8, 8)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    cache.update(k, v, layer_idx=3)

    protected_first = cache.layers[0].table.blocks[0]
    middle = cache.layers[1].table.blocks[0]
    protected_last = cache.layers[3].table.blocks[0]

    assert (protected_first.key_bits, protected_first.value_bits) == (8.0, 4.0)
    assert (middle.key_bits, middle.value_bits) == (2.0, 2.0)
    assert (protected_last.key_bits, protected_last.value_bits) == (8.0, 4.0)
    assert protected_first.page_meta["precision"] == "protected"
    assert protected_last.page_meta["precision"] == "protected"

    report = cache.memory_report()
    assert report["bit_histogram"]["K8.0/V4.0"] == 4
    assert report["bit_histogram"]["K2.0/V2.0"] == 2
    assert report["precision_histogram"]["protected"] == 4
    print("ok: test_block_kv_cache_protected_layers_override_bits")


class DeferredPolicy(GroupingPolicy):
    def __init__(self):
        self.enabled = False

    def on_seal(self, sealed, table):
        if not self.enabled:
            return []
        return [blk for blk in table.blocks if blk.state == BlockState.SEALED]


def test_attention_score_importance_drives_mixed_precision():
    policy = DeferredPolicy()
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=policy,
        quant_backend="turboquant",
        mixed_precision=True,
        importance_metric="attention_score",
        important_ratio=0.25,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
    ))

    k, v = _kv(1, 2, 12, 8)
    cache.update(k, v, layer_idx=0)

    # Three sealed pages are still FP16. Give the first page most attention.
    attn = torch.zeros(1, 1, 1, 12)
    attn[..., 0:4] = 1.0
    attn[..., 4:8] = 0.1
    attn[..., 8:12] = 0.01
    cache.record_attention(0, attn)

    policy.enabled = True
    k1, v1 = _kv(1, 2, 4, 8)
    cache.update(k1, v1, layer_idx=0)

    blocks = cache.layers[0].table.blocks
    assert all(blk.state == BlockState.COMPRESSED for blk in blocks[:3])
    assert blocks[3].state == BlockState.SEALED
    assert blocks[3].page_meta["precision"] == "deferred"
    assert blocks[0].page_meta["precision"] == "high"
    assert (blocks[0].key_bits, blocks[0].value_bits) == (4.0, 4.0)
    assert all((blk.key_bits, blk.value_bits) == (2.0, 2.0) for blk in blocks[1:3])
    print("ok: test_attention_score_importance_drives_mixed_precision")


def test_attention_score_deferred_pages_compress_after_recording():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        mixed_precision=True,
        importance_metric="attention_score",
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
    ))

    k, v = _kv(1, 2, 8, 8)
    cache.update(k, v, layer_idx=0)
    blocks = cache.layers[0].table.blocks
    assert all(blk.state == BlockState.SEALED for blk in blocks)
    assert all(blk.page_meta["precision"] == "deferred" for blk in blocks)

    attn = torch.zeros(1, 1, 1, 8)
    attn[..., 0:4] = 0.1
    attn[..., 4:8] = 1.0
    cache.record_attention(0, attn)

    assert all(blk.state == BlockState.COMPRESSED for blk in blocks)
    assert blocks[1].page_meta["precision"] == "high"
    assert blocks[0].page_meta["precision"] == "low"
    print("ok: test_attention_score_deferred_pages_compress_after_recording")


def test_budgeted_quant_cursor_limits_compression_per_update():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=8,
        value_bits=8,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        quant_budget_per_update=1,
    ))

    k, v = _kv(1, 2, 12, 8)
    cache.update(k, v, layer_idx=0)
    blocks = cache.layers[0].table.blocks
    assert [blk.state for blk in blocks] == [
        BlockState.COMPRESSED,
        BlockState.SEALED,
        BlockState.SEALED,
    ]
    report = cache.memory_report()
    assert report["n_compressed_blocks"] == 1
    assert report["n_pending_quant_blocks"] == 2

    k1, v1 = _kv(1, 2, 1, 8)
    cache.update(k1, v1, layer_idx=0)
    assert [blk.state for blk in blocks[:3]] == [
        BlockState.COMPRESSED,
        BlockState.COMPRESSED,
        BlockState.SEALED,
    ]
    assert cache.memory_report()["n_pending_quant_blocks"] == 1
    print("ok: test_budgeted_quant_cursor_limits_compression_per_update")


def test_budgeted_attention_score_compresses_ready_pages_by_budget():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        mixed_precision=True,
        importance_metric="attention_score",
        important_ratio=0.5,
        high_key_bits=4,
        high_value_bits=4,
        low_key_bits=2,
        low_value_bits=2,
        quant_budget_per_update=1,
    ))

    k, v = _kv(1, 2, 8, 8)
    cache.update(k, v, layer_idx=0)
    blocks = cache.layers[0].table.blocks
    assert all(blk.state == BlockState.SEALED for blk in blocks)
    assert cache.memory_report()["n_pending_quant_blocks"] == 2

    attn = torch.zeros(1, 1, 1, 8)
    attn[..., 0:4] = 0.1
    attn[..., 4:8] = 1.0
    cache.record_attention(0, attn)

    assert blocks[0].state == BlockState.COMPRESSED
    assert blocks[1].state == BlockState.SEALED
    assert cache.memory_report()["n_pending_quant_blocks"] == 1

    k1, v1 = _kv(1, 2, 1, 8)
    cache.update(k1, v1, layer_idx=0)
    assert all(blk.state == BlockState.COMPRESSED for blk in blocks[:2])
    assert cache.memory_report()["n_pending_quant_blocks"] == 0
    print("ok: test_budgeted_attention_score_compresses_ready_pages_by_budget")


def test_record_attentions_accepts_hf_tuple_shapes():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=2,
        policy=DeferredPolicy(),
    ))
    k, v = _kv(1, 2, 8, 8)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)

    layer0 = torch.zeros(1, 1, 1, 8)
    layer1 = torch.zeros(1, 1, 1, 8)
    layer0[..., :4] = 1.0
    layer1[..., 4:] = 1.0
    cache.record_attentions((layer0, layer1))

    assert cache.layers[0].table.blocks[0].page_meta["attention_score"] == 4.0
    assert cache.layers[1].table.blocks[1].page_meta["attention_score"] == 4.0

    cache.record_attentions(((layer0, layer1),))
    assert cache.layers[0].table.blocks[0].page_meta["attention_score"] == 8.0
    assert cache.layers[1].table.blocks[1].page_meta["attention_score"] == 8.0
    print("ok: test_record_attentions_accepts_hf_tuple_shapes")


def test_block_kv_cache_state_dict_roundtrip_and_continue():
    cfg = BlockCacheConfig(
        block_size=4,
        key_bits=4,
        value_bits=4,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
    )
    cache = BlockKVCache(cfg)
    k, v = _kv(1, 2, 10, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)
    before = cache.memory_report()

    restored = BlockKVCache(cfg)
    restored.load_state_dict(cache.state_dict())
    restored_k, restored_v = restored.layers[0]._materialize(dtype=torch.float16)

    assert restored.seen_tokens == cache.seen_tokens
    assert restored.memory_report() == before
    assert len(restored.layers[0]._tq_compressed_runs) == len(
        cache.layers[0]._tq_compressed_runs
    )
    assert torch.allclose(restored_k, full_k, atol=0, rtol=0)
    assert torch.allclose(restored_v, full_v, atol=0, rtol=0)

    k1, v1 = _kv(1, 2, 1, 8)
    next_k, next_v = restored.update(k1, v1, layer_idx=0)
    assert next_k.shape == (1, 2, 11, 8)
    assert next_v.shape == (1, 2, 11, 8)
    assert restored.get_seq_length(0) == 11
    print("ok: test_block_kv_cache_state_dict_roundtrip_and_continue")


def test_block_kv_cache_state_dict_skvq_roundtrip():
    cfg = BlockCacheConfig(
        block_size=4,
        key_bits=2,
        value_bits=1.5,
        policy=TokenBlockPolicy(),
        quant_backend="skvq",
        group_size=8,
        clipping=1.0,
    )
    cache = BlockKVCache(cfg)
    k, v = _kv(1, 2, 8, 8)
    full_k, full_v = cache.update(k, v, layer_idx=0)

    restored = BlockKVCache(cfg)
    restored.load_state_dict(cache.state_dict())
    restored_k, restored_v = restored.layers[0]._materialize(dtype=torch.float16)

    assert restored.memory_report() == cache.memory_report()
    assert torch.allclose(restored_k, full_k, atol=0, rtol=0)
    assert torch.allclose(restored_v, full_v, atol=0, rtol=0)
    print("ok: test_block_kv_cache_state_dict_skvq_roundtrip")


def test_decompressed_block_cache_lru_and_invalidation():
    cache = BlockKVCache(BlockCacheConfig(
        block_size=4,
        key_bits=4,
        value_bits=4,
        policy=TokenBlockPolicy(),
        quant_backend="turboquant",
        max_cached_decompressed_blocks=1,
    ))
    k, v = _kv(2, 2, 12, 8)
    cache.update(k, v, layer_idx=0)
    layer = cache.layers[0]

    assert len(layer._decompressed_cache) == 1
    layer._materialize(dtype=torch.float16)
    assert len(layer._decompressed_cache) == 1

    layer.reorder_cache(torch.tensor([1, 0]))
    assert len(layer._decompressed_cache) == 0
    layer._materialize(dtype=torch.float16)
    assert len(layer._decompressed_cache) == 1
    layer.crop(8)
    assert len(layer._decompressed_cache) == 0
    print("ok: test_decompressed_block_cache_lru_and_invalidation")


def main():
    test_block_table_splits_into_blocks()
    test_token_block_policy_compresses_all_sealed()
    test_window_policy_keeps_recent_fp16()
    test_hybrid_policy_sink_and_window()
    test_per_vector_compress_decompress_roundtrip()
    test_per_block_compress_decompress_roundtrip()
    test_norm_importance_score_many_matches_single_block_scores()
    test_top_ratio_allocator_run_aware_selects_contiguous_segment()
    test_block_kv_cache_update_returns_full_history()
    test_incremental_materialize_matches_legacy_path()
    test_turboquant_batched_compression_matches_single_page_path()
    test_turboquant_batched_materialize_matches_blockwise_path()
    test_live_fp16_blocks_are_compacted_after_prefill()
    test_block_table_total_len_tracks_crop_reset_and_restore()
    test_block_kv_cache_window_policy_memory_drops()
    test_block_kv_cache_per_layer_independence()
    test_reorder_cache_permutes_batch()
    test_skvq_page_compressor_roundtrip()
    test_skvq_page_compressor_reorder_roundtrip()
    test_skvq_page_compressor_asymmetric_group_size()
    test_block_kv_cache_skvq_mixed_precision_pages()
    test_block_kv_cache_skvq_reorder_metadata()
    test_block_kv_cache_turboquant_reorder_metadata()
    test_custom_page_backend_registry()
    test_shared_method_cache_factory()
    test_paper_pure_mix_protection_defaults_are_latency_safe()
    test_block_kv_cache_protected_layers_override_bits()
    test_attention_score_importance_drives_mixed_precision()
    test_attention_score_deferred_pages_compress_after_recording()
    test_budgeted_quant_cursor_limits_compression_per_update()
    test_budgeted_attention_score_compresses_ready_pages_by_budget()
    test_record_attentions_accepts_hf_tuple_shapes()
    test_block_kv_cache_state_dict_roundtrip_and_continue()
    test_block_kv_cache_state_dict_skvq_roundtrip()
    test_decompressed_block_cache_lru_and_invalidation()
    print("\nAll block_cache tests passed.")


if __name__ == "__main__":
    main()
