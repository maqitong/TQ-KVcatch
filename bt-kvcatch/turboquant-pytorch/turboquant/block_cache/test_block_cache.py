"""Self-contained tests for the block-structured KV cache (no model needed).

Run with:
    python -m turboquant.block_cache.test_block_cache
"""

from __future__ import annotations

import torch

from turboquant.block_cache import (
    BlockCacheConfig,
    BlockKVCache,
    BlockState,
    BlockTable,
    HybridPolicy,
    TokenBlockPolicy,
    WindowBlockPolicy,
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


def main():
    test_block_table_splits_into_blocks()
    test_token_block_policy_compresses_all_sealed()
    test_window_policy_keeps_recent_fp16()
    test_hybrid_policy_sink_and_window()
    test_per_vector_compress_decompress_roundtrip()
    test_per_block_compress_decompress_roundtrip()
    test_block_kv_cache_update_returns_full_history()
    test_block_kv_cache_window_policy_memory_drops()
    test_block_kv_cache_per_layer_independence()
    test_reorder_cache_permutes_batch()
    print("\nAll block_cache tests passed.")


if __name__ == "__main__":
    main()
