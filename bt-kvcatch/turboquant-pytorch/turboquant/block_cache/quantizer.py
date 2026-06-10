"""Per-block TurboQuant compressor wrapper.

Adds a `granularity` knob on top of the existing `MSECompressor` so block
contents can be quantized either with one norm per token (default, matches
TurboQuant V3) or with a single norm shared across the whole block (SKVQ-
style group-level quantization).

  granularity='per-vector'  -> each of the block_size vectors keeps its own
                               FP16 L2 norm. Best reconstruction.
                               Norm overhead per (B, H) block: block_size * 2 B
  granularity='per-block'   -> one FP16 mean-L2-norm scalar per (B, H) block.
                               Norm overhead per (B, H) block: 2 B
"""

from __future__ import annotations

from typing import Literal
import math

import torch
import torch.nn.functional as F

from ..compressors_v3 import MSECompressor


def _pack_uint8(indices: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    """Pack a (..., D) tensor of uint8 codebook indices into (..., n_groups)
    of uint8 where each byte stores `8 // bits` adjacent indices.

    Returns the packed tensor and the right-padding length on the last dim.
    """
    indices_per_byte = 8 // bits
    D = indices.shape[-1]
    pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
    flat = indices.long()
    if pad:
        flat = F.pad(flat, (0, pad))
    n_groups = flat.shape[-1] // indices_per_byte
    powers = torch.tensor(
        [2 ** (bits * i) for i in range(indices_per_byte - 1, -1, -1)],
        dtype=torch.long,
        device=flat.device,
    )
    grouped = flat.reshape(*flat.shape[:-1], n_groups, indices_per_byte)
    packed = (grouped * powers).sum(-1).to(torch.uint8)
    return packed, pad


def _unpack_uint8(
    packed: torch.Tensor, bits: int, D: int, pad: int
) -> torch.Tensor:
    """Inverse of `_pack_uint8`. packed shape (..., n_groups) -> (..., D)."""
    indices_per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = torch.tensor(
        [bits * i for i in range(indices_per_byte - 1, -1, -1)],
        dtype=torch.long,
        device=packed.device,
    )
    expanded = (packed.long().unsqueeze(-1) >> shifts) & mask
    flat = expanded.reshape(*packed.shape[:-1], -1)
    if pad:
        flat = flat[..., :D]
    return flat


class BlockMSECompressor:
    """MSECompressor wrapped with a configurable quantization granularity.

    Both modes share the same random rotation matrix and Lloyd-Max codebook,
    so a per-vector compressor and a per-block compressor with the same
    `(head_dim, bits, seed)` are interchangeable for diagnostic comparison.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        granularity: Literal["per-vector", "per-block"] = "per-vector",
        device: str = "cpu",
    ):
        if granularity not in ("per-vector", "per-block"):
            raise ValueError(f"unknown granularity: {granularity}")
        self.granularity = granularity
        self._inner = MSECompressor(head_dim, bits, seed=seed, device=device)

    @property
    def head_dim(self) -> int:
        return self._inner.head_dim

    @property
    def bits(self) -> int:
        return self._inner.bits

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """Compress (B, H, S, D)."""
        if self.granularity == "per-vector":
            d = self._inner.compress(states)
            d["granularity"] = "per-vector"
            return d

        # per-block: one mean-L2-norm scalar per (B, H) group
        B, H, S, D = states.shape
        flat = states.float().reshape(B * H, S, D)
        per_token_norm = torch.norm(flat, dim=-1)  # (BH, S)
        block_norm = per_token_norm.mean(dim=1, keepdim=True).clamp(min=1e-8)  # (BH, 1)
        scaled = flat / block_norm.unsqueeze(-1)  # (BH, S, D)

        # rotate + per-coordinate Lloyd-Max
        rotated = scaled.reshape(-1, D) @ self._inner.Pi.T  # (B*H*S, D)
        indices = self._inner.quantize_indices(rotated)  # (N, D)
        packed, pad = _pack_uint8(indices, self._inner.bits)
        n_groups = packed.shape[-1]

        return {
            "idx_bytes": packed.reshape(B, H, S, n_groups),
            "block_norm": block_norm.reshape(B, H, 1).to(torch.float16),
            "shape": (B, H, S, D),
            "idx_pad": pad,
            "granularity": "per-block",
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        if compressed.get("granularity", "per-vector") == "per-vector":
            return self._inner.decompress(compressed)

        B, H, S, D = compressed["shape"]
        idx_bytes = compressed["idx_bytes"].reshape(B * H * S, -1)
        indices = _unpack_uint8(
            idx_bytes, self._inner.bits, D, compressed["idx_pad"]
        )  # (B*H*S, D)
        rotated_hat = self._inner.centroids[indices]  # (N, D)
        scaled_hat = rotated_hat @ self._inner.Pi  # (N, D)
        block_norm = (
            compressed["block_norm"].float().reshape(B * H, 1, 1)
        )  # (BH, 1, 1)
        out = scaled_hat.reshape(B * H, S, D) * block_norm
        return out.reshape(B, H, S, D)

    def memory_bytes(self, B: int, H: int, S: int) -> dict:
        """Storage cost in bytes for a single block-shaped tensor."""
        D = self.head_dim
        N = B * H * S
        indices_per_byte = 8 // self.bits
        idx_bytes = N * math.ceil(D / indices_per_byte)
        if self.granularity == "per-vector":
            norm_bytes = N * 2  # one FP16 per token
        else:
            norm_bytes = B * H * 2  # one FP16 per (B, H) block
        compressed = idx_bytes + norm_bytes
        fp16 = N * D * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0.0,
        }
