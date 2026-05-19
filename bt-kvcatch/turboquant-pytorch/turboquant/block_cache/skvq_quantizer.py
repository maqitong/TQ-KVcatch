"""SKVQ-style page compressor for block-structured KV cache.

This module ports the parts of SKVQ that fit the current KVcatch page model:
group-wise min/max quantization, optional reorder metadata, clipping, and
3-level 1.5-bit quantization. It stores packed uint8 indices plus per-group
scale/zero-point tensors, then dequantizes back to dense K/V for HF attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _canonical_bits(bits: int | float) -> int | float:
    bits = float(bits)
    if bits == 1.5:
        return bits
    if bits != round(bits):
        raise ValueError(f"SKVQPageCompressor supports integer bits or 1.5, got {bits}")
    bits_int = int(bits)
    if bits_int <= 0 or bits_int > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits_int}")
    return bits_int


def _levels(bits: int | float) -> int:
    bits = _canonical_bits(bits)
    return 3 if bits == 1.5 else 2 ** int(bits)


def _container_bits(bits: int | float) -> int:
    return 2 if _canonical_bits(bits) == 1.5 else int(bits)


def _natural_group_starts(hidden: int, group_size: int, device: torch.device) -> torch.Tensor:
    starts = list(range(0, hidden, group_size))
    if not starts or starts[-1] != hidden:
        starts.append(hidden)
    return torch.tensor(starts, dtype=torch.long, device=device)


def _pack_indices(indices: torch.Tensor, bits: int | float) -> tuple[torch.Tensor, int]:
    """Pack uint8 indices along the last dimension."""
    cbits = _container_bits(bits)
    values_per_byte = 8 // cbits
    width = indices.shape[-1]
    pad = (values_per_byte - width % values_per_byte) % values_per_byte
    work = indices.long()
    if pad:
        work = F.pad(work, (0, pad))
    groups = work.shape[-1] // values_per_byte
    shifts = torch.tensor(
        [cbits * i for i in range(values_per_byte - 1, -1, -1)],
        dtype=torch.long,
        device=work.device,
    )
    packed = (work.reshape(*work.shape[:-1], groups, values_per_byte) << shifts).sum(-1)
    return packed.to(torch.uint8), pad


def _unpack_indices(
    packed: torch.Tensor, bits: int | float, width: int, pad: int
) -> torch.Tensor:
    cbits = _container_bits(bits)
    values_per_byte = 8 // cbits
    mask = (1 << cbits) - 1
    shifts = torch.tensor(
        [cbits * i for i in range(values_per_byte - 1, -1, -1)],
        dtype=torch.long,
        device=packed.device,
    )
    unpacked = ((packed.long().unsqueeze(-1) >> shifts) & mask).reshape(
        *packed.shape[:-1], -1
    )
    if pad:
        unpacked = unpacked[..., :width]
    return unpacked


class SKVQPageCompressor:
    """Natural-group SKVQ compressor for one transformer cache layer."""

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        group_size: int = 128,
        key_group_size: int | None = None,
        value_group_size: int | None = None,
        clipping: float = 0.92,
        reorder_idx: Optional[dict[str, torch.Tensor]] = None,
        group_st_idx: Optional[dict[str, torch.Tensor]] = None,
    ):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if key_group_size is not None and key_group_size <= 0:
            raise ValueError("key_group_size must be positive")
        if value_group_size is not None and value_group_size <= 0:
            raise ValueError("value_group_size must be positive")
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.hidden = head_dim * n_kv_heads
        self.group_size = group_size
        self.key_group_size = key_group_size or group_size
        self.value_group_size = value_group_size or group_size
        self.clipping = float(clipping)
        self.reorder_idx = reorder_idx
        self.group_st_idx = group_st_idx

    def _group_size(self, ttype: str) -> int:
        return self.key_group_size if ttype == "k" else self.value_group_size

    def _group_starts(self, ttype: str, device: torch.device) -> torch.Tensor:
        if self.group_st_idx is not None and ttype in self.group_st_idx:
            return self.group_st_idx[ttype].long().to(device)
        return _natural_group_starts(self.hidden, self._group_size(ttype), device)

    def _reorder(self, states: torch.Tensor, ttype: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.reorder_idx is None or ttype not in self.reorder_idx:
            return states, None
        idx = self.reorder_idx[ttype].long().to(states.device)
        if idx.numel() != self.hidden:
            raise ValueError(f"reorder index length {idx.numel()} != hidden {self.hidden}")
        return states[..., idx], idx

    @torch.no_grad()
    def compress(
        self,
        states: torch.Tensor,
        *,
        bits: int | float,
        ttype: str,
        layer_idx: int,
    ) -> dict:
        """Compress a page-shaped tensor `(B, H, S, D)`."""
        bits = _canonical_bits(bits)
        if ttype not in ("k", "v"):
            raise ValueError(f"ttype must be 'k' or 'v', got {ttype}")

        B, H, S, D = states.shape
        if H != self.n_kv_heads or D != self.head_dim:
            raise ValueError(
                f"expected (H={self.n_kv_heads}, D={self.head_dim}), got (H={H}, D={D})"
            )
        if S == 0:
            return {
                "backend": "skvq",
                "bits": bits,
                "shape": (B, H, S, D),
                "qdata": torch.empty(B, S, 0, dtype=torch.uint8, device=states.device),
                "scale": torch.empty(B, S, 0, dtype=states.dtype, device=states.device),
                "zero": torch.empty(B, S, 0, dtype=states.dtype, device=states.device),
                "group_widths": [],
                "group_pads": [],
                "packed_widths": [],
                "reordered": False,
            }

        work = states.transpose(1, 2).reshape(B, S, self.hidden)
        work, reorder_idx = self._reorder(work, ttype)
        starts = self._group_starts(ttype, states.device)
        n_levels = _levels(bits)
        max_int = n_levels - 1

        packed_groups: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        zeros: list[torch.Tensor] = []
        group_widths: list[int] = []
        group_pads: list[int] = []
        packed_widths: list[int] = []

        for group_idx in range(starts.numel() - 1):
            start = int(starts[group_idx].item())
            end = int(starts[group_idx + 1].item())
            chunk = work[..., start:end].float()
            gmin, gmax = chunk.aminmax(dim=-1, keepdim=True)
            if self.clipping < 1.0:
                gmin = gmin * self.clipping
                gmax = gmax * self.clipping
            scale = ((gmax - gmin) / max_int).clamp(min=1e-5)
            q = ((chunk - gmin) / scale).round().clamp(0, max_int).to(torch.uint8)
            packed, pad = _pack_indices(q, bits)

            packed_groups.append(packed)
            scales.append(scale.to(states.dtype))
            zeros.append(gmin.to(states.dtype))
            group_widths.append(end - start)
            group_pads.append(pad)
            packed_widths.append(packed.shape[-1])

        qdata = torch.cat(packed_groups, dim=-1)
        scale = torch.cat(scales, dim=-1)
        zero = torch.cat(zeros, dim=-1)

        return {
            "backend": "skvq",
            "bits": bits,
            "container_bits": _container_bits(bits),
            "shape": (B, H, S, D),
            "qdata": qdata,
            "scale": scale,
            "zero": zero,
            "group_widths": group_widths,
            "group_pads": group_pads,
            "packed_widths": packed_widths,
            "reordered": reorder_idx is not None,
            "layer_idx": layer_idx,
            "ttype": ttype,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        B, H, S, D = compressed["shape"]
        if S == 0:
            return torch.empty(B, H, S, D, device=compressed["qdata"].device)

        bits = compressed["bits"]
        qdata = compressed["qdata"]
        scale = compressed["scale"].float()
        zero = compressed["zero"].float()
        group_widths = compressed["group_widths"]
        group_pads = compressed["group_pads"]
        packed_widths = compressed["packed_widths"]

        parts: list[torch.Tensor] = []
        q_cursor = 0
        for group_idx, width in enumerate(group_widths):
            packed_width = packed_widths[group_idx]
            packed = qdata[..., q_cursor : q_cursor + packed_width]
            q_cursor += packed_width
            q = _unpack_indices(packed, bits, width, group_pads[group_idx]).float()
            group_scale = scale[..., group_idx : group_idx + 1]
            group_zero = zero[..., group_idx : group_idx + 1]
            parts.append(q.mul(group_scale).add(group_zero))

        flat = torch.cat(parts, dim=-1)
        reorder_idx = None
        if compressed.get("reordered"):
            ttype = compressed.get("ttype")
            if self.reorder_idx is not None and ttype in self.reorder_idx:
                reorder_idx = self.reorder_idx[ttype].long().to(flat.device)
        if reorder_idx is not None:
            inv_idx = reorder_idx.long().argsort()
            flat = flat.gather(-1, inv_idx.expand_as(flat))

        dtype = compressed["scale"].dtype
        return flat.reshape(B, S, H, D).transpose(1, 2).contiguous().to(dtype)

    def memory_bytes(self, B: int, H: int, S: int, bits: int | float, ttype: str = "k") -> dict:
        bits = _canonical_bits(bits)
        cbits = _container_bits(bits)
        values_per_byte = 8 // cbits
        starts = list(range(0, self.hidden, self._group_size(ttype)))
        if not starts or starts[-1] != self.hidden:
            starts.append(self.hidden)
        q_bytes_per_token = sum(
            (starts[i + 1] - starts[i] + values_per_byte - 1) // values_per_byte
            for i in range(len(starts) - 1)
        )
        n_groups = len(starts) - 1
        q_bytes = B * S * q_bytes_per_token
        param_bytes = B * S * n_groups * 2 * 2
        compressed = q_bytes + param_bytes
        fp16 = B * H * S * self.head_dim * 2
        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / compressed if compressed > 0 else 0.0,
        }
