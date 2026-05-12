from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


try:
    from turboquant.lloyd_max import LloydMaxCodebook
    from turboquant.turboquant import generate_rotation_matrix
except ImportError:
    _TQ_ROOT = Path(__file__).resolve().parents[1] / "bt-kvcatch" / "turboquant-pytorch"
    if _TQ_ROOT.exists():
        sys.path.insert(0, str(_TQ_ROOT))
    try:
        from turboquant.lloyd_max import LloydMaxCodebook
        from turboquant.turboquant import generate_rotation_matrix
    except ImportError:
        LloydMaxCodebook = None

        def generate_rotation_matrix(d: int, seed: int | None = None, device: str = "cpu") -> torch.Tensor:
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(seed)
            gaussian = torch.randn(d, d, generator=gen)
            q, r = torch.linalg.qr(gaussian)
            signs = torch.sign(torch.diag(r))
            signs[signs == 0] = 1.0
            return (q * signs.unsqueeze(0)).to(device)


def _normal_pdf(x: float) -> float:
    if math.isinf(x):
        return 0.0
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    if x == float("-inf"):
        return 0.0
    if x == float("inf"):
        return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _solve_gaussian_lloyd_max(dim: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    levels = 2 ** bits
    sigma = 1.0 / math.sqrt(dim)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / levels for i in range(levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(levels - 1)]
        edges = [float("-inf")] + boundaries + [float("inf")]
        new_centroids = []
        for i in range(levels):
            a = edges[i] / sigma if not math.isinf(edges[i]) else edges[i]
            b = edges[i + 1] / sigma if not math.isinf(edges[i + 1]) else edges[i + 1]
            denom = _normal_cdf(b) - _normal_cdf(a)
            if denom <= 1e-15:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(sigma * (_normal_pdf(a) - _normal_pdf(b)) / denom)

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(levels - 1)]
    return torch.tensor(centroids, dtype=torch.float32), torch.tensor(boundaries, dtype=torch.float32)


class TurboQuantBackend(nn.Module):
    """Fake-quant TurboQuant backend with the same tensor contract as SKVQ."""

    _codebook_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        head_dim: int,
        hidden: int,
        key_bits: int,
        value_bits: int,
        layer_idx: int,
        num_layers: int,
        protected_layers: int = 0,
        protected_bits: int = 8,
        group_size: int | None = None,
        use_reorder: bool = False,
        seed_base: int = 42,
        clipping: list[float] | float | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.hidden = hidden
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.protected_layers = protected_layers
        self.protected_bits = protected_bits
        self.group_size = group_size
        self.use_reorder = use_reorder
        self.seed_base = seed_base
        self.clipping = clipping
        self._param_cache: dict[
            tuple[str, int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def tag(self) -> str:
        protect = f"-protect{self.protected_layers}" if self.protected_layers else ""
        reorder = "-tqrod" if self.use_reorder else "-tq"
        return f"{reorder}{protect}"

    def _bits(self, ttype: Literal["k", "v"]) -> int:
        bits = self.key_bits if ttype == "k" else self.value_bits
        if bits != round(bits):
            raise ValueError("TurboQuant backend only supports integer bit-widths")

        is_protected = (
            self.protected_layers > 0
            and (
                self.layer_idx < self.protected_layers
                or self.layer_idx >= self.num_layers - self.protected_layers
            )
        )
        if is_protected:
            bits = self.protected_bits
        return min(int(bits), 8)

    def _clip_scale(self) -> float:
        if self.clipping is None:
            return 1.0
        if isinstance(self.clipping, (float, int)):
            return float(self.clipping)
        return float(self.clipping[self.layer_idx])

    @classmethod
    def _codebook(cls, dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, bits)
        if key not in cls._codebook_cache:
            if LloydMaxCodebook is None:
                cls._codebook_cache[key] = _solve_gaussian_lloyd_max(dim, bits)
            else:
                codebook = LloydMaxCodebook(dim, bits)
                cls._codebook_cache[key] = (codebook.centroids, codebook.boundaries)
        return cls._codebook_cache[key]

    def _params(
        self,
        ttype: Literal["k", "v"],
        group_idx: int,
        dim: int,
        bits: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (ttype, group_idx, dim, str(device), dtype)
        if cache_key in self._param_cache:
            return self._param_cache[cache_key]

        seed_offset = 0 if ttype == "k" else 500_000
        seed = self.seed_base + self.layer_idx * 1000 + seed_offset + group_idx
        rotation = generate_rotation_matrix(dim, seed=seed, device="cpu").to(device=device, dtype=dtype)
        centroids, boundaries = self._codebook(dim, bits)
        centroids = centroids.to(device=device, dtype=dtype)
        boundaries = boundaries.to(device=device, dtype=dtype)
        self._param_cache[cache_key] = (rotation, centroids, boundaries)
        return rotation, centroids, boundaries

    @torch.no_grad()
    def _quant_flat(
        self,
        flat: torch.Tensor,
        ttype: Literal["k", "v"],
        group_idx: int,
        bits: int,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if flat.numel() == 0 or bits == 16:
            return flat.to(out_dtype)

        dim = flat.shape[-1]
        rotation, centroids, boundaries = self._params(
            ttype, group_idx, dim, bits, flat.device, torch.float32
        )
        work = flat.float()
        norms = work.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        clip_scale = self._clip_scale()
        if clip_scale < 1.0:
            norm_cap = norms.max().mul(clip_scale).clamp(min=1e-8)
            recon_norms = norms.clamp(max=norm_cap)
        else:
            recon_norms = norms

        rotated = (work / norms) @ rotation.T
        indices = torch.bucketize(rotated, boundaries)
        rotated_hat = centroids[indices]
        out = (rotated_hat @ rotation) * recon_norms
        return out.to(out_dtype)

    def _natural_group_starts(self, device: torch.device) -> torch.Tensor:
        group_size = self.group_size or self.head_dim
        starts = list(range(0, self.hidden, group_size))
        if not starts or starts[-1] != self.hidden:
            starts.append(self.hidden)
        return torch.tensor(starts, dtype=torch.long, device=device)

    @torch.no_grad()
    def quant(
        self,
        ttype: Literal["k", "v"],
        tensor: torch.Tensor,
        layer_idx: int | None = None,
        reorder_idx: dict[str, torch.Tensor] | None = None,
        group_st_idx: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        if tensor is None:
            return None, None, None

        bits = self._bits(ttype)
        bs, num_heads, seqlen, head_dim = tensor.shape
        if seqlen == 0 or bits == 16:
            return tensor, None, None
        if num_heads * head_dim != self.hidden:
            raise ValueError(f"expected hidden={self.hidden}, got {num_heads * head_dim}")

        dtype = tensor.dtype

        if self.use_reorder:
            if reorder_idx is None or group_st_idx is None:
                raise ValueError("TurboQuant reorder mode requires SKVQ reorder metadata")
            flat_hidden = tensor.transpose(1, 2).reshape(bs, seqlen, self.hidden)
            idx = reorder_idx[ttype].long().to(tensor.device)
            work = flat_hidden[..., idx]
            gst = group_st_idx[ttype].long().to(tensor.device)
            out = torch.empty_like(work)
            for group_idx in range(gst.numel() - 1):
                start = int(gst[group_idx].item())
                end = int(gst[group_idx + 1].item())
                chunk = work[..., start:end].reshape(-1, end - start)
                out[..., start:end] = self._quant_flat(
                    chunk, ttype, group_idx, bits, dtype
                ).reshape(bs, seqlen, end - start)
            inv_idx = idx.argsort()
            out = out.gather(-1, inv_idx.expand_as(out))
            return out.reshape(bs, seqlen, num_heads, head_dim).transpose(1, 2).contiguous(), None, None

        if self.group_size is not None and self.group_size != head_dim:
            flat_hidden = tensor.transpose(1, 2).reshape(bs, seqlen, self.hidden)
            gst = self._natural_group_starts(tensor.device)
            out = torch.empty_like(flat_hidden)
            for group_idx in range(gst.numel() - 1):
                start = int(gst[group_idx].item())
                end = int(gst[group_idx + 1].item())
                chunk = flat_hidden[..., start:end].reshape(-1, end - start)
                out[..., start:end] = self._quant_flat(
                    chunk, ttype, group_idx, bits, dtype
                ).reshape(bs, seqlen, end - start)
            return out.reshape(bs, seqlen, num_heads, head_dim).transpose(1, 2).contiguous(), None, None

        flat = tensor.reshape(-1, head_dim)
        out = self._quant_flat(flat, ttype, 0, bits, dtype)
        return out.reshape_as(tensor).contiguous(), None, None
