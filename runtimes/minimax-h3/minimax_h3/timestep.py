"""Timestep domain math and the fp32 time embedder. A 1:1 Python port of
``src/timestep.rs`` (which ports ``time_shift_sigma``, ``time_shift_slope`` and
``TimeEmbedder`` from ComfyUI ``comfy/ldm/minimax/model.py``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other
    shift."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: float, from_shift: float, to_shift: float) -> float:
    """d(sigma_to)/d(sigma_from) at the same base-grid point. Scaling a
    stream's returned velocity by this slope makes the flat ODE that any
    sampler integrates on the from-schedule equal to that stream's true ODE."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


class TimeEmbedder:
    """Standard sinusoidal time embedder, fp32 throughout (cos before sin)."""

    def __init__(
        self,
        src,
        freq_dim: int,
        hidden: int,
        out: int,
    ):
        from .blocks import linear

        self.freq_dim = freq_dim
        self.proj_in = linear(src, "time_embedder.proj_in", hidden, freq_dim, True, torch.float32)
        self.proj_out = linear(src, "time_embedder.proj_out", out, hidden, True, torch.float32)

    def to(self, device=None, dtype=None):
        self.proj_in.to(device=device, dtype=dtype)
        self.proj_out.to(device=device, dtype=dtype)
        return self

    def forward(self, t: Tensor) -> Tensor:
        """t: [M] fp32 in [0, 1] -> [M, out] fp32"""
        half = self.freq_dim // 2
        freqs = torch.tensor(
            [math.exp(-math.log(10000.0) * i / half) for i in range(half)],
            dtype=t.dtype,
            device=t.device,
        ).unsqueeze(0)  # [1, half]
        args = t.unsqueeze(1).expand(t.shape[0], half) * freqs.expand(t.shape[0], half)
        emb = torch.cat([args.cos(), args.sin()], dim=1)  # [M, freq_dim]
        x = self.proj_in.forward(emb)
        x = torch.nn.functional.silu(x)
        return self.proj_out.forward(x)
