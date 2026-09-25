"""3D (t, h, w) RoPE with the split-half rotation used by the reference
("kitchen" fused rms+rope kernels). A 1:1 Python port of ``src/rotary.rs``
(which ports ``rope_freqs`` / ``rope_rotation_table`` /
``apply_rope_split_half`` from ComfyUI ``comfy/ldm/minimax/model.py``).
"""

from __future__ import annotations

import torch
from torch import Tensor


def rope_freqs(position_ids: Tensor, inv_freq: Tensor) -> Tensor:
    """position_ids [S, 3] fp32 x inv_freq [rope_inv_freq_len] -> [S, 6k] fp32,
    where the halves are duplicated so the table only needs half of them."""
    s = position_ids.shape[0]
    k = inv_freq.shape[0]
    pos = position_ids.unsqueeze(2)  # [S, 3, 1]
    inv = inv_freq.unsqueeze(0).unsqueeze(0)  # [1, 1, k]
    per_axis = pos * inv  # [S, 3, k]
    t = per_axis[:, 0, :].reshape(s, k)
    h = per_axis[:, 1, :].reshape(s, k)
    w = per_axis[:, 2, :].reshape(s, k)
    half = torch.cat([t, h, w], dim=1)  # [S, 3k]
    return torch.cat([half, half], dim=1)  # [S, 6k]


def rope_cos_sin(angles: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """[S, 6k] pair angles -> (cos, sin) of shape [S, 3k]. The reference builds
    a [1, S, 1, half, 2, 2] table from the first half of the angles; we extract
    cos/sin directly and apply the 2x2 [[c, -s], [s, c]] rotation in
    `apply_rope_split_half`."""
    half = angles.shape[1] // 2
    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope_split_half(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Split-half rope: the first `rot = 2P` dims of q/k [S, H, D] are paired
    as (i, i + P) and rotated by the P angles, so
      out[i]     = c_i * x[i]     - s_i * x[i+P]
      out[i + P] = s_i * x[i]     + c_i * x[i+P]
    The remaining dims pass through untouched."""
    s, heads, d = q.shape
    p = cos.shape[1]
    rot = 2 * p
    rest = d - rot
    c = cos.unsqueeze(1).expand(s, heads, p)
    sn = sin.unsqueeze(1).expand(s, heads, p)

    def rotate(x: Tensor) -> Tensor:
        a = x[:, :, :p]
        b = x[:, :, p : 2 * p]
        out_a = a * c - b * sn
        out_b = a * sn + b * c
        return torch.cat([out_a, out_b], dim=2)

    qrot = rotate(q)
    krot = rotate(k)
    q = torch.cat([qrot, q[:, :, rot:]], dim=2) if rest > 0 else qrot
    k = torch.cat([krot, k[:, :, rot:]], dim=2) if rest > 0 else krot
    return q, k


from typing import Tuple  # noqa: E402  (kept at the bottom to match the small module)
