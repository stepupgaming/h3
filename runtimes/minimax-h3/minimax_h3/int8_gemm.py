"""Fused int8-weight GEMM helpers for deferred ConvRot linears.

Production default remains ephemeral cast+matmul in ``blocks.Linear``.
This module holds:

* ``weight_int8_mm`` — Triton kernel: bf16/fp16 x @ int8 W^T with per-out scale
  (no full bf16 weight materialization). Used only when faster than cast.
* micro helpers shared by the bench script.

Numerics match cast within bf16 rounding of the mixed-precision accumulate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

_TRITON_OK: Optional[bool] = None
_KERNEL = None


def triton_available() -> bool:
    global _TRITON_OK, _KERNEL
    if _TRITON_OK is not None:
        return _TRITON_OK
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _int8w_kernel(
            x_ptr,
            q_ptr,
            s_ptr,
            y_ptr,
            M,
            N,
            K,
            stride_xm,
            stride_xk,
            stride_qn,
            stride_qk,
            stride_ym,
            stride_yn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_m = offs_m < M
                mask_n = offs_n < N
                mask_k = k < K
                # x tile [BM, BK]
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                # q is [N, K]; load as [BK, BN] with q[n, k]
                q = tl.load(
                    q_ptr + offs_n[None, :] * stride_qn + k[:, None] * stride_qk,
                    mask=mask_n[None, :] & mask_k[:, None],
                    other=0.0,
                )
                acc += tl.dot(x.to(tl.float32), q.to(tl.float32))
            s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc = acc * s[None, :]
            y = acc.to(tl.bfloat16)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                y,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )

        _KERNEL = _int8w_kernel
        _TRITON_OK = True
    except Exception:
        _TRITON_OK = False
        _KERNEL = None
    return _TRITON_OK


def weight_int8_mm(
    x: Tensor,
    q: Tensor,
    scale: Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> Tensor:
    """y = (x @ q.to(x.dtype).T) * scale  without materializing q.to(dtype).

    x: [M, K] bf16/fp16, q: [N, K] int8, scale: [N] (any float).
    """
    if not triton_available():
        raise RuntimeError("triton unavailable")
    import triton

    assert x.is_cuda and q.is_cuda and scale.is_cuda
    assert x.dim() == 2 and q.dim() == 2
    m, k = x.shape
    n, k2 = q.shape
    if k != k2:
        raise ValueError(f"K mismatch x{k} vs q{k2}")
    if scale.numel() != n:
        raise ValueError(f"scale {tuple(scale.shape)} vs N={n}")
    x = x.contiguous()
    q = q.contiguous()
    scale = scale.contiguous()
    # Kernel writes bf16; cast if caller used fp16 stream.
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _KERNEL[grid](
        x,
        q,
        scale,
        y,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    if x.dtype != torch.bfloat16:
        y = y.to(dtype=x.dtype)
    return y


def cast_int8_mm(x: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    """Reference: ephemeral inflate + matmul + row scale."""
    wf = q.to(dtype=x.dtype)
    y = x.matmul(wf.t())
    del wf
    sc = scale if scale.dtype == x.dtype else scale.to(dtype=x.dtype)
    y.mul_(sc.unsqueeze(0))
    return y


def pick_fast_backend(
    x: Tensor,
    q: Tensor,
    scale: Tensor,
    *,
    iters: int = 8,
    warmup: int = 3,
) -> Tuple[str, float, float]:
    """Time cast vs triton on this shape; return (name, ms_cast, ms_other)."""
    import time

    def _time(fn) -> float:
        for _ in range(warmup):
            y = fn()
            del y
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn()
            del y
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms_cast = _time(lambda: cast_int8_mm(x, q, scale))
    if not triton_available():
        return "cast", ms_cast, float("inf")
    try:
        # compile / warmup
        _ = weight_int8_mm(x, q, scale)
        ms_tr = _time(lambda: weight_int8_mm(x, q, scale))
    except Exception:
        return "cast", ms_cast, float("inf")
    return ("triton" if ms_tr < ms_cast * 0.95 else "cast"), ms_cast, ms_tr
