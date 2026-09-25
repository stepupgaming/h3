"""Multi-head attention with fused QKV projection, per-head RMSNorm and
split-half RoPE. A 1:1 Python port of ``src/attention.rs`` (which ports
``Attention`` from ComfyUI ``comfy/ldm/minimax/model.py``).

Kernel modes, behind the same dispatch seam as the Candle port:

- ``eager``: materializes the full [H, S, S] scores matrix.
- ``flash``: portable flash-style chunked attention with online softmax
  (O(S) tiles). Fallback when fused FA2 is unavailable — not Dao flash-attn.
- ``fa2``: real FlashAttention-2 via the ``flash-attn`` package (HQ path).
  Windows owner box: community cu130 wheel (see pyproject). Layout convert
  HND ``[H,S,D]`` → FA ``[B,S,H,D]``.
- ``auto``: eager below ``AUTO_THRESHOLD`` rows, portable flash at or above it.
- ``sdpa``: torch's native ``scaled_dot_product_attention`` (may OOM / lack FA).
- ``sage``: SageAttention (default production / fast path). Sub-backends via
  ``set_sage_backend`` / CLI ``--sage-backend``. Approximate on fp8 backends.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .blocks import Linear, RmsNorm, linear, rms_norm
from .rotary import apply_rope_split_half

AUTO_THRESHOLD = 4096
FLASH_BLOCK = 256

# Dense SageAttention sub-backends only (not sparse).
# ``auto`` = package dispatcher (SA2 routes sm120 → fp8 CUDA).
SAGE_BACKENDS = (
    "auto",
    "fp16_cuda",
    "fp16_triton",
    "fp8",
    "fp8pp",
    "sa1",
)

# CLI / SampleRequest aliases → canonical AttnImpl value.
ATTN_CHOICES = (
    "eager",
    "flash",
    "auto",
    "sdpa",
    "sage",
    "fa2",
    "flash-attn",
    "flash_attn",
)

_SAGE_BACKEND = "auto"
_SAGE_RESOLVED: Optional[tuple[str, Callable]] = None
_FA2_FN: Optional[Callable] = None


class AttnImpl:
    EAGER = "eager"
    FLASH = "flash"  # portable chunked online-softmax (not Dao FA)
    FA2 = "fa2"  # real flash-attn package (HQ)
    AUTO = "auto"
    SDPA = "sdpa"
    SAGE = "sage"  # SageAttention (default H3 path; see --sage-backend)

    @staticmethod
    def normalize(name: str) -> str:
        key = (name or "").strip().lower().replace("-", "_")
        if key in ("fa2", "flash_attn", "flashattn"):
            return AttnImpl.FA2
        if key == "flash":
            return AttnImpl.FLASH
        if key in ("eager", "auto", "sdpa", "sage"):
            return key
        raise ValueError(
            f"unknown attn impl {name!r}; choose from {ATTN_CHOICES}"
        )


def set_sage_backend(name: str) -> None:
    """Select dense SageAttention kernel. Process-wide; call before sample."""
    global _SAGE_BACKEND, _SAGE_RESOLVED
    key = (name or "auto").strip().lower()
    if key not in SAGE_BACKENDS:
        raise ValueError(f"unknown sage backend {name!r}; choose from {SAGE_BACKENDS}")
    _SAGE_BACKEND = key
    _SAGE_RESOLVED = None


def get_sage_backend() -> str:
    return _SAGE_BACKEND


def _softmax(t: Tensor, dim: int) -> Tensor:
    max_ = t.max(dim=dim, keepdim=True).values
    e = torch.exp(t - max_)
    return e / e.sum(dim=dim, keepdim=True)


class Attention:
    def __init__(
        self,
        src,
        prefix: str,
        hidden: int,
        heads: int,
        head_dim: int,
        eps: float,
        dtype: torch.dtype,
        attn_impl: str,
    ):
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = linear(src, f"{prefix}.qkv_proj", inner * 3, hidden, False, dtype)
        self.q_norm = rms_norm(src, f"{prefix}.q_norm", head_dim, eps, dtype)
        self.k_norm = rms_norm(src, f"{prefix}.k_norm", head_dim, eps, dtype)
        self.out_proj = linear(src, f"{prefix}.out_proj", hidden, inner, False, dtype)
        self.attn_impl = attn_impl

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.qkv_proj.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.q_norm.to(device=device, dtype=dtype)
        self.k_norm.to(device=device, dtype=dtype)
        self.out_proj.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def pin_host(self) -> None:
        self.qkv_proj.pin_host()
        self.out_proj.pin_host()

    def adopt_host_pageable(self) -> None:
        self.qkv_proj.adopt_host_pageable()
        self.out_proj.adopt_host_pageable()

    def forward(self, x: Tensor, rope_cos: Tensor | None = None, rope_sin: Tensor | None = None) -> Tensor:
        s = x.shape[0]
        inner = self.heads * self.head_dim
        qkv = self.qkv_proj.forward(x)  # [S, 3*inner]
        q = qkv[:, :inner]
        k = qkv[:, inner : 2 * inner]
        v = qkv[:, 2 * inner :]
        q = q.reshape(s, self.heads, self.head_dim)
        k = k.reshape(s, self.heads, self.head_dim)
        v = v.reshape(s, self.heads, self.head_dim)

        q = rms_norm_headwise(q, self.q_norm.w, self.q_norm.eps)
        k = rms_norm_headwise(k, self.k_norm.w, self.k_norm.eps)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope_split_half(q, k, rope_cos, rope_sin)

        q = q.transpose(0, 1)  # [H, S, D]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        seq = q.shape[1]
        impl = AttnImpl.normalize(self.attn_impl) if self.attn_impl else AttnImpl.SAGE
        if impl == AttnImpl.AUTO:
            impl = AttnImpl.EAGER if seq <= AUTO_THRESHOLD else AttnImpl.FLASH
        if impl == AttnImpl.EAGER:
            out = _eager(q, k, v)
        elif impl == AttnImpl.SDPA:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif impl == AttnImpl.SAGE:
            out = _sage(q, k, v, seq)
        elif impl == AttnImpl.FA2:
            out = _fa2(q, k, v)
        elif impl == AttnImpl.FLASH:
            out = _flash(q, k, v, FLASH_BLOCK)
        else:
            raise ValueError(f"unhandled attn impl {impl!r}")
        out = out.transpose(0, 1).reshape(s, inner)
        return self.out_proj.forward(out)


def _resolve_sage_callable(backend: str) -> Callable:
    """Return fn(q,k,v) for [B,H,S,D] HND tensors. Raises if unavailable."""
    try:
        import sageattention as sa
    except ImportError as e:
        raise ImportError(
            "sageattention is not installed; use --attn sdpa/eager/flash or install sageattention"
        ) from e

    def _require(name: str):
        fn = getattr(sa, name, None)
        if fn is None:
            raise ImportError(
                f"sage backend needs sageattention.{name}, but this install only exports "
                f"{[n for n in dir(sa) if not n.startswith('_')]}. "
                "On Windows install the SA2 CUDA wheel (see pyproject sageattention url)."
            )
        return fn

    if backend == "auto":
        fn = _require("sageattn")

        def call(q, k, v):
            return fn(q, k, v, tensor_layout="HND")

        return call

    if backend == "sa1":
        # Prefer Triton SA1-style entry if present; else package sageattn.
        fn = getattr(sa, "sageattn_qk_int8_pv_fp16_triton", None) or _require("sageattn")

        def call(q, k, v):
            return fn(q, k, v, tensor_layout="HND")

        return call

    if backend == "fp16_triton":
        fn = _require("sageattn_qk_int8_pv_fp16_triton")

        def call(q, k, v):
            return fn(q, k, v, tensor_layout="HND")

        return call

    if backend == "fp16_cuda":
        fn = _require("sageattn_qk_int8_pv_fp16_cuda")
        # sm80 CUDA ext on sm120 can run "fast" with huge numeric error — refuse.
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) >= (12, 0):
                raise RuntimeError(
                    "sage-backend fp16_cuda is unsafe on sm120+ (owner-box A/B: "
                    "max|d|~6 vs flash while looking fast). Use auto/fp8/fp8pp."
                )

        def call(q, k, v):
            return fn(q, k, v, tensor_layout="HND", pv_accum_dtype="fp32")

        return call

    if backend == "fp8":
        fn = _require("sageattn_qk_int8_pv_fp8_cuda")

        def call(q, k, v):
            # SA2 sm120 path: per_warp + fp32 accum when ++ is off.
            return fn(
                q,
                k,
                v,
                tensor_layout="HND",
                qk_quant_gran="per_warp",
                pv_accum_dtype="fp32",
                smooth_v=True,
            )

        return call

    if backend == "fp8pp":
        fn = _require("sageattn_qk_int8_pv_fp8_cuda")

        def call(q, k, v):
            # Wan2GP / SA2 sm120 "++": fp32+fp16 accum, smooth_v off.
            return fn(
                q,
                k,
                v,
                tensor_layout="HND",
                qk_quant_gran="per_warp",
                pv_accum_dtype="fp32+fp16",
                smooth_v=False,
            )

        return call

    raise ValueError(f"unhandled sage backend {backend!r}")


def _sage_callable() -> Callable:
    global _SAGE_RESOLVED
    backend = _SAGE_BACKEND
    if _SAGE_RESOLVED is not None and _SAGE_RESOLVED[0] == backend:
        return _SAGE_RESOLVED[1]
    fn = _resolve_sage_callable(backend)
    _SAGE_RESOLVED = (backend, fn)
    return fn


def _sage(q: Tensor, k: Tensor, v: Tensor, seq: int) -> Tensor:
    """SageAttention on [H, S, D] (HND needs a batch dim).

    Falls back to torch SDPA if sageattention isn't installed or tensors aren't
    CUDA, so tiny CPU smoke tests still run. Forced backends that fail to import
    raise (so A/B doesn't silently measure the wrong kernel).
    """
    if not q.is_cuda:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # SA1/SA2 smooth_k may subtract mean from K in-place — clone K.
    q = q.contiguous()
    k = k.contiguous().clone()
    v = v.contiguous()
    qb = q.unsqueeze(0)
    kb = k.unsqueeze(0)
    vb = v.unsqueeze(0)

    try:
        fn = _sage_callable()
    except ImportError:
        if _SAGE_BACKEND in ("auto", "sa1"):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        raise

    out = fn(qb, kb, vb)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(0)


def _eager(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """q/k/v: [H, S, D]. Eager scaled dot-product attention."""
    scale = 1.0 / (q.shape[2] ** 0.5)
    scores = q @ k.transpose(1, 2) * scale  # [H, S, S]
    probs = _softmax(scores, 2)
    return probs @ v


def _fa2_callable() -> Callable:
    """Lazy-import flash_attn_func. Raises ImportError with install hint."""
    global _FA2_FN
    if _FA2_FN is not None:
        return _FA2_FN
    try:
        from flash_attn import flash_attn_func
    except ImportError as e:
        raise ImportError(
            "flash-attn is not installed; HQ path needs the FlashAttention-2 package. "
            "On Windows (owner box) pin the community cu130 wheel from pyproject.toml "
            "(same family as gemmy locate-anything-la-flash). "
            "Fallback: --attn flash (portable chunked) or --attn sage (default fast)."
        ) from e
    _FA2_FN = flash_attn_func
    return _FA2_FN


def _fa2(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Real FlashAttention-2 on [H, S, D] via flash_attn_func [B, S, H, D].

    Requires CUDA fp16/bf16. Does not silently fall back — HQ must fail closed
    if the wheel is missing so A/B never measures the wrong kernel.
    """
    if not q.is_cuda:
        raise RuntimeError("--attn fa2 requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"--attn fa2 needs fp16/bf16 activations, got {q.dtype}; "
            "sample on cuda uses bf16"
        )
    fn = _fa2_callable()
    # HND [H,S,D] → BSHD [1,S,H,D]
    qb = q.transpose(0, 1).unsqueeze(0).contiguous()
    kb = k.transpose(0, 1).unsqueeze(0).contiguous()
    vb = v.transpose(0, 1).unsqueeze(0).contiguous()
    out = fn(qb, kb, vb, causal=False)
    if isinstance(out, tuple):
        out = out[0]
    # [1,S,H,D] → [H,S,D]
    return out.squeeze(0).transpose(0, 1).contiguous()


def _flash(q: Tensor, k: Tensor, v: Tensor, block: int) -> Tensor:
    """q/k/v: [H, S, D]. Flash-style chunked attention with online softmax —
    the same rescaling math a fused kernel runs, with per-tile ops. Peak
    transient memory is one [H, Bq, Bk] scores tile plus running state, i.e.
    O(S); the full [H, S, S] matrix is never built."""
    scale = 1.0 / (q.shape[2] ** 0.5)
    h, s, d = q.shape
    out_blocks = []
    for q_off in range(0, s, block):
        q_end = min(q_off + block, s)
        qc = q[:, q_off:q_end]  # [H, Bq, D]
        m = None  # running row max  [H, Bq, 1]
        l = None  # running sum of exps
        o = None  # running weighted sum [H, Bq, D]
        for k_off in range(0, s, block):
            k_end = min(k_off + block, s)
            kc = k[:, k_off:k_end]
            vc = v[:, k_off:k_end]
            scores = qc @ kc.transpose(1, 2) * scale  # [H, Bq, Bk]
            block_max = scores.max(dim=2, keepdim=True).values
            if m is None:
                m_new, rescale = block_max, None
            else:
                m_new = torch.maximum(block_max, m)
                rescale = torch.exp(m - m_new)  # reweight already-accumulated state
            p = torch.exp(scores - m_new)  # [H, Bq, Bk]
            p_sum = p.sum(dim=2, keepdim=True)
            o_part = p @ vc  # [H, Bq, D]
            if rescale is None:
                l, o = p_sum, o_part
            else:
                l = l * rescale + p_sum
                o = o * rescale + o_part
            m = m_new
        out_blocks.append(o / l)  # [H, Bq, D]
    return torch.cat(out_blocks, dim=1)  # [H, S, D]


def rms_norm_headwise(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """RMSNorm applied per-head on [S, heads, head_dim] (weight is [head_dim])."""
    s, h, d = x.shape
    flat = x.reshape(s * h, d)
    n = rms_norm_fn(flat, w, eps)
    return n.reshape(s, h, d)


def rms_norm_fn(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """x * rsqrt(mean(x^2, -1) + eps) * w, broadcasting w over the last dim."""
    x2 = x * x
    mean = x2.mean(dim=1, keepdim=True)
    inv = (mean + eps).pow(-0.5)
    return x * inv * w


__all__ = [
    "ATTN_CHOICES",
    "AUTO_THRESHOLD",
    "Attention",
    "AttnImpl",
    "FLASH_BLOCK",
    "SAGE_BACKENDS",
    "_eager",
    "_fa2",
    "_flash",
    "get_sage_backend",
    "set_sage_backend",
]


__all__ = [
    "Attention",
    "AttnImpl",
    "AUTO_THRESHOLD",
    "FLASH_BLOCK",
    "SAGE_BACKENDS",
    "get_sage_backend",
    "set_sage_backend",
    "rms_norm_fn",
    "rms_norm_headwise",
]
