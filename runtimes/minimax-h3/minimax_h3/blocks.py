"""Core building blocks: Linear, RMSNorm, MLP, AdalnProj, DiTBlock,
RefinerBlock, TokenRefiner, FinalLayer.

A 1:1 Python port of ``src/blocks.rs`` (which ports the homonymous classes in
ComfyUI ``comfy/ldm/minimax/model.py``). Kept dependency-light: each layer
holds its own plain ``Tensor`` weights (no ``nn.Module``), so the loader can
place them on any device/dtype and stream them per block — exactly the design
of the Candle port.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .weights import rotate_activation


def linear(src, prefix: str, out: int, inp: int, bias: bool, dtype: torch.dtype) -> "Linear":
    name = f"{prefix}.weight"
    # Deferred ConvRot INT8 (streamed loads): keep the packed int8 in RAM, w
    # stays a zero-size placeholder until the block streams to the device;
    # `Linear.forward` folds the ConvRot Hadamard into the activation and
    # dequantizes on the device from the 1-byte int8 payload.
    d = src.get_deferred_int8(name, (out, inp), dtype)
    if d is not None:
        b = src.get(f"{prefix}.bias", (out,), dtype) if bias else None
        return Linear(None, b, None, deferred_int8=d)
    w, scale = src.get_linear(name, (out, inp), dtype)
    b = src.get(f"{prefix}.bias", (out,), dtype) if bias else None
    return Linear(w, b, scale)


def rms_norm(src, prefix: str, dim: int, eps: float, dtype: torch.dtype) -> "RmsNorm":
    w = src.get(f"{prefix}.weight", (dim,), dtype)
    return RmsNorm(w, eps)


class Linear:
    """y = x @ w^T (+ bias); with `scale` set the weight is fp8-quantized and
    the per-output-channel fp32 scale is folded into the matmul output,
    `y = (x @ w_q^T) * scale[None, :]` — the fp32 dequant is never
    materialized as a full weight.

    With `deferred_int8` set (streamed runs on the real int8 ConvRot
    checkpoint), `w` is a placeholder; the packed int8 `q` + per-row scale
    live in the payload and are uploaded to the device by `to()`. Forwarding
    a RAM copy that was never transferred is a programming error — bail.

    Host-side deferred payloads keep a pinned CPU mirror so async H2D
    prefetch can overlap the next block's transfer with the current compute.
    """

    # Deferred int8 GEMM backend:
    # - "cast" (default): ephemeral int8→bf16/fp16 inflate + matmul — proven winner
    #   on owner sm120 (RTX 5060 Ti); cuBLAS bf16 GEMM beats stock int8 paths here.
    # - "int_mm": experimental torch._int_mm + dynamic act quant (not bit-exact; slower)
    # - "triton": minimax_h3.int8_gemm fused bf16×int8 (near-cast numerics; slower on sm120)
    # - "pack": torch._weight_int8pack_mm (stock; ~100× slower on sm120 — keep for rebench)
    # Flip only after scripts/bench_int8_linear.py shows a real win on-device.
    # Hand custom W8A16 CUDA was tried and removed (~5–10× slower than cast on sm120).
    deferred_int8_backend: str = "cast"

    def __init__(
        self,
        w: Tensor,
        b: Tensor | None,
        scale: Tensor | None,
        deferred_int8=None,
    ):
        self.w = w
        self.b = b
        self.scale = scale
        self.deferred_int8 = deferred_int8
        # Pinned host mirrors for deferred payloads (set on first CPU residence).
        # Keyed by payload attr ("q"/"scale" for DeferredInt8).
        self._host_parts: dict[str, Tensor] | None = None
        self._host_b: Tensor | None = None
        self._device_ready = False
        # Host mirrors are captured lazily on first H2D so load doesn't pin
        # the entire ~21 GB checkpoint up front.

    def _deferred(self):
        """Active deferred payload, or None."""
        return self.deferred_int8

    def _parts(self) -> tuple[tuple[str, Tensor], ...]:
        """(attr, tensor) host-resident parts of the active payload."""
        d = self._deferred()
        if d is None:
            return ()
        return tuple((a, t) for a, t in d.host_parts() if t is not None)

    def pin_host(self) -> bool:
        """Pin deferred host storages for async H2D (idempotent).

        Returns True if this layer is pinned (or already was).
        """
        if self._deferred() is None:
            return True
        return self._capture_host_deferred(pin=True)

    def adopt_host_pageable(self) -> None:
        """Keep current CPU tensors as the host mirror **without** pinning.

        Used when the sample path stops pinning to protect system RAM. Later
        ``.to(cuda)`` will not try to pin; H2D may be a bit slower.
        """
        d = self._deferred()
        if d is None:
            return
        if self._host_parts is not None:
            return
        parts = self._parts()
        if not parts or parts[0][1].device.type != "cpu":
            return
        self._host_parts = dict(parts)
        if self.b is not None and self.b.device.type == "cpu":
            self._host_b = self.b
        self._device_ready = False

    def _capture_host_deferred(self, pin: bool = True) -> bool:
        """Snapshot CPU copies used as the source of H2D.

        When ``pin=True``, replaces pageable storages with pinned ones (one
        tensor at a time) so we don't keep two full host copies. When
        ``pin=False``, just records the existing pageable tensors as host
        mirrors (RAM-reserve path).
        """
        d = self._deferred()
        assert d is not None
        if self._host_parts is not None:
            return True
        parts = self._parts()
        if not parts or parts[0][1].device.type != "cpu":
            return True

        if not pin:
            self.adopt_host_pageable()
            return True

        def _pin_replace(t: Tensor) -> Tensor:
            if t.is_pinned():
                return t
            # Allocate pinned, copy, then drop all refs to pageable `t` so the
            # OS can reclaim it (critical on a 64 GB box — double residency is
            # what spiked Task Manager to ~99%).
            pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            pinned.copy_(t)
            return pinned

        # Drop live refs before deleting old tensors so storage can free.
        mirrors: dict[str, Tensor] = {}
        for attr, old in parts:
            setattr(d, attr, None)
            pinned = _pin_replace(old)
            del old
            setattr(d, attr, pinned)
            mirrors[attr] = pinned
        self._host_parts = mirrors

        if self.b is not None and self.b.device.type == "cpu":
            old_b = self.b
            self.b = None  # type: ignore[assignment]
            b = _pin_replace(old_b)
            del old_b
            self.b = b
            self._host_b = b
        self._device_ready = False
        return True

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        d = self._deferred()
        if d is not None:
            if device is None:
                return self
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type == "cpu":
                # Drop device copies; restore host mirrors as the live payload.
                if self._host_parts is None:
                    # First time we've seen CPU after a device-only construction.
                    # Do **not** pin here — pinning is an explicit sample setup step
                    # so RAM reserve can leave some layers pageable.
                    parts = self._parts()
                    if parts and parts[0][1].device.type == "cpu":
                        self.adopt_host_pageable()
                    return self
                for attr, t in self._host_parts.items():
                    setattr(d, attr, t)
                d.drop_device_cache()
                if self._host_b is not None:
                    self.b = self._host_b
                self._device_ready = False
                return self
            # Device path: prefer host mirror → H2D (async only if pinned).
            parts = self._parts()
            if self._host_parts is None and parts and parts[0][1].device.type == "cpu":
                # Never pin implicitly mid-run (would blow a RAM reserve).
                self.adopt_host_pageable()
            if self._host_parts is not None:
                for attr, t in self._host_parts.items():
                    setattr(d, attr, t.to(device=dev, non_blocking=non_blocking))
                if self._host_b is not None:
                    self.b = self._host_b.to(device=dev, dtype=dtype, non_blocking=non_blocking)
                elif self.b is not None:
                    self.b = self.b.to(device=dev, dtype=dtype, non_blocking=non_blocking)
            else:
                for attr, t in parts:
                    setattr(d, attr, t.to(device=dev, non_blocking=non_blocking))
                if self.b is not None:
                    self.b = self.b.to(device=dev, dtype=dtype, non_blocking=non_blocking)
            self._device_ready = True
            return self
        self.w = self.w.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self.b is not None:
            self.b = self.b.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self.scale is not None:
            self.scale = self.scale.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.deferred_int8 is not None:
            d = self.deferred_int8
            if d.q is None or d.q.device != x.device:
                raise RuntimeError(
                    "Linear::forward called on a deferred int8 weight before it was "
                    "transferred to the input device (call .to(device) first)"
                )
            # y = x @ W^T, W = rotate(q·scale): fold the block-Hadamard into
            # the activation (H_gs symmetric/orthogonal), then int8 weight path.
            # Cast is ephemeral — no long-lived bf16 weight cache (VRAM budget).
            xr = rotate_activation(x, d.gs)
            scale = d.scale if d.scale.dtype == x.dtype else d.scale.to(dtype=x.dtype)
            backend = Linear.deferred_int8_backend
            if (
                backend == "int_mm"
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
            ):
                y = _deferred_int8_int_mm(xr, d.q, scale)
            elif (
                backend == "triton"
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
            ):
                y = _deferred_int8_triton(xr, d.q, d.scale, scale)
            elif backend == "pack" and x.is_cuda and hasattr(torch, "_weight_int8pack_mm"):
                y = _deferred_int8_pack(xr, d.q, d.scale, x.dtype)
            else:
                # Default / fallback: cast weight to activation dtype, matmul,
                # fold per-row scale. Keeping q as int8 until the cast keeps
                # the resident weight at 1 byte/elem.
                y = _deferred_int8_cast(xr, d.q, scale)
            if self.b is not None:
                y.add_(self.b)
            return y
        if self.scale is None:
            y = x @ self.w.t()
        else:
            # fp8 e4m3 weight: decode the raw bits to fp32, matmul in the
            # stream dtype, then broadcast the per-output-channel scale.
            wq = decode_e4m3(self.w).to(x.dtype)
            y = x @ wq.t()
            y = y * self.scale.to(x.dtype).unsqueeze(0)
            if self.b is not None:
                y = y + self.b
        return y


def _deferred_int8_cast(xr: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    """Production path: ephemeral int8→act-dtype inflate + matmul + row scale."""
    wf = q.to(dtype=xr.dtype)
    y = xr.matmul(wf.t())
    del wf
    y.mul_(scale.unsqueeze(0))
    return y


def _deferred_int8_pack(xr: Tensor, q: Tensor, scale: Tensor, out_dtype: torch.dtype) -> Tensor:
    """Stock ``torch._weight_int8pack_mm``; falls back to cast if missing/slow path errors."""
    try:
        y = torch._weight_int8pack_mm(xr, q, scale)
        return y if y.dtype == out_dtype else y.to(dtype=out_dtype)
    except Exception:
        sc = scale if scale.dtype == xr.dtype else scale.to(dtype=xr.dtype)
        return _deferred_int8_cast(xr, q, sc)


def _deferred_int8_triton(xr: Tensor, q: Tensor, scale_f: Tensor, scale_act: Tensor) -> Tensor:
    """Triton bf16×int8 GEMM; falls back to cast on import/launch failure."""
    try:
        from .int8_gemm import triton_available, weight_int8_mm

        if not triton_available():
            return _deferred_int8_cast(xr, q, scale_act)
        return weight_int8_mm(xr, q, scale_f)
    except Exception:
        return _deferred_int8_cast(xr, q, scale_act)


def _deferred_int8_int_mm(xr: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    """Experimental int8 weight GEMM without a full bf16 weight clone.

    Uses ``torch._int_mm`` when both operands can be int8. Activations are
    dynamically quantized per-row to int8; result is rescaled by
    ``act_scale_row * weight_scale_col``. Numerics are **not** bit-identical to
    the cast path — callers must gate on a tolerance and a speed win before
    enabling via ``Linear.deferred_int8_backend = "int_mm"``.

    Falls back to cast-matmul if shapes/dtypes are unsupported.
    """
    # _int_mm: (m,k) int8 @ (k,n) int8 -> (m,n) int32. q is [out, in] = [n, k].
    if xr.dim() != 2 or q.dim() != 2 or xr.shape[-1] != q.shape[-1]:
        return _deferred_int8_cast(xr, q, scale)
    try:
        # Per-row absmax quant for activations (symmetric).
        # y_approx = (xr_i8 * xr_scale) @ (q_i8^T * w_scale)
        #          = (xr_i8 @ q_i8^T) * xr_scale_row * w_scale_col
        eps = 1e-8
        # xr: [m, k], q: [n, k]
        absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        xr_scale = absmax / 127.0
        xr_i8 = torch.clamp((xr / xr_scale).round(), -127, 127).to(torch.int8)
        # q already int8; _int_mm wants (m,k) @ (k,n) so pass q.t().contiguous()
        qt = q.t().contiguous()  # [k, n]
        if hasattr(torch, "_int_mm"):
            acc = torch._int_mm(xr_i8.contiguous(), qt)  # [m, n] int32
        else:
            acc = xr_i8.float().matmul(qt.float()).to(torch.int32)
        y = (
            acc.to(dtype=xr.dtype)
            * xr_scale.to(dtype=xr.dtype)
            * scale.unsqueeze(0).to(dtype=xr.dtype)
        )
        return y
    except Exception:
        return _deferred_int8_cast(xr, q, scale)


class RmsNorm:
    def __init__(self, w: Tensor, eps: float):
        self.w = w
        self.eps = eps

    def to(self, device=None, dtype=None):
        self.w = self.w.to(device=device, dtype=dtype)
        return self

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm_fn(x, self.w, self.eps)


def rms_norm_fn(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """x * rsqrt(mean(x^2, -1) + eps) * w, broadcasting w over the last dim."""
    mean = (x * x).mean(dim=-1, keepdim=True)
    inv = (mean + eps).pow(-0.5)
    return x * inv * w


class MLP:
    """Gated SiLU MLP: fc1 [2*ffn, hidden] (no bias), split, silu(gate)*up, fc2."""

    def __init__(self, src, prefix: str, hidden: int, ffn: int, dtype: torch.dtype):
        self.fc1 = linear(src, f"{prefix}.fc1", ffn * 2, hidden, False, dtype)
        self.fc2 = linear(src, f"{prefix}.fc2", hidden, ffn, False, dtype)
        self.ffn = ffn

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.fc1.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.fc2.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def pin_host(self) -> None:
        self.fc1.pin_host()
        self.fc2.pin_host()

    def adopt_host_pageable(self) -> None:
        self.fc1.adopt_host_pageable()
        self.fc2.adopt_host_pageable()

    def forward(self, x: Tensor) -> Tensor:
        ffn = self.ffn
        y = self.fc1.forward(x)  # [S, 2*ffn]
        # In-place silu on the gate half, then mul into up — same math as
        # silu(gate) * up without a second activation temp for silu alone.
        gate = y[:, :ffn]
        up = y[:, ffn:]
        torch.nn.functional.silu(gate, inplace=True)
        up.mul_(gate)
        return self.fc2.forward(up)


class AdalnProj:
    """AdaLN projection: t_emb [M, t_dim] -> `expand` tensors of
    [M*modalities, hidden]."""

    def __init__(
        self,
        src,
        prefix: str,
        t_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        apply_silu: bool,
        dtype: torch.dtype,
    ):
        out = expand * hidden * modalities
        self.expand = expand
        self.modalities = modalities
        self.linear = linear(src, prefix, out, t_dim, True, dtype)
        self.apply_silu = apply_silu
        # Optional precomputed mods[list of expand tensors]; when set, forward
        # ignores t_emb and returns these (Sol-Engine-style AdaLN precompute).
        self._precomputed: list[Tensor] | None = None

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.linear.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self._precomputed is not None and device is not None:
            self._precomputed = [
                t.to(device=device, non_blocking=non_blocking) for t in self._precomputed
            ]
        return self

    def pin_host(self) -> None:
        self.linear.pin_host()

    def adopt_host_pageable(self) -> None:
        self.linear.adopt_host_pageable()

    def clear_precomputed(self) -> None:
        self._precomputed = None

    def set_precomputed(self, mods: list[Tensor]) -> None:
        """Install precomputed modulation rows (length == expand)."""
        if len(mods) != self.expand:
            raise ValueError(f"expected {self.expand} mod tensors, got {len(mods)}")
        self._precomputed = mods

    def forward(self, t_emb: Tensor) -> list[Tensor]:
        if self._precomputed is not None:
            return self._precomputed
        x = torch.nn.functional.silu(t_emb) if self.apply_silu else t_emb
        x = self.linear.forward(x)  # [M, expand*hidden*modalities]
        m, cols = x.shape
        hidden = cols // (self.expand * self.modalities)
        x = x.reshape(m * self.modalities, self.expand * hidden)
        out = []
        for i in range(self.expand):
            out.append(x[:, i * hidden : (i + 1) * hidden].contiguous())
        return out


def mod_scale_shift(h: Tensor, shift: Tensor, scale: Tensor, segments) -> Tensor:
    """h[a:b] = h[a:b] * (1 + scale[row]) + shift[row] for each segment.

    Prefer pre-casting AdaLN rows to the stream dtype once per step
    (``MiniMaxH3Model.precompute_adaln_for_t_emb``). If rows still differ
    (fp32 curve island), cast per segment like the reference.

    Writes into a single preallocated output (no per-segment cat).
    """
    hidden = h.shape[1]
    out = torch.empty_like(h)
    same_dtype = scale.dtype == h.dtype and shift.dtype == h.dtype
    for a, b, row in segments:
        if same_dtype:
            sc = scale[row].reshape(hidden)
            sh = shift[row].reshape(hidden)
        else:
            sc = scale[row].reshape(hidden).to(dtype=h.dtype)
            sh = shift[row].reshape(hidden).to(dtype=h.dtype)
        # out[a:b] = h[a:b] * (1 + sc) + sh
        torch.addcmul(sh, h[a:b], sc + 1, out=out[a:b])
    return out


def mod_gate(x: Tensor, gate: Tensor, other: Tensor, segments) -> Tensor:
    """In-place ``x[a:b] += gate[row] * other[a:b]`` for each segment.

    Prefer pre-cast gate rows (stream dtype). Returns ``x`` (mutated).
    """
    hidden = x.shape[1]
    same_dtype = gate.dtype == x.dtype
    for a, b, row in segments:
        g = gate[row].reshape(hidden) if same_dtype else gate[row].reshape(hidden).to(dtype=x.dtype)
        x[a:b].addcmul_(other[a:b], g)
    return x


class DiTBlock:
    def __init__(
        self,
        src,
        prefix: str,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        t_dim: int,
        eps: float,
        qk_eps: float,
        apply_silu: bool,
        dtype: torch.dtype,
        adaln_dtype: torch.dtype,
        attn_impl: str,
    ):
        from .attention import Attention

        self.norm1 = rms_norm(src, f"{prefix}.norm1", hidden, eps, dtype)
        self.norm2 = rms_norm(src, f"{prefix}.norm2", hidden, eps, dtype)
        self.attn = Attention(src, f"{prefix}.attn", hidden, heads, head_dim, qk_eps, dtype, attn_impl)
        self.mlp = MLP(src, f"{prefix}.mlp", hidden, ffn, dtype)
        self.adaln_proj = AdalnProj(
            src, f"{prefix}.adaln_proj.linear", t_dim, hidden, 6, 3, apply_silu, adaln_dtype
        )

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.norm1.to(device=device, dtype=dtype)
        self.norm2.to(device=device, dtype=dtype)
        self.attn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.mlp.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.adaln_proj.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def pin_host(self) -> None:
        self.attn.pin_host()
        self.mlp.pin_host()
        self.adaln_proj.pin_host()

    def adopt_host_pageable(self) -> None:
        self.attn.adopt_host_pageable()
        self.mlp.adopt_host_pageable()
        self.adaln_proj.adopt_host_pageable()

    def forward(self, x: Tensor, t_emb: Tensor, mod_segments, rope_cos, rope_sin) -> Tensor:
        from .profile import get_profiler

        prof = get_profiler()
        if prof is not None and prof.enabled:
            with prof.phase("adaln"):
                mods = self.adaln_proj.forward(t_emb)
            with prof.phase("norm_mod"):
                h = self.norm1.forward(x)
                h = mod_scale_shift(h, mods[0], mods[1], mod_segments)
            with prof.phase("attn"):
                attn_out = self.attn.forward(h, rope_cos, rope_sin)
            with prof.phase("norm_mod"):
                x = mod_gate(x, mods[2], attn_out, mod_segments)
                h = self.norm2.forward(x)
                h = mod_scale_shift(h, mods[3], mods[4], mod_segments)
            with prof.phase("mlp"):
                mlp_out = self.mlp.forward(h)
            with prof.phase("norm_mod"):
                out = mod_gate(x, mods[5], mlp_out, mod_segments)
            return out
        mods = self.adaln_proj.forward(t_emb)
        h = self.norm1.forward(x)
        h = mod_scale_shift(h, mods[0], mods[1], mod_segments)
        attn_out = self.attn.forward(h, rope_cos, rope_sin)
        x = mod_gate(x, mods[2], attn_out, mod_segments)
        h = self.norm2.forward(x)
        h = mod_scale_shift(h, mods[3], mods[4], mod_segments)
        mlp_out = self.mlp.forward(h)
        return mod_gate(x, mods[5], mlp_out, mod_segments)


class RefinerBlock:
    def __init__(
        self,
        src,
        prefix: str,
        hidden: int,
        heads: int,
        head_dim: int,
        ffn: int,
        eps: float,
        qk_eps: float,
        dtype: torch.dtype,
        attn_impl: str,
    ):
        from .attention import Attention

        self.norm1 = rms_norm(src, f"{prefix}.norm1", hidden, eps, dtype)
        self.norm2 = rms_norm(src, f"{prefix}.norm2", hidden, eps, dtype)
        self.attn = Attention(src, f"{prefix}.attn", hidden, heads, head_dim, qk_eps, dtype, attn_impl)
        self.mlp = MLP(src, f"{prefix}.mlp", hidden, ffn, dtype)

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.norm1.to(device=device, dtype=dtype)
        self.norm2.to(device=device, dtype=dtype)
        self.attn.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.mlp.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def pin_host(self) -> None:
        self.attn.pin_host()
        self.mlp.pin_host()

    def adopt_host_pageable(self) -> None:
        self.attn.adopt_host_pageable()
        self.mlp.adopt_host_pageable()

    def forward(self, x: Tensor, rope_cos, rope_sin) -> Tensor:
        h = self.norm1.forward(x)
        h = self.attn.forward(h, rope_cos, rope_sin)
        x = x + h
        h = self.norm2.forward(x)
        h = self.mlp.forward(h)
        return x + h


class TokenRefiner:
    def __init__(self, src, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps, dtype, attn_impl):
        self.blocks = [
            RefinerBlock(
                src, f"token_refiner.blocks.{i}", hidden, heads, head_dim, ffn, eps, qk_eps, dtype, attn_impl
            )
            for i in range(num_layers)
        ]
        self.final_norm = rms_norm(src, "token_refiner.final_norm", hidden, final_eps, dtype)

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        for b in self.blocks:
            b.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.final_norm.to(device=device, dtype=dtype)
        return self

    def pin_host(self) -> None:
        for b in self.blocks:
            b.pin_host()

    def adopt_host_pageable(self) -> None:
        for b in self.blocks:
            b.adopt_host_pageable()

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for b in self.blocks:
            h = b.forward(h, None, None)
        return self.final_norm.forward(h)


class FinalLayer:
    def __init__(self, src, hidden, t_dim, video_dim, audio_dim, eps, apply_silu, dtype, adaln_dtype):
        self.norm = rms_norm(src, "final_layer.norm", hidden, eps, dtype)
        self.adaln_proj = AdalnProj(
            src, "final_layer.adaln_proj.linear", t_dim, hidden, 2, 1, apply_silu, adaln_dtype
        )
        # output heads are the checkpoint's fp32 island
        self.video_out = linear(src, "final_layer.video_out", video_dim, hidden, True, torch.float32)
        self.audio_out = linear(src, "final_layer.audio_out", audio_dim, hidden, True, torch.float32)

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        self.norm.to(device=device, dtype=dtype)
        self.adaln_proj.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.video_out.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.audio_out.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, x: Tensor, t_emb: Tensor, video_seg, audio_seg):
        """video_seg / audio_seg: (start, stop, timestep_row) of the target
        streams."""
        mods = self.adaln_proj.forward(t_emb)  # shift, scale [M, hidden]
        shift, scale = mods[0], mods[1]
        hidden = x.shape[1]

        # Reference: (norm(x[va:vb]) * (1 + scale[vrow]) + shift[vrow]) cast to
        # fp32. In curve mode scale/shift are fp32, so torch type-promotes the
        # whole modulation to fp32 — replicate by lifting the normed slice to
        # fp32 first (a no-op for classic mode).
        hv = _apply_mod_row(
            self.norm.forward(x[video_seg[0] : video_seg[1]]).to(torch.float32),
            shift,
            scale,
            video_seg[2],
            hidden,
        )
        ha = _apply_mod_row(
            self.norm.forward(x[audio_seg[0] : audio_seg[1]]).to(torch.float32),
            shift,
            scale,
            audio_seg[2],
            hidden,
        )
        v = self.video_out.forward(hv)
        a = self.audio_out.forward(ha)
        return v, a


def _apply_mod_row(h: Tensor, shift: Tensor, scale: Tensor, row: int, hidden: int) -> Tensor:
    n = h.shape[0]
    sc = scale[row].reshape(hidden).to(h.dtype)
    sh = shift[row].reshape(hidden).to(h.dtype)
    return h * (1.0 + sc) + sh


# ---------------------------------------------------------------------------
# fp8 (e4m3) decode


def decode_e4m3(bits: Tensor) -> Tensor:
    """Decode a uint8 tensor of raw e4m3 bit patterns to fp32.

    e4m3fn layout: 1 sign bit, 4 exponent bits, 3 mantissa bits; no inf/nan.
      normal:   (-1)^s * (1 + m/8) * 2^(e-7)
      subnormal (e == 0): (-1)^s * (m/8) * 2^-6
    Used to reconstruct fp8 weights on load so `Linear.forward` can fold the
    per-channel scale into the matmul output exactly like the Candle port.
    """
    b = bits.to(torch.int32)
    sign = ((b >> 7) & 1).float()
    exp = (b >> 3) & 0xF
    mant = (b & 0x7).float()
    sign_val = torch.where(sign == 1, torch.tensor(-1.0, device=bits.device), torch.tensor(1.0, device=bits.device))
    normal = (1.0 + mant / 8.0) * torch.pow(2.0, (exp - 7).float())
    subnormal = (mant / 8.0) * (2.0 ** -6.0)
    val = torch.where(exp == 0, subnormal, normal)
    return val * sign_val


__all__ = [
    "Linear",
    "RmsNorm",
    "MLP",
    "AdalnProj",
    "DiTBlock",
    "RefinerBlock",
    "TokenRefiner",
    "FinalLayer",
    "linear",
    "rms_norm",
    "rms_norm_fn",
    "mod_scale_shift",
    "mod_gate",
    "decode_e4m3",
]
