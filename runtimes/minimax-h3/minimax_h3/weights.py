"""Weight sources (random init / safetensors), checkpoint loading, saving and
auditing.

A 1:1 Python port of ``src/weights.rs``. The ``TensorSource`` protocol lets
load-time transforms wrap each other the same way the Candle port does:
``RandomSource`` → ``QuantizingSource`` → ``AdalnSvdSource`` → model.

**Key-name contract with the Rust loader.** ``expected_params`` emits the
exact tensor names/shapes ``SafetensorsSource`` (Rust) expects — that is the
contract `scripts/parity_check.py` audits mechanically by writing a
checkpoint here and running the Rust binary's `audit`/`infer` against it.
``strip_prefix`` must stay in lockstep with ``src/weights.rs``
(``model.diffusion_model.`` / ``diffusion_model.`` / ``model.`` / ``t5_xxl.``),
and any tensor written via ``save_safetensors`` should use these names to be
loadable by the Candle runtime.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# TensorSource protocol


class TensorSource:
    """Source of named weight tensors (random init or safetensors).

    ``get`` returns a tensor of the requested dtype/shape (sources convert).
    ``get_linear`` lets a source hand back a *quantized* Linear weight together
    with a per-output-channel fp32 scale (e.g. fp8 e4m3 with the scale folded
    into the matmul output). The default returns the plain weight, no scale.
    """

    def get(self, name: str, shape, dtype: torch.dtype) -> Tensor:
        raise NotImplementedError

    def get_linear(self, name: str, shape, dtype: torch.dtype):
        return self.get(name, shape, dtype), None

    def get_deferred_int8(self, name: str, shape, dtype: torch.dtype):
        """Optional deferred ConvRot INT8 source for a linear weight. Default:
        none — every weight dequantizes at load. `SafetensorsSource` returns a
        `DeferredInt8` under deferred loading (streamed runs): the caller keeps
        the packed int8 in RAM and dequantizes on the target device at
        transfer time (`Linear.forward` folds the ConvRot Hadamard into the
        activation, so only the 1-byte int8 payload ever crosses PCIe)."""
        return None

    def device(self):
        raise NotImplementedError

    def compute_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def stored_shape(self, name: str) -> Optional[Tuple[int, ...]]:
        """Stored shape for `name`, if this source actually has that tensor
        (format probe — detects whether a checkpoint ships the curve table
        rather than trusting the config)."""
        return None


# ---------------------------------------------------------------------------
# Box-Muller draws (deterministic per seed, reproducible within this package —
# not bit-identical to torch.randn or the Rust StdRng stream)


def _randn_std_normal(shape, seed: int, scale: float = 1.0, device="cpu"):
    """Standard-normal draws via Box-Muller; deterministic per seed. Scaled
    small for random-weight smoke tests so fp32 softmax stays finite."""
    rng = random.Random(seed)
    n, d = shape
    vals = []
    while len(vals) < n * d:
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        mag = math.sqrt(-2.0 * math.log(u1))
        ang = 2.0 * math.pi * u2
        vals.append(mag * math.cos(ang) * scale)
        vals.append(mag * math.sin(ang) * scale)
    t = torch.tensor(vals[: n * d], dtype=torch.float32, device=device).reshape(n, d)
    return t


class RandomSource(TensorSource):
    """Random initialization (for the shape smoke test). Every `get` call
    re-seeds with the same seed, so all tensors of one shape are identical —
    a quirk preserved from the Rust `RandomSource`."""

    def __init__(self, device="cpu", dtype: torch.dtype = torch.float32, seed: int = 42):
        self._device = device
        self.dtype = dtype
        self.seed = seed

    def get(self, _name: str, shape, dtype: torch.dtype) -> Tensor:
        total = 1
        for s in shape:
            total *= s
        # same Box-Muller stream as _randn_std_normal, scaled 0.02
        t = _randn_std_normal((1, total), self.seed, scale=0.02, device=self._device).reshape(shape)
        return t.to(dtype)

    def device(self):
        return self._device

    def compute_dtype(self) -> torch.dtype:
        return self.dtype


# ---------------------------------------------------------------------------
# safetensors loading (prefix-tolerant, records missing / mismatched keys)
#
# Lazy file-backed reader (`safetensors.safe_open`): a 20 GB pruned int8
# checkpoint never materializes its raw buffer in RAM. ConvRot INT8 weights
# (`X.weight` I8 + `X.weight_scale` F32 per-output-channel scale) are
# dequantized to the compute dtype on read — the same contract as the Candle
# runtime's `SafetensorsSource` — so the pruned int8_convrot production
# checkpoints load directly.


_PREFIXES = ["model.diffusion_model.", "diffusion_model.", "model.", "t5_xxl."]


def strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


# --- ConvRot INT8 decode (Comfy `int8_convrot`, comfy-kitchen contract) ----


def convrot_group_size(in_dim: int) -> Optional[int]:
    """Largest power-of-4 divisor of `in_dim` (256 for the main linears,
    64 for the 2688-wide adaln projections) — mirrors `convrot_group_size`
    in src/weights.rs."""
    for g in (256, 64, 16, 4):
        if in_dim % g == 0:
            return g
    return None


def is_companion_key(name: str) -> bool:
    """Scale / format companions of quantized linears are consumed by the
    loader, never by the model — they must not surface as unused keys in
    audits. Covers ConvRot INT8 (``weight_scale`` + ``comfy_quant``) and
    NVFP4 (``weight_scale`` + ``weight_scale_2`` + ``comfy_quant``)."""
    return (
        name.endswith(".weight_scale")
        or name.endswith(".weight_scale_2")
        or name.endswith(".comfy_quant")
    )


def _convrot_hadamard_gs(gs: int) -> np.ndarray:
    """The normalized block-Hadamard `H_gs` as a [gs, gs] matrix: Kronecker
    power of the 4×4 *regular* Hadamard base, each factor scaled by 1/2 so the
    full transform is normalized by 1/√gs (H_gs² = I). Equals the O(gs·log₄gs)
    butterfly the Candle port applies (apply_convrot_hadamard) — the matrix is
    symmetric, so `x @ H` and `H @ x` agree."""
    base = np.array(
        [[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    h = base * 0.5
    while h.shape[0] < gs:
        h = np.kron(h, base * 0.5)
    return h


def apply_convrot_hadamard(x: np.ndarray, gs: int) -> np.ndarray:
    """`x <- x @ H_gs` for a batch of rows, in place, via the base-4 digit
    butterfly (O(gs·log₄gs) per row) — a direct port of
    `apply_convrot_hadamard` in src/weights.rs. `x` is [..., gs]; the
    transform runs along the last axis."""
    modes = 0
    s = gs
    while s > 1:
        s //= 4
        modes += 1
    for m in range(modes):
        tmp = x.copy()
        stride = 4 ** m
        for base in range(0, gs, stride * 4):
            i0 = base
            v0 = tmp[..., i0 : i0 + stride]
            v1 = tmp[..., i0 + stride : i0 + 2 * stride]
            v2 = tmp[..., i0 + 2 * stride : i0 + 3 * stride]
            v3 = tmp[..., i0 + 3 * stride : i0 + 4 * stride]
            x[..., i0 : i0 + stride] = (v0 + v1 + v2 - v3) * 0.5
            x[..., i0 + stride : i0 + 2 * stride] = (v0 + v1 - v2 + v3) * 0.5
            x[..., i0 + 2 * stride : i0 + 3 * stride] = (v0 - v1 + v2 + v3) * 0.5
            x[..., i0 + 3 * stride : i0 + 4 * stride] = (-v0 + v1 + v2 + v3) * 0.5
    return x


def dequantize_convrot(q: Tensor, scale: Tensor) -> Tensor:
    """Dequantize a ConvRot INT8 weight: `W = rotate(q · scale, H_gs, gs)` —
    per output row, each group of `gs` consecutive in-cols is rotated by the
    normalized block-Hadamard and scaled by that row's per-output-channel
    scale. `q` is [out, in] int8, `scale` is [out] (or [out, 1]) f32; returns
    [out, in] f32. Group size is auto-selected by the power-of-4 rule.
    Mirrors `dequantize_convrot` in src/weights.rs (rotate-then-scale is
    linear, so rotate(q·s) == rotate(q)·s — bit-identical to the Rust order)."""
    out, inp = q.shape
    gs = convrot_group_size(inp)
    if gs is None:
        raise ValueError(f"in_features {inp} not divisible by any convrot group size")
    return dequantize_convrot_gs(q, scale, gs)


def dequantize_convrot_gs(q: Tensor, scale: Tensor, gs: int) -> Tensor:
    """`dequantize_convrot` with the group size pinned explicitly — the
    loader path uses the auto rule (2688-wide adaln inputs → 64), while the
    golden test simulates a slice of such a row (a 256-wide slice of a
    2688-wide row is 4×64 groups, *not* one 256 group). Mirrors
    `dequantize_convrot_gs` in src/weights.rs."""
    out, inp = q.shape
    assert inp % gs == 0, f"in_features {inp} not divisible by group size {gs}"
    s = scale.detach().cpu().numpy().astype(np.float32).reshape(-1)
    qn = q.detach().cpu().numpy().astype(np.float32)
    ng = inp // gs
    w = apply_convrot_hadamard(qn.reshape(out, ng, gs).copy(), gs)  # [out, ng, gs]
    w *= s[:, None, None]
    return torch.from_numpy(w.reshape(out, inp))


# --- Deferred ConvRot INT8 (streamed runs: dequantize on the device) ------
#
# The ConvRot decode is `W = rotate(q·scale)` with the normalized block-
# Hadamard applied per group of `gs` along the *input* dim. Since H_gs is
# symmetric and orthogonal (H_gs² = I), the rotation can be folded into the
# activation: y = x @ W^T = (x @ H_block) @ (q·scale)^T. So a streamed linear
# keeps the packed int8 `q` (1 byte/elem) + per-row f32 scale in RAM, uploads
# them as-is, and computes `y = (x_rot @ q^T) * scale[None, :]` on the device —
# the dequantized weight is never materialized on the host or in VRAM.


@dataclass
class DeferredInt8:
    q: Tensor  # [out, in] int8 (packed, 1 byte/elem)
    scale: Tensor  # [out] f32 per-output-channel scale
    gs: int  # group size (256 for main linears, 64 for 2688-wide adaln)

    def host_parts(self) -> Tuple[Tuple[str, Optional[Tensor]], ...]:
        """(attr, tensor) pairs that live on the host between uploads."""
        return (("q", self.q), ("scale", self.scale))

    def drop_device_cache(self) -> None:
        """Release device-transient decode caches on eviction (int8 has none)."""


_HADAMARD_CACHE: Dict[int, Tensor] = {}


def hadamard_matrix(gs: int, device="cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    """The normalized block-Hadamard `H_gs` as a cached [gs, gs] torch tensor
    (built from the same Kronecker power the loader's numpy decode uses). Used
    by the deferred path to rotate the *activation* on the device."""
    key = (gs, device, dtype)
    h = _HADAMARD_CACHE.get(key)
    if h is None:
        h = torch.from_numpy(_convrot_hadamard_gs(gs)).to(device=device, dtype=dtype)
        _HADAMARD_CACHE[key] = h
    return h


def rotate_activation(x: Tensor, gs: int) -> Tensor:
    """Fold the ConvRot block-Hadamard into the activation: `x_rot = x @
    H_block`, applying `H_gs` to each group of `gs` along the last dim via a
    batched matmul with the cached [gs, gs] matrix (H_gs is symmetric, so
    `x @ H_gs` is exactly the rotate the loader applies to the weight)."""
    s, inp = x.shape
    ng = inp // gs
    h = hadamard_matrix(gs, x.device, x.dtype)
    xr = x.reshape(s, ng, gs) @ h  # [S, ng, gs]
    return xr.reshape(s, inp)


class SafetensorsSource(TensorSource):
    def __init__(self, path, device="cpu", compute_dtype: torch.dtype = torch.float32):
        from safetensors import safe_open

        self._sf = safe_open(str(path), framework="pt", device="cpu")
        self._orig: Dict[str, str] = {}  # stripped name -> file key
        self._meta: Dict[str, Tuple[Tuple[int, ...], str]] = {}  # stripped -> (shape, dtype)
        for k in self._sf.keys():
            sk = strip_prefix(k)
            if sk in self._orig:
                raise ValueError(f"duplicate stripped key {sk}")
            self._orig[sk] = k
            sl = self._sf.get_slice(k)
            self._meta[sk] = (tuple(sl.get_shape()), str(sl.get_dtype()))
        self._device = device
        self._compute_dtype = compute_dtype
        self.used: set = set()
        self.missing: List[str] = []
        self.shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        self._cache: Dict[str, Tensor] = {}  # decoded / cast tensors

    def _is_int8(self, name: str) -> bool:
        m = self._meta.get(name)
        return m is not None and m[1].upper() in ("I8", "INT8")

    def _companion(self, name: str) -> str:
        """`X.weight` -> `X.weight_scale` — the trailing `.weight` is
        *replaced*, not extended (mirrors the Rust loader)."""
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        return base + ".weight_scale"

    def _load(self, name: str) -> Tensor:
        if name in self._cache:
            return self._cache[name]
        t = self._sf.get_tensor(self._orig[name])
        self._cache[name] = t
        return t

    def report(self):
        unused = sorted(k for k in self._meta if k not in self.used and not is_companion_key(k))
        return self.missing, self.shape_mismatch, unused

    def get(self, name: str, shape, dtype: torch.dtype) -> Tensor:
        if name not in self._meta:
            # Record and synthesize zeros instead of aborting, so a single
            # model construction pass collects *all* missing keys; callers
            # inspect `report()` and fail with the full list.
            self.missing.append(name)
            return torch.zeros(shape, dtype=dtype, device=self._device)
        cshape, _ = self._meta[name]
        if tuple(cshape) != tuple(shape):
            self.shape_mismatch.append((name, tuple(cshape), tuple(shape)))
        self.used.add(name)
        if self._is_int8(name):
            comp = self._companion(name)
            if comp in self._meta:
                self.used.add(comp)
                w = dequantize_convrot(self._load(name), self._load(comp))
                w = w.reshape(shape)
                return w.to(device=self._device, dtype=dtype)
        t = self._load(name)
        return t.to(device=self._device, dtype=dtype)

    def get_linear(self, name: str, shape, dtype: torch.dtype):
        # ConvRot INT8 linears are dequantized to a plain weight (no fp8 scale)
        # — the loader owns the decode, exactly like the Candle port.
        if self._is_int8(name) and self._companion(name) in self._meta:
            return self.get(name, shape, dtype), None
        return self.get(name, shape, dtype), None

    def get_deferred_int8(self, name: str, shape, dtype: torch.dtype):
        """Deferred ConvRot INT8 source: hands back the packed int8 + scale
        instead of the dequantized weight, so streamed runs keep ~21 GB of
        packed int8 in RAM (the checkpoint's own size) and dequantize each
        block on the GPU at transfer time — never materializing the 42 GB
        bf16 dequant (the pagefile thrash this repo hit) and shipping only
        1 byte/elem over PCIe instead of the 4-byte f32 dequant."""
        if not self._is_int8(name):
            return None
        comp = self._companion(name)
        if comp not in self._meta:
            return None
        cshape, _ = self._meta[name]
        if tuple(cshape) != tuple(shape):
            self.shape_mismatch.append((name, tuple(cshape), tuple(shape)))
        self.used.add(name)
        self.used.add(comp)
        q = self._load(name).to(self._device)  # keep int8
        scale = self._load(comp).to(self._device).reshape(-1)
        inp = shape[1]
        gs = convrot_group_size(inp)
        if gs is None:
            raise ValueError(f"in_features {inp} not divisible by any convrot group size")
        return DeferredInt8(q=q, scale=scale, gs=gs)

    def stored_shape(self, name: str) -> Optional[Tuple[int, ...]]:
        m = self._meta.get(name)
        return m[0] if m is not None else None

    def device(self):
        return self._device

    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype


def save_safetensors(path, tensors: Sequence[Tuple[str, Tensor]]) -> None:
    import safetensors.torch

    safetensors.torch.save_file(dict(tensors), str(path))


# ---------------------------------------------------------------------------
# Expected parameter table


def expected_params(cfg) -> List[Tuple[str, Tuple[int, ...], torch.dtype]]:
    out: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
    hidden = cfg.hidden_size
    inner = cfg.inner_dim()
    t_dim = cfg.time_embed_dim
    dtype = torch.bfloat16
    use_curves = cfg.use_adaln_curves()
    adaln_dtype = torch.float32 if use_curves else dtype

    def add(name: str, shape, dt: torch.dtype):
        out.append((name, tuple(shape), dt))

    add("video_patch_proj.weight", (hidden, cfg.video_patch_dim()), torch.float32)
    add("video_patch_proj.bias", (hidden,), torch.float32)
    add("audio_patch_proj.weight", (hidden, cfg.audio_latents_dim), torch.float32)
    add("audio_patch_proj.bias", (hidden,), torch.float32)
    add("condition_proj.weight", (hidden, cfg.text_dim), dtype)
    add("condition_proj.bias", (hidden,), dtype)
    if use_curves:
        add("adaln_t_table", (cfg.adaln_curve_grid, t_dim), torch.float32)
    else:
        add("time_embedder.proj_in.weight", (cfg.time_embed_hidden_size, cfg.timestep_input_dim), torch.float32)
        add("time_embedder.proj_in.bias", (cfg.time_embed_hidden_size,), torch.float32)
        add("time_embedder.proj_out.weight", (t_dim, cfg.time_embed_hidden_size), torch.float32)
        add("time_embedder.proj_out.bias", (t_dim,), torch.float32)
    add("rope.inv_freq", (cfg.rope_inv_freq_len,), torch.float32)

    for i in range(cfg.token_refiner_num_layers):
        p = f"token_refiner.blocks.{i}"
        add(f"{p}.norm1.weight", (hidden,), dtype)
        add(f"{p}.norm2.weight", (hidden,), dtype)
        add(f"{p}.attn.qkv_proj.weight", (inner * 3, hidden), dtype)
        add(f"{p}.attn.out_proj.weight", (hidden, inner), dtype)
        add(f"{p}.attn.q_norm.weight", (cfg.attention_head_dim,), dtype)
        add(f"{p}.attn.k_norm.weight", (cfg.attention_head_dim,), dtype)
        add(f"{p}.mlp.fc1.weight", (cfg.ffn_hidden_size * 2, hidden), dtype)
        add(f"{p}.mlp.fc2.weight", (hidden, cfg.ffn_hidden_size), dtype)
    add("token_refiner.final_norm.weight", (hidden,), dtype)

    for i in range(cfg.num_layers):
        p = f"blocks.{i}"
        add(f"{p}.norm1.weight", (hidden,), dtype)
        add(f"{p}.norm2.weight", (hidden,), dtype)
        add(f"{p}.attn.qkv_proj.weight", (inner * 3, hidden), dtype)
        add(f"{p}.attn.out_proj.weight", (hidden, inner), dtype)
        add(f"{p}.attn.q_norm.weight", (cfg.attention_head_dim,), dtype)
        add(f"{p}.attn.k_norm.weight", (cfg.attention_head_dim,), dtype)
        add(f"{p}.mlp.fc1.weight", (cfg.ffn_hidden_size * 2, hidden), dtype)
        add(f"{p}.mlp.fc2.weight", (hidden, cfg.ffn_hidden_size), dtype)
        add(f"{p}.adaln_proj.linear.weight", (6 * 3 * hidden, t_dim), adaln_dtype)
        add(f"{p}.adaln_proj.linear.bias", (6 * 3 * hidden,), adaln_dtype)

    add("final_layer.norm.weight", (hidden,), dtype)
    add("final_layer.adaln_proj.linear.weight", (2 * hidden, t_dim), adaln_dtype)
    add("final_layer.adaln_proj.linear.bias", (2 * hidden,), adaln_dtype)
    add("final_layer.video_out.weight", (cfg.video_patch_dim(), hidden), torch.float32)
    add("final_layer.video_out.bias", (cfg.video_patch_dim(),), torch.float32)
    add("final_layer.audio_out.weight", (cfg.audio_latents_dim, hidden), torch.float32)
    add("final_layer.audio_out.bias", (cfg.audio_latents_dim,), torch.float32)
    return out


def total_params(cfg) -> int:
    total = 0
    for _, shape, _ in expected_params(cfg):
        n = 1
        for s in shape:
            n *= s
        total += n
    return total


# ---------------------------------------------------------------------------
# fp8 (e4m3) weight quantization


E4M3_MAX = 448.0


def is_fp32_island(name: str) -> bool:
    """Weights kept at full precision under `--quantize`: the conditioning and
    decoding paths where fp8 rounding would visibly hurt fidelity."""
    return any(
        s in name
        for s in ("patch_proj", "time_embedder", "final_layer", "adaln", "rope", "inv_freq")
    )


def quantize_fp8_per_channel(t: Tensor):
    """Per-output-channel fp8 (e4m3) quantization of a [out, in] weight:
    `scale[o] = max_k |w[o,k]| / 448`, `w_q[o,k] = round(w[o,k] / scale[o])`
    clamped to [-448, 448], then stored as raw e4m3 bits (uint8). Dequant is
    `w ~= w_q * scale`, so the returned scale is exactly the vector
    `Linear::forward` folds into its matmul."""
    t32 = t.to(torch.float32)
    amax = t32.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)  # [out, 1]
    scale = amax / E4M3_MAX  # [out, 1]
    q = (t32 / scale).round().clamp(-E4M3_MAX, E4M3_MAX)
    bits = quantize_fp8_bits(q)
    return bits, scale.reshape(scale.shape[0])


class QuantizingSource(TensorSource):
    """Wraps any `TensorSource` and fp8-quantizes the big linear weights at
    load time. The fp32 island stays wide (see `is_fp32_island`)."""

    def __init__(self, inner: TensorSource):
        self.inner = inner

    def get(self, name: str, shape, dtype: torch.dtype) -> Tensor:
        return self.inner.get(name, shape, dtype)

    def get_linear(self, name: str, shape, dtype: torch.dtype):
        w, scale = self.inner.get_linear(name, shape, dtype)
        if scale is not None or len(shape) != 2 or is_fp32_island(name):
            return w, scale
        wq, s = quantize_fp8_per_channel(w)
        return wq, s.to(dtype)

    def get_deferred_int8(self, name: str, shape, dtype: torch.dtype):
        return self.inner.get_deferred_int8(name, shape, dtype)

    def stored_shape(self, name: str):
        return self.inner.stored_shape(name)

    def device(self):
        return self.inner.device()

    def compute_dtype(self) -> torch.dtype:
        return self.inner.compute_dtype()


def quantize_fp8_bits(q: Tensor) -> Tensor:
    """Round fp32 values (already clamped to [-448, 448]) to e4m3 bit patterns
    stored as uint8. e4m3 has 3 mantissa bits: round-to-nearest on the
    mantissa, handle subnormal underflow the same way a cast does.

    **Non-standard e4m3 decode** (kept in lockstep with the Candle port):
    exponent field 15 decodes as a finite value (2^(15-7) = 256) rather than
    NaN/Inf as in the IEEE e4m3 spec. That is consistent across the two
    runtimes, but once the real MiniMax weights drop, reconcile the codec
    with whatever Comfy's PR / the official inference code actually ships
    before trusting --quantize outputs."""
    s = (q < 0).to(torch.int32)
    a = q.abs()
    # exponent such that mantissa fits: e = floor(log2(a)); value = m * 2^(e-3)
    e = torch.floor(torch.log2(a.clamp_min(1e-30)))
    # mantissa (4 bits, 3 stored + implicit 1 for normals)
    mant = torch.floor(a / torch.pow(2.0, e - 3) + 0.5)
    # normal if a >= 2^-6 (the smallest normal is 2^-7 * 1.0); below is subnormal
    normal = a >= (2.0 ** -6.0)
    e_n = (e + 7).clamp(0, 15).to(torch.int32)
    e_z = torch.zeros_like(e_n)
    exp_part = torch.where(normal, e_n, e_z)
    mant_part = torch.where(normal, (mant - 8.0).clamp(0, 7), (a / (2.0 ** -9.0)).round().clamp(0, 7))
    bits = (s << 7) | (exp_part << 3) | mant_part.to(torch.int32)
    return bits.to(torch.uint8)


# ---------------------------------------------------------------------------
# AdaLN curve table + SVD pruning (load-time)


def ships_curve_table(src: TensorSource) -> Optional[int]:
    shape = src.stored_shape("adaln_t_table")
    return shape[0] if shape is not None else None


def resolve_curve_grid(src: TensorSource, cfg) -> int:
    return ships_curve_table(src) or cfg.curve_grid()


def linspace_tensor(n: int, device="cpu") -> Tensor:
    return torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=device)


def lerp_curve(table: Tensor, t_vals: Tensor) -> Tensor:
    """Piecewise-linear evaluation of the curve table at `t_vals` in [0, 1]:
    `lerp(table[i0], table[i0+1], frac)` with i0 clamped so t=1.0 stays on the
    last interval."""
    grid = table.shape[0]
    pos = t_vals.clamp(0.0, 1.0) * (grid - 1)
    i0 = torch.floor(pos).long().clamp(0, grid - 2)
    i1 = i0 + 1
    frac = (pos - i0.float()).unsqueeze(1)
    t0 = table[i0]
    t1 = table[i1]
    return t0 + (t1 - t0) * frac


def resolve_curve_table(src: TensorSource, cfg) -> Tensor:
    """The full-width (pre-projection) curve used by both the model and the SVD
    basis, so they always share one source: the checkpoint's stored table at
    its stored grid (format probe — wins even when the config doesn't declare
    curves), else the config-declared table, else a synthesis from the classic
    time embedder. Returns [grid, time_embed_dim] fp32."""
    t_dim = cfg.time_embed_dim
    grid = ships_curve_table(src)
    if grid is not None:
        return src.get("adaln_t_table", (grid, t_dim), torch.float32)
    if cfg.use_adaln_curves():
        return src.get("adaln_t_table", (cfg.curve_grid(), t_dim), torch.float32)
    return sample_curve_at(src, cfg, cfg.curve_grid())


def sample_curve(src: TensorSource, cfg, n: int) -> Tensor:
    """Sample the (post-silu) timestep curve at `n` points in [0, 1], for the
    SVD basis: the resolved table lerp-densified to `n` rows."""
    t_vals = linspace_tensor(n, src.device())
    table = resolve_curve_table(src, cfg)
    return lerp_curve(table, t_vals)


def sample_curve_at(src: TensorSource, cfg, grid: int) -> Tensor:
    """Post-silu curve samples at `grid` evenly spaced timesteps (classic
    checkpoints; curve checkpoints use the shipped table instead)."""
    from .timestep import TimeEmbedder

    te = TimeEmbedder(src, cfg.timestep_input_dim, cfg.time_embed_hidden_size, cfg.time_embed_dim)
    t = linspace_tensor(grid, src.device())
    return torch.nn.functional.silu(te.forward(t))


def top_r_svd_basis(curve: Tensor, rank: int, iters: int = 100) -> Tensor:
    """Top-`rank` right singular vectors of `curve` ([n, d]) via power
    iteration on the Gram matrix C^T C, with Gram-Schmidt orthogonalization
    against previously converged vectors, plus a final re-orthonormalization
    pass (V^T V = I up to fp32). Mirrors `top_r_svd_basis` in src/weights.rs."""
    n, d = curve.shape
    rank = max(1, min(rank, d, n))
    g = curve.t() @ curve  # [d, d]
    eps = torch.tensor(1e-12, dtype=curve.dtype, device=curve.device)
    basis: List[Tensor] = []
    for k in range(rank):
        v = _randn_std_normal((d, 1), 9000 + k, device=curve.device).squeeze(1).to(curve.dtype)
        for _ in range(iters):
            gv = g @ v.unsqueeze(1)
            v_new = gv.squeeze(1) / (gv.squeeze(1).pow(2).sum().sqrt() + eps)
            for b in basis:
                v_new = v_new - b * (v_new * b).sum()
            v_new = v_new / (v_new.pow(2).sum().sqrt() + eps)
            v = v_new
        basis.append(v)
    # final re-orthonormalization
    ortho: List[Tensor] = []
    for v in basis:
        u = v.clone()
        for b in ortho:
            u = u - b * (u * b).sum()
        ortho.append(u / (u.pow(2).sum().sqrt() + eps))
    return torch.stack(ortho, dim=1)  # [d, rank]


class AdalnSvdSource(TensorSource):
    """Load-time SVD pruning of the per-block adaln projections: `get_linear`
    for `*.adaln_proj.linear.weight` returns `W @ V` ([out, rank]) instead of
    the full [out, t_dim]; `get("adaln_t_table")` returns the curve
    coefficients `table @ V` ([grid, rank])."""

    def __init__(self, inner: TensorSource, cfg):
        rank = cfg.adaln_svd_rank
        if rank is None or rank < 1:
            raise ValueError(f"AdalnSvdSource requires adaln_svd_rank >= 1, got {rank}")
        self.inner = inner
        self.cfg = cfg
        self.rank = min(rank, cfg.time_embed_dim)
        self._basis = None

    @staticmethod
    def is_adaln(name: str) -> bool:
        return "adaln_proj.linear.weight" in name

    def basis(self) -> Tensor:
        if self._basis is None:
            curve = sample_curve(self.inner, self.cfg, 512)  # [512, t_dim]
            self._basis = top_r_svd_basis(curve, self.rank, 100)
        return self._basis

    def get(self, name: str, shape, dtype: torch.dtype) -> Tensor:
        if name == "adaln_t_table":
            v = self.basis()  # [t_dim, rank]
            full = resolve_curve_table(self.inner, self.cfg)
            return full @ v  # [grid, rank]
        return self.inner.get(name, shape, dtype)

    def get_linear(self, name: str, shape, dtype: torch.dtype):
        if self.is_adaln(name):
            out = shape[0]
            t_dim = self.cfg.time_embed_dim
            w, _ = self.inner.get_linear(name, (out, t_dim), torch.float32)
            v = self.basis()  # [t_dim, rank]
            return w.to(torch.float32) @ v, None  # [out, rank]
        return self.inner.get_linear(name, shape, dtype)

    def get_deferred_int8(self, name: str, shape, dtype: torch.dtype):
        # AdaLN is never deferred-quantized here; pass through for other linears.
        if self.is_adaln(name):
            return None
        return self.inner.get_deferred_int8(name, shape, dtype)

    def stored_shape(self, name: str):
        return self.inner.stored_shape(name)

    def device(self):
        return self.inner.device()

    def compute_dtype(self) -> torch.dtype:
        return self.inner.compute_dtype()


# ---------------------------------------------------------------------------
# footprints (mirror the Candle runtime's reporting, for cross-checking)


def _elem(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def quantized_footprint(cfg):
    """Bytes the checkpoint occupies under --quantize (and --adaln-svd-rank
    when set). Returns (total, fp32-island) bytes, mirroring
    `quantized_footprint` in src/weights.rs."""
    rank = cfg.adaln_svd_rank
    total = 0.0
    island = 0.0
    saw_table = False
    for name, shape, dt in expected_params(cfg):
        pruned_adaln = rank is not None and len(shape) == 2 and name.endswith("adaln_proj.linear.weight")
        is_table = name == "adaln_t_table"
        if is_table:
            saw_table = True
        el = float(_elem(shape))
        if rank is not None and (pruned_adaln or is_table):
            el = float(shape[0]) * rank
        if is_fp32_island(name) or pruned_adaln:
            b = 4.0 if pruned_adaln else _dtype_bytes(dt)
            total += b * el
            island += b * el
        elif len(shape) == 2:
            total += el + shape[0] * 4.0
        else:
            total += _dtype_bytes(dt) * el
    if rank is not None and not saw_table:
        t = float(cfg.curve_grid()) * rank * 4.0
        total += t
        island += t
    return total, island


def _dtype_bytes(dt: torch.dtype) -> float:
    return {torch.float64: 8.0, torch.float32: 4.0, torch.float16: 2.0, torch.bfloat16: 2.0}.get(dt, 4.0)


def fmt_footprint(cfg) -> str:
    total, island = quantized_footprint(cfg)
    fp32 = total_params(cfg) * 4.0

    def fmt(b):
        return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"

    svd = f"; adaln SVD-pruned to rank {cfg.adaln_svd_rank}" if cfg.adaln_svd_rank else ""
    return (
        f"{fmt(total)} total ({fmt(island)} fp32 island; {total/fp32*100:.0f}% of fp32 {fmt(fp32)}{svd})"
    )


__all__ = [
    "TensorSource",
    "RandomSource",
    "SafetensorsSource",
    "QuantizingSource",
    "AdalnSvdSource",
    "strip_prefix",
    "save_safetensors",
    "expected_params",
    "total_params",
    "is_fp32_island",
    "quantize_fp8_per_channel",
    "ships_curve_table",
    "resolve_curve_grid",
    "resolve_curve_table",
    "sample_curve",
    "lerp_curve",
    "top_r_svd_basis",
    "quantized_footprint",
    "fmt_footprint",
    "linspace_tensor",
    "randn_std_normal",
    "DeferredInt8",
    "hadamard_matrix",
    "rotate_activation",
]
