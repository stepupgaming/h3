"""Lossless optimization unit checks (CPU, tiny tensors — no real weights)."""

from __future__ import annotations

import torch

from minimax_h3.blocks import Linear, mod_gate, mod_scale_shift
from minimax_h3.config import H3Config
from minimax_h3.model import MiniMaxH3Model
from minimax_h3.weights import (
    DeferredInt8,
    RandomSource,
)


def test_mod_scale_shift_matches_reference_cat():
    torch.manual_seed(0)
    h = torch.randn(12, 8)
    shift = torch.randn(4, 8)
    scale = torch.randn(4, 8)
    segments = [(0, 3, 0), (3, 7, 2), (7, 12, 1)]

    # Legacy cat path
    parts = []
    for a, b, row in segments:
        sc = scale[row].reshape(8).to(h.dtype)
        sh = shift[row].reshape(8).to(h.dtype)
        parts.append(h[a:b] * (1.0 + sc) + sh)
    ref = torch.cat(parts, dim=0)

    out = mod_scale_shift(h, shift, scale, segments)
    assert torch.allclose(out, ref, rtol=0, atol=0)


def test_mod_gate_matches_reference_cat():
    torch.manual_seed(1)
    x = torch.randn(12, 8)
    other = torch.randn(12, 8)
    gate = torch.randn(4, 8)
    segments = [(0, 3, 0), (3, 7, 2), (7, 12, 1)]

    parts = []
    for a, b, row in segments:
        g = gate[row].reshape(8).to(x.dtype)
        parts.append(x[a:b] + other[a:b] * g)
    ref = torch.cat(parts, dim=0)

    x_in = x.clone()
    out = mod_gate(x_in, gate, other, segments)
    assert out is x_in  # in-place
    assert torch.allclose(out, ref, rtol=0, atol=0)


def test_mod_with_precast_stream_dtype():
    """Precast AdaLN rows to stream dtype must match per-segment cast."""
    torch.manual_seed(2)
    h = torch.randn(16, 8, dtype=torch.bfloat16)
    shift_f32 = torch.randn(3, 8, dtype=torch.float32)
    scale_f32 = torch.randn(3, 8, dtype=torch.float32)
    gate_f32 = torch.randn(3, 8, dtype=torch.float32)
    other = torch.randn(16, 8, dtype=torch.bfloat16)
    segments = [(0, 5, 0), (5, 11, 2), (11, 16, 1)]

    ref_ss = mod_scale_shift(h.clone(), shift_f32, scale_f32, segments)
    pre_shift = shift_f32.to(dtype=torch.bfloat16)
    pre_scale = scale_f32.to(dtype=torch.bfloat16)
    out_ss = mod_scale_shift(h.clone(), pre_shift, pre_scale, segments)
    assert torch.allclose(out_ss.float(), ref_ss.float(), rtol=0, atol=0)

    x0 = torch.randn(16, 8, dtype=torch.bfloat16)
    ref_g = mod_gate(x0.clone(), gate_f32, other, segments)
    pre_gate = gate_f32.to(dtype=torch.bfloat16)
    out_g = mod_gate(x0.clone(), pre_gate, other, segments)
    assert torch.allclose(out_g.float(), ref_g.float(), rtol=0, atol=0)


def test_mlp_swiglu_inplace_matches():
    from minimax_h3.blocks import MLP

    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=7)
    mlp = MLP(src, "t.mlp", cfg.hidden_size, cfg.ffn_hidden_size, torch.float32)
    x = torch.randn(5, cfg.hidden_size)
    y = mlp.forward(x)
    # Recompute with explicit silu (non-inplace path)
    y1 = mlp.fc1.forward(x)
    ffn = mlp.ffn
    gate, up = y1[:, :ffn].clone(), y1[:, ffn:].clone()
    ref = mlp.fc2.forward(torch.nn.functional.silu(gate) * up)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-5)


def test_prepare_text_states_matches_embed_path():
    """prepare_text_states must match forward's condition_proj+refiner path."""
    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=11)
    model = MiniMaxH3Model(src, cfg, "eager")
    model.to("cpu")
    torch.manual_seed(3)
    text = torch.randn(7, cfg.text_dim)

    prepared = model.prepare_text_states(text, stream_gpu=None)
    assert prepared.shape == (7, cfg.hidden_size)

    # Manual embed path (same ops as model.forward text branch).
    proj = model.condition_proj.forward(text.to(model.compute_dtype))
    # Refiner blocks may have been moved to CPU after prepare; ensure on CPU.
    model.token_refiner.to("cpu")
    ref = model.token_refiner.forward(proj)
    assert torch.allclose(prepared, ref, rtol=1e-5, atol=1e-5)

    # Idempotent when already hidden-sized.
    again = model.prepare_text_states(prepared, stream_gpu=None)
    assert torch.allclose(again, prepared.to(model.compute_dtype), rtol=0, atol=0)


def test_deferred_int8_cast_backend_default():
    """Default cast path is deterministic; backend flag restores to cast."""
    torch.manual_seed(5)
    # gs must be a power of 4 (convrot: 4/16/64/256).
    out, inp, gs = 32, 64, 64
    q = torch.randint(-20, 20, (out, inp), dtype=torch.int8)
    scale = torch.linspace(0.01, 0.05, out)
    lin = Linear(None, None, None, deferred_int8=DeferredInt8(q=q, scale=scale, gs=gs))
    lin.to("cpu")
    x = torch.randn(4, inp)
    Linear.deferred_int8_backend = "cast"
    y1 = lin.forward(x)
    y2 = lin.forward(x)
    assert torch.allclose(y1, y2, rtol=0, atol=0)
    assert Linear.deferred_int8_backend == "cast"




if __name__ == "__main__":
    test_mod_scale_shift_matches_reference_cat()
    test_mod_gate_matches_reference_cat()
    test_mod_with_precast_stream_dtype()
    test_mlp_swiglu_inplace_matches()
    test_prepare_text_states_matches_embed_path()
    test_deferred_int8_cast_backend_default()
    print("OK: opt lossless unit checks")
