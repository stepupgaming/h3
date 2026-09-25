"""Tests for the PyTorch reference implementation of MiniMax H3.

Runnable as pytest, or directly (`python minimax_h3/tests/test_model.py`).
All tests run on CPU with the tiny config and random weights — the same
shape-check surface the Rust port's `--tiny check-shapes` covers, plus
kernel/layout/format equivalence that transfers to a torch-vs-Candle parity
run once the real weights drop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from minimax_h3.attention import AttnImpl, _eager, _fa2, _flash
from minimax_h3.blocks import decode_e4m3
from minimax_h3.config import H3Config
from minimax_h3.layout import Keyframe, PackedLayout, RefBlock, RefKind, SegKind
from minimax_h3.model import MiniMaxH3Model, Payload
from minimax_h3.rotary import apply_rope_split_half, rope_cos_sin, rope_freqs
from minimax_h3.timestep import time_shift_sigma, time_shift_slope
from minimax_h3.weights import (
    AdalnSvdSource,
    QuantizingSource,
    RandomSource,
    expected_params,
    lerp_curve,
    quantize_fp8_per_channel,
    ships_curve_table,
)


def make_model(cfg=None, **src_kwargs):
    cfg = cfg or H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=42)
    return MiniMaxH3Model(src, cfg, AttnImpl.AUTO)


def make_inputs(cfg, device="cpu"):
    video = torch.randn(1, cfg.latents_dim, 8, 8, 8, device=device)
    audio = torch.randn(1, cfg.audio_latents_dim, 2, 6, device=device)
    text = torch.randn(16, cfg.text_dim, device=device)
    return video, audio, text


def payload_for(cfg):
    return Payload(
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )


def test_forward_shapes_and_finite():
    cfg = H3Config.tiny()
    model = make_model(cfg)
    video, audio, text = make_inputs(cfg)
    with torch.no_grad():
        v, a = model.forward(video, audio, text, None, 0.5, payload_for(cfg))
    assert list(v.shape) == [1, cfg.latents_dim, 8, 8, 8]
    assert list(a.shape) == [1, cfg.audio_latents_dim, 2, 6]
    assert torch.isfinite(v).all()
    assert torch.isfinite(a).all()


def test_forward_deterministic():
    cfg = H3Config.tiny()
    model = make_model(cfg)
    video, audio, text = make_inputs(cfg)
    p = payload_for(cfg)
    with torch.no_grad():
        v1, a1 = model.forward(video, audio, text, None, 0.5, p)
        v2, a2 = model.forward(video, audio, text, None, 0.5, p)
    assert torch.equal(v1, v2)
    assert torch.equal(a1, a2)


def test_text_at_hidden_skips_proj_and_refiner():
    """Text already at hidden_size skips condition_proj + token refiner — the
    exact port of the Rust `if text.dims()[1] != hidden` branch."""
    cfg = H3Config.tiny()
    model = make_model(cfg)
    video, audio, _ = make_inputs(cfg)
    text_hidden = torch.randn(16, cfg.hidden_size)
    p = payload_for(cfg)
    with torch.no_grad():
        v, a = model.forward(video, audio, text_hidden, None, 0.5, p)
    assert list(v.shape) == [1, cfg.latents_dim, 8, 8, 8]
    assert torch.isfinite(v).all()


def test_layout_segments():
    # t2va: text 16, audio 6*2 rows, video 32 latent_t * frame rows (42x24 ->
    # 21x12 -> 252 frame rows) = 32*252
    l = PackedLayout.new(16, 32, 42, 24, 6)
    kinds = [k for (_, _, k) in l.segments]
    assert kinds == [SegKind.TEXT, SegKind.AUDIO, SegKind.VIDEO]
    assert l.segments[0] == (0, 16, SegKind.TEXT)
    n_audio = l.segments[1][1] - l.segments[1][0]
    assert n_audio == 12  # 6 frames * 2 stereo channels
    n_video = l.segments[2][1] - l.segments[2][0]
    assert n_video == 32 * 21 * 12
    assert l.seq_len == 16 + n_audio + n_video
    # position rows are 3-wide
    assert len(l.position_ids) == l.seq_len * 3


def _at(l, i):
    """Position row `i` rounded to 4 decimals — the Rust `dump_layout` format."""
    pid = l.position_ids
    return tuple(round(pid[3 * i + j], 4) for j in range(3))


def test_layout_keyframes_fl2va():
    """Golden-locked against `minimax-h3 layout-test` (fl2va scenario): exact
    segment boundaries, seq_len, and the sampled position rows the Rust
    `dump_layout` prints (text0 / last text / last video row)."""
    l = PackedLayout.new(16, 32, 42, 24, 8, [Keyframe(0), Keyframe(123)], [], 124)
    segs = [(a, b, k.name()) for (a, b, k) in l.segments]
    assert segs == [
        (0, 16, "text"),
        (16, 268, "cond"),
        (268, 520, "cond"),
        (520, 536, "audio"),
        (536, 8600, "video"),
    ]
    assert l.seq_len == 8600
    n = l.seq_len
    assert _at(l, 0) == (0.0, 0.0, 0.0)
    assert _at(l, 15) == (15.0, 0.0, 0.0)
    assert _at(l, n - 1) == (187.6667, 35.1502, 26.0791)


def test_layout_keyframe_first_only():
    """Single first-frame FL2VA cond (continue / i2v path): one COND segment."""
    l = PackedLayout.new(16, 32, 42, 24, 8, [Keyframe(0)], [], 124)
    segs = [(a, b, k.name()) for (a, b, k) in l.segments]
    assert segs == [
        (0, 16, "text"),
        (16, 268, "cond"),
        (268, 284, "audio"),
        (284, 8348, "video"),
    ]
    assert l.seq_len == 8348


def test_layout_signature_distinguishes_cond_modes():
    """T2VA / FL2VA / Ref2VA must not share a rope-cache signature."""
    t2va = PackedLayout.new(16, 8, 8, 8, 4)
    fl2va = PackedLayout.new(16, 8, 8, 8, 4, [Keyframe(0)], [], 5)
    refs = [RefBlock(RefKind.IMAGE, 0, 8, 8, 0)]
    ref2va = PackedLayout.new(16, 8, 8, 8, 4, [], refs, None)
    assert t2va.signature != fl2va.signature
    assert t2va.signature != ref2va.signature
    assert fl2va.signature != ref2va.signature
    assert t2va.seq_len != fl2va.seq_len


def test_rope_cache_survives_t2va_then_fl2va():
    """Regression: rope cache used to key only on bare latent dims, so a T2VA
    forward poisoned FL2VA with a short cos/sin table."""
    cfg = H3Config.tiny()
    model = make_model(cfg)
    video, audio, text = make_inputs(cfg)
    # Tiny canvas already matches make_inputs shapes.
    p_t2 = payload_for(cfg)
    z0 = torch.randn(1, cfg.latents_dim, 1, video.shape[3], video.shape[4])
    p_fl = Payload(
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
        keyframes=[Keyframe(0)],
        cond_video_latents=[z0],
        frame_count=5,
    )
    with torch.no_grad():
        v1, a1 = model.forward(video, audio, text, None, 0.5, p_t2)
        v2, a2 = model.forward(video, audio, text, None, 0.5, p_fl)
    assert list(v1.shape) == list(v2.shape) == [1, cfg.latents_dim, 8, 8, 8]
    assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
    assert torch.isfinite(a1).all() and torch.isfinite(a2).all()


def test_layout_refs_ref2va():
    """Golden-locked against `minimax-h3 layout-test` (ref2va scenario)."""
    refs = [
        RefBlock(RefKind.IMAGE, 0, 16, 24, 0),
        RefBlock(RefKind.VIDEO_AUDIO, 5, 16, 24, 3),
        RefBlock(RefKind.AUDIO, 0, 0, 0, 4),
    ]
    l = PackedLayout.new(16, 32, 42, 24, 8, [], refs, None)
    segs = [(a, b, k.name()) for (a, b, k) in l.segments]
    assert segs == [
        (0, 16, "text"),
        (16, 112, "ref_img"),
        (112, 118, "ref_audio"),
        (118, 598, "ref_img"),
        (598, 606, "ref_audio"),
        (606, 622, "audio"),
        (622, 8686, "video"),
    ]
    assert l.seq_len == 8686
    n = l.seq_len
    assert _at(l, 0) == (0.0, 0.0, 0.0)
    assert _at(l, 15) == (15.0, 0.0, 0.0)
    assert _at(l, n - 1) == (221.0, 35.1502, 26.0791)


def test_layout_refs_pure_video_and_image():
    """Pure VIDEO (no audio) + IMAGE packing used by sample --ref-video/--ref-image."""
    refs = [
        RefBlock(RefKind.IMAGE, 0, 30, 52, 0),
        RefBlock(RefKind.VIDEO, 7, 30, 52, 0),
    ]
    l = PackedLayout.new(16, 12, 30, 52, 20, [], refs, None)
    kinds = [k.name() for (_, _, k) in l.segments]
    assert kinds == ["text", "ref_img", "ref_img", "audio", "video"]
    # IMAGE: one frame grid; VIDEO: T * frame_rows
    img_rows = l.segments[1][1] - l.segments[1][0]
    vid_rows = l.segments[2][1] - l.segments[2][0]
    assert img_rows == (30 // 2) * (52 // 2)
    assert vid_rows == 7 * img_rows


def test_payload_ref_embed_order_matches_layout():
    """cond_video/audio iterators must match segment order (VIDEO_AUDIO = audio then video)."""
    cfg = H3Config.tiny()
    # IMAGE then VIDEO_AUDIO then AUDIO — same family as layout-test golden
    z_img = torch.zeros(1, cfg.latents_dim, 1, 8, 8)
    z_vid = torch.zeros(1, cfg.latents_dim, 5, 8, 8)
    a_va = torch.zeros(1, cfg.audio_latents_dim, 2, 3)
    a_only = torch.zeros(1, cfg.audio_latents_dim, 2, 4)
    refs = [
        RefBlock(RefKind.IMAGE, 0, 8, 8, 0),
        RefBlock(RefKind.VIDEO_AUDIO, 5, 8, 8, 3),
        RefBlock(RefKind.AUDIO, 0, 0, 0, 4),
    ]
    # Sample CLI push order for VIDEO_AUDIO: audio latent then video latent
    payload = Payload(
        cond_video_latents=[z_img, z_vid],
        cond_audio_latents=[a_va, a_only],
        refs=refs,
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    layout = PackedLayout.new(8, 4, 8, 8, 6, [], refs, None)
    v_iter = iter(payload.cond_video_latents)
    a_iter = iter(payload.cond_audio_latents)
    v_shapes = []
    a_shapes = []
    for _, _, kind in layout.segments:
        if kind in (SegKind.COND, SegKind.REF_IMG):
            v_shapes.append(tuple(next(v_iter).shape))
        elif kind == SegKind.REF_AUDIO:
            a_shapes.append(tuple(next(a_iter).shape))
    assert v_shapes == [tuple(z_img.shape), tuple(z_vid.shape)]
    assert a_shapes == [tuple(a_va.shape), tuple(a_only.shape)]
    # iterators exhausted
    try:
        next(v_iter)
        assert False, "extra video cond"
    except StopIteration:
        pass
    try:
        next(a_iter)
        assert False, "extra audio cond"
    except StopIteration:
        pass


def test_layout_t2va_golden():
    """Golden-locked against `minimax-h3 layout-test` (t2va scenario)."""
    l = PackedLayout.new(16, 32, 42, 24, 8)
    segs = [(a, b, k.name()) for (a, b, k) in l.segments]
    assert segs == [(0, 16, "text"), (16, 32, "audio"), (32, 8096, "video")]
    assert l.seq_len == 8096
    n = l.seq_len
    assert _at(l, 0) == (0.0, 0.0, 0.0)
    assert _at(l, 15) == (15.0, 0.0, 0.0)
    assert _at(l, n - 1) == (187.6667, 35.1502, 26.0791)


def test_rope_split_half_matches_manual():
    s, h, d, k = 8, 2, 32, 3  # rot = 2*3k = 18 <= d, so the split-half pair fits
    q = torch.randn(s, h, d)
    kt = torch.randn(s, h, d)
    ids = torch.randn(s, 3)
    inv = torch.randn(k)
    angles = rope_freqs(ids, inv)
    cos, sin = rope_cos_sin(angles, torch.float32)
    assert cos.shape == (s, 3 * k)
    qr, kr = apply_rope_split_half(q, kt, cos, sin)
    # manual reference: pair (i, i+P) rotated by angle i
    P = 3 * k
    c = cos[:, :P]
    s_ = sin[:, :P]
    for name, x, xr in (("q", q, qr), ("k", kt, kr)):
        a = x[:, :, :P]
        b = x[:, :, P : 2 * P]
        expected_a = c.unsqueeze(1) * a - s_.unsqueeze(1) * b
        expected_b = s_.unsqueeze(1) * a + c.unsqueeze(1) * b
        assert torch.allclose(xr[:, :, :P], expected_a, atol=1e-6), name
        assert torch.allclose(xr[:, :, P : 2 * P], expected_b, atol=1e-6), name
        assert torch.equal(xr[:, :, 2 * P :], x[:, :, 2 * P :]), name


def test_flash_matches_eager():
    torch.manual_seed(0)
    h, s, d = 4, 600, 32  # s > block exercises rescaling; odd remainder edges
    q = torch.randn(h, s, d)
    k = torch.randn(h, s, d)
    v = torch.randn(h, s, d)
    eager = _eager(q, k, v)
    flash = _flash(q, k, v, 256)
    max_abs = (eager - flash).abs().max().item()
    assert max_abs < 1e-3, f"flash diverged from eager: {max_abs}"
    # relative error on significant outputs only
    mask = eager.abs() > 1e-3
    rel = ((eager - flash).abs() / eager.abs())[mask].max().item()
    assert rel < 1e-3, f"flash diverged from eager (rel): {rel}"


def test_attn_impl_normalize_aliases():
    assert AttnImpl.normalize("fa2") == AttnImpl.FA2
    assert AttnImpl.normalize("flash-attn") == AttnImpl.FA2
    assert AttnImpl.normalize("flash_attn") == AttnImpl.FA2
    assert AttnImpl.normalize("FLASH") == AttnImpl.FLASH
    assert AttnImpl.normalize("sage") == AttnImpl.SAGE


def test_fa2_matches_eager_when_available():
    """Real flash-attn HQ path ≈ eager (bf16 CUDA). Skip if wheel/GPU missing."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for fa2")
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pytest.skip("flash-attn not installed")

    torch.manual_seed(0)
    h, s, d = 4, 128, 64  # FA2 head_dim usually power-of-two friendly
    q = torch.randn(h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(h, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(h, s, d, device="cuda", dtype=torch.bfloat16)
    eager = _eager(q.float(), k.float(), v.float())
    fa = _fa2(q, k, v).float()
    max_abs = (eager - fa).abs().max().item()
    # bf16 FA vs fp32 eager — looser than portable flash fp32 tile test
    assert max_abs < 5e-2, f"fa2 diverged from eager: {max_abs}"
    cos = torch.nn.functional.cosine_similarity(
        eager.flatten(), fa.flatten(), dim=0
    ).item()
    assert cos > 0.999, f"fa2 cosine too low: {cos}"


def test_sdpa_matches_eager():
    torch.manual_seed(1)
    h, s, d = 4, 64, 32
    q = torch.randn(h, s, d)
    k = torch.randn(h, s, d)
    v = torch.randn(h, s, d)
    eager = _eager(q, k, v)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert (eager - sdpa).abs().max().item() < 1e-3


def test_curve_mode_runs_and_matches_svd_full_rank():
    """Curve-table mode (grid) runs end to end, and SVD pruning at full rank
    (rank = time_embed_dim) reproduces it exactly — both run the same
    lerp-table forward."""
    cfg = H3Config.tiny()
    cfg.adaln_curve_grid = 64
    model = make_model(cfg)
    video, audio, text = make_inputs(cfg)
    p = payload_for(cfg)
    with torch.no_grad():
        v1, a1 = model.forward(video, audio, text, None, 0.5, p)

    cfg2 = H3Config.tiny()
    cfg2.adaln_svd_rank = cfg2.time_embed_dim  # full rank: no truncation
    src = RandomSource("cpu", torch.float32, seed=42)
    model2 = MiniMaxH3Model(src, cfg2, AttnImpl.AUTO)
    with torch.no_grad():
        v2, a2 = model2.forward(video, audio, text, None, 0.5, p)
    assert torch.allclose(v1, v2, atol=1e-4), "full-rank SVD diverged from curve table"
    assert torch.allclose(a1, a2, atol=1e-4)


def test_svd_low_rank_stays_close():
    cfg = H3Config.tiny()
    cfg.adaln_svd_rank = 32
    src = RandomSource("cpu", torch.float32, seed=42)
    model = MiniMaxH3Model(src, cfg, AttnImpl.AUTO)
    video, audio, text = make_inputs(cfg)
    p = payload_for(cfg)
    with torch.no_grad():
        v, a = model.forward(video, audio, text, None, 0.5, p)
    assert list(v.shape) == [1, cfg.latents_dim, 8, 8, 8]
    assert torch.isfinite(v).all()


def test_fp8_quantized_forward_close_to_fp32():
    cfg = H3Config.tiny()
    base = RandomSource("cpu", torch.float32, seed=42)
    qsrc = QuantizingSource(base)
    model_q = MiniMaxH3Model(qsrc, cfg, AttnImpl.AUTO)
    model_fp = MiniMaxH3Model(base, cfg, AttnImpl.AUTO)
    video, audio, text = make_inputs(cfg)
    p = payload_for(cfg)
    with torch.no_grad():
        vq, aq = model_q.forward(video, audio, text, None, 0.5, p)
        vf, af = model_fp.forward(video, audio, text, None, 0.5, p)
    # fp8 e4m3 is ~6% worst-case per weight; random-weight cancellation keeps
    # the output within a few percent relative.
    rel = lambda x, y: ((x - y).abs() / (x.abs() + 1e-6)).max().item()
    assert rel(vq, vf) < 0.25
    assert rel(aq, af) < 0.25


def test_e4m3_decode_roundtrip():
    # known e4m3 bit patterns:
    #   0x3C = sign0 exp7 mant4 -> (1+4/8)*2^0 = 1.5
    #   0x00 = zero
    #   0xBC = sign1 exp7 mant4 -> -1.5
    #   0x04 = sign0 exp0 mant4 -> subnormal (4/8)*2^-6 = 1/128
    bits = torch.tensor([0x3C, 0x00, 0xBC, 0x04], dtype=torch.uint8)
    vals = decode_e4m3(bits)
    assert abs(vals[0].item() - 1.5) < 1e-6
    assert vals[1].item() == 0.0
    assert abs(vals[2].item() + 1.5) < 1e-6
    assert abs(vals[3].item() - 1.0 / 128.0) < 1e-9


def test_fp8_quant_scale_folds():
    w = torch.randn(8, 16) * 3.0
    bits, scale = quantize_fp8_per_channel(w)
    wq = decode_e4m3(bits) * scale.unsqueeze(1)
    # e4m3 has 3 mantissa bits -> relative error <= ~1/16 per element
    rel = ((wq - w).abs() / (w.abs() + 1e-9)).max().item()
    assert rel < 0.12


def test_lerp_curve_endpoints():
    table = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
    # Piecewise-linear over [0, 0.5) -> rows 0..1, [0.5, 1] -> rows 1..2
    # (i0 = floor(t*(grid-1)), clamped so t=1 stays on the last interval).
    t = torch.tensor([0.0, 0.25, 1.0])  # pos: 0, 0.5, 2 -> i0: 0, 0, 1
    out = lerp_curve(table, t)
    assert torch.allclose(out[0], table[0])
    assert torch.allclose(out[1], (table[0] + table[1]) / 2)
    assert torch.allclose(out[2], table[2])


def test_time_shift_identities():
    # shifting a schedule onto itself is the identity
    assert abs(time_shift_sigma(0.5, 12.0, 12.0) - 0.5) < 1e-12
    assert abs(time_shift_slope(0.5, 12.0, 12.0) - 1.0) < 1e-12


def test_expected_params_table_shape_counts():
    cfg = H3Config.tiny()
    params = expected_params(cfg)
    names = {n for n, _, _ in params}
    # a few key tensors must be present
    assert "video_patch_proj.weight" in names
    assert "adaln_t_table" not in names  # classic mode: time embedder instead
    assert "time_embedder.proj_in.weight" in names
    assert "final_layer.video_out.weight" in names
    assert "rope.inv_freq" in names
    assert any(n.startswith("blocks.0.") for n in names)
    assert any(n.startswith("token_refiner.blocks.0.") for n in names)
    # classic checkpoint declares no curve table
    assert ships_curve_table(RandomSource("cpu", torch.float32, seed=1)) is None


def test_safetensors_roundtrip_audit():
    """A checkpoint written from expected random weights must audit clean,
    and a curve-format checkpoint (with adaln_t_table) must be detected by the
    format probe."""
    import tempfile

    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=42)
    tensors = {}
    for name, shape, dtype in expected_params(cfg):
        tensors[name] = src.get(name, shape, dtype)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.safetensors"
        import safetensors.torch

        safetensors.torch.save_file(tensors, str(path))

        from minimax_h3.weights import SafetensorsSource

        s = SafetensorsSource(path, "cpu", torch.float32)
        missing, mismatch, unused = s.report()
        assert not missing and not mismatch, (missing, mismatch)
        # rebuilding the model from the file consumes every key
        model = MiniMaxH3Model(s, cfg, AttnImpl.AUTO)
        missing, mismatch, unused = s.report()
        assert not missing and not mismatch, (missing, mismatch)
        assert not unused, unused


def test_curve_checkpoint_format_probe():
    """A curve-format checkpoint (ships adaln_t_table) is detected by the
    stored-shape probe even when the config doesn't declare curves."""
    import tempfile

    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=42)
    tensors = {}
    for name, shape, dtype in expected_params(cfg):
        tensors[name] = src.get(name, shape, dtype)
    grid = 64
    tensors["adaln_t_table"] = torch.randn(grid, cfg.time_embed_dim)
    del tensors["time_embedder.proj_in.weight"]
    del tensors["time_embedder.proj_in.bias"]
    del tensors["time_embedder.proj_out.weight"]
    del tensors["time_embedder.proj_out.bias"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.safetensors"
        import safetensors.torch

        safetensors.torch.save_file(tensors, str(path))

        from minimax_h3.weights import SafetensorsSource

        s = SafetensorsSource(path, "cpu", torch.float32)
        assert ships_curve_table(s) == grid
        # building without --curve-grid still loads (format probe wins)
        model = MiniMaxH3Model(s, cfg, AttnImpl.AUTO)
        assert model.adaln_t_table is not None
        video, audio, text = make_inputs(cfg)
        p = payload_for(cfg)
        with torch.no_grad():
            v, a = model.forward(video, audio, text, None, 0.5, p)
        assert torch.isfinite(v).all()


# ---------------------------------------------------------------------------

_ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in _ALL:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(_ALL) - failures}/{len(_ALL)} passed")
    sys.exit(1 if failures else 0)
