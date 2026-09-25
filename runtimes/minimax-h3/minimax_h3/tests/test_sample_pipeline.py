"""Unit tests for frame_grid, conditioning plan, layout cache, sample helpers."""

from __future__ import annotations

import torch

from minimax_h3.conditioning import ConditioningError, ConditioningPlan, build_conditioning
from minimax_h3.config import H3Config
from minimax_h3.frame_grid import (
    flow_sigmas,
    floor_frames,
    latent_t_from_frames,
    snap_frames,
    temporal_shape,
)
from minimax_h3.layout import Keyframe, RefBlock, RefKind
from minimax_h3.model import MiniMaxH3Model, Payload
from minimax_h3.weights import RandomSource


def test_snap_frames_17k_plus_5():
    assert snap_frames(1) == 5
    assert snap_frames(5) == 5
    assert snap_frames(6) == 22
    assert snap_frames(39) == 39
    assert snap_frames(40) == 56
    assert floor_frames(56) == 56
    assert floor_frames(55) == 39


def test_temporal_shape_matches_cli_formula():
    ts = temporal_shape(39)
    assert ts.frame_count == 39
    assert ts.latent_t == latent_t_from_frames(39)
    assert ts.latent_t == ((39 - 5) // 17) * 5 + 2
    assert ts.audio_t == round(39 / 24.0 * 40.0)
    assert abs(ts.duration_s - 39 / 24.0) < 1e-9


def test_flow_sigmas_length_and_endpoints():
    s = flow_sigmas(20, 12.0)
    assert len(s) == 21
    assert s[-1] == 0.0
    assert s[0] > s[-2] > 0.0


def test_build_conditioning_t2va_empty():
    cfg = H3Config.tiny()
    plan = build_conditioning(cfg=cfg, frame_count=39, lat_h=4, lat_w=4)
    assert plan.keyframes == []
    assert plan.refs == []
    p = plan.to_payload(seed=0, sigma_shift_video=12.0, sigma_shift_audio=3.0)
    assert p.keyframes == []
    assert p.frame_count is None


def test_build_conditioning_rejects_mixed_modes():
    cfg = H3Config.tiny()
    # Use dummy paths that fail as raw media so we never need real files for
    # the mutual-exclusion check (raises before load when both set).
    try:
        build_conditioning(
            cfg=cfg,
            frame_count=39,
            lat_h=4,
            lat_w=4,
            first_frame="x.png",
            ref_specs=[("image", "y.png")],
            allow_keyframe_refs=False,
        )
        assert False, "expected ConditioningError"
    except ConditioningError as e:
        assert "cannot combine" in str(e)


def test_conditioning_plan_payload_frame_count_only_with_keyframes():
    plan = ConditioningPlan(keyframes=[Keyframe(0)], frame_count=39)
    p = plan.to_payload(seed=1, sigma_shift_video=12.0, sigma_shift_audio=3.0)
    assert p.frame_count == 39
    plan2 = ConditioningPlan()
    p2 = plan2.to_payload(seed=1, sigma_shift_video=12.0, sigma_shift_audio=3.0)
    assert p2.frame_count is None


def test_model_layout_cache_distinguishes_ref_shapes():
    """Same ref *count* but different shapes must not reuse layout."""
    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=0)
    model = MiniMaxH3Model(src, cfg, "eager")
    # Two IMAGE refs, same count, different spatial size.
    p1 = Payload(
        refs=[RefBlock(RefKind.IMAGE, 0, 2, 2, 0)],
        seed=0,
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    p2 = Payload(
        refs=[RefBlock(RefKind.IMAGE, 0, 4, 4, 0)],
        seed=0,
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    l1 = model.layout(8, 2, 4, 4, 4, p1)
    l2 = model.layout(8, 2, 4, 4, 4, p2)
    assert l1.signature != l2.signature
    assert l1 is not l2
    # Same payload again hits cache.
    l1b = model.layout(8, 2, 4, 4, 4, p1)
    assert l1b is l1


def test_euler_denoise_runs_tiny_cpu():
    from minimax_h3.sample import euler_denoise

    cfg = H3Config.tiny()
    src = RandomSource("cpu", torch.float32, seed=1)
    model = MiniMaxH3Model(src, cfg, "eager")
    model.to("cpu")
    video = torch.randn(1, cfg.latents_dim, 2, 2, 2)
    audio = torch.randn(1, cfg.audio_latents_dim, 2, 4)
    text = torch.randn(4, cfg.text_dim)
    payload = Payload(
        seed=0,
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    sigmas = flow_sigmas(2, cfg.sigma_shift_video)
    v2, a2 = euler_denoise(
        model,
        video=video,
        audio=audio,
        text=text,
        payload=payload,
        sigmas=sigmas,
        stream_gpu=None,
        precompute_adaln=False,
        log=False,
    )
    assert v2.shape == video.shape
    assert a2.shape == audio.shape
    assert torch.isfinite(v2).all()
    assert torch.isfinite(a2).all()
