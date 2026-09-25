"""Production sample pipeline — load, pack, denoise, save.

Deep library seam for owner-box / gemmy integration. CLI and scripts should be
thin adapters over this module rather than reimplementing Euler + packing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .attention import AttnImpl, get_sage_backend
from .canvas import CanvasSpec, resolve_canvas
from .conditioning import (
    ConditioningError,
    ConditioningPlan,
    RefSpec,
    build_conditioning,
    load_text_embeds,
)
from .config import H3Config
from .frame_grid import flow_sigmas, temporal_shape
from .model import MiniMaxH3Model, Payload
from .profile import PhaseProfiler, set_profiler
from .weights import SafetensorsSource, save_safetensors

PathLike = Union[str, Path]


@dataclass
class SampleRequest:
    """Inputs for one DiT sample run (latents out; no VAE)."""

    weights: PathLike
    text: PathLike
    out: PathLike
    cfg: H3Config
    steps: int = 20
    length: int = 39
    seed: int = 42
    device: str = "cuda"
    attn: str = AttnImpl.SAGE
    # Canvas: either a resolved CanvasSpec, or width/height, or resolve kwargs.
    canvas: Optional[CanvasSpec] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect: Optional[str] = None
    megapixels: Optional[float] = None
    short_edge: Optional[int] = None
    canvas_mode: str = "preview"
    # Cond
    first_frame: Optional[PathLike] = None
    last_frame: Optional[PathLike] = None
    ref_specs: Optional[Sequence[RefSpec]] = None
    allow_keyframe_refs: bool = False
    # Lossless speed defaults (owner box)
    prefetch: int = 1
    precompute_adaln: bool = True
    preproject_text: bool = True
    ram_reserve_gb: float = 8.0
    max_pin_gb: float = 4.0
    # Bookkeeping
    save_every: int = 0
    profile: bool = False
    profile_json: Optional[PathLike] = None
    log: bool = True


@dataclass
class SampleResult:
    """Outputs of a completed sample."""

    video: torch.Tensor
    audio: torch.Tensor
    video_path: Path
    audio_path: Path
    frame_count: int
    latent_t: int
    lat_h: int
    lat_w: int
    audio_t: int
    canvas: CanvasSpec
    payload: Payload
    plan: ConditioningPlan
    sigmas: List[float]
    seconds: float
    pin_info: Optional[Dict[str, Any]] = None
    profile: Optional[PhaseProfiler] = None
    notes: List[str] = field(default_factory=list)


class SampleError(RuntimeError):
    """Fatal sample setup or non-finite output."""


def _log(req: SampleRequest, msg: str) -> None:
    if req.log:
        print(msg, flush=True)


def _log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_sample_canvas(req: SampleRequest) -> CanvasSpec:
    if req.canvas is not None:
        return req.canvas
    if req.width is not None and req.height is not None:
        return resolve_canvas(
            width=int(req.width),
            height=int(req.height),
            mode=req.canvas_mode or "raw",
        )
    return resolve_canvas(
        aspect=req.aspect,
        megapixels=req.megapixels,
        short_edge=req.short_edge,
        width=req.width,
        height=req.height,
        mode=req.canvas_mode or "preview",
    )


def load_sample_model(
    req: SampleRequest,
) -> Tuple[MiniMaxH3Model, SafetensorsSource, torch.dtype]:
    """Load checkpoint into a streamed-ready model (weights on CPU)."""
    compute = torch.bfloat16 if req.device == "cuda" else torch.float32
    weights_path = Path(req.weights)
    src = SafetensorsSource(weights_path, "cpu", compute)
    attn = AttnImpl.normalize(req.attn)
    model = MiniMaxH3Model(src, req.cfg, attn)
    model.stream_prefetch_depth = int(req.prefetch)
    missing, shape_mismatch, unused = src.report()
    if missing:
        raise SampleError(f"checkpoint missing tensors: {missing}")
    if shape_mismatch:
        raise SampleError(f"checkpoint shape mismatches: {shape_mismatch}")
    if unused and req.log:
        _log_err(f"note: unused checkpoint keys: {unused}")
    return model, src, compute


def pin_for_sample(
    model: MiniMaxH3Model,
    req: SampleRequest,
) -> Optional[Dict[str, Any]]:
    """Pin a limited host-weight runway when CUDA prefetch is on."""
    if req.device != "cuda" or int(req.prefetch) <= 0:
        return None
    _log(
        req,
        f"pinning a limited host-weight runway for async H2D "
        f"(max_pin={req.max_pin_gb:.1f} GB, keep ≥{req.ram_reserve_gb:.1f} GB free)...",
    )
    t_pin = time.perf_counter()
    pin_info = model.pin_stream_hosts(
        reserve_gb=float(req.ram_reserve_gb),
        max_pin_gb=float(req.max_pin_gb),
    )
    dt_pin = time.perf_counter() - t_pin
    ab = pin_info.get("avail_gb_before")
    aa = pin_info.get("avail_gb_after")
    ub = pin_info.get("used_gb_before")
    ua = pin_info.get("used_gb_after")
    tot = pin_info.get("total_gb")
    ab_s = f"{ab:.1f} GB avail" if ab is not None else "unknown avail"
    aa_s = f"{aa:.1f} GB avail" if aa is not None else "unknown avail"
    used_s = ""
    if ub is not None and ua is not None and tot is not None and tot > 0:
        used_s = f", in-use {ub:.1f}->{ua:.1f} / {tot:.1f} GB ({100 * ua / tot:.0f}%)"
    _log(
        req,
        f"pin pass {dt_pin:.1f}s — pinned_groups={pin_info.get('pinned_groups')} "
        f"(~{pin_info.get('pinned_gb', 0):.2f} GB locked) "
        f"pageable_groups={pin_info.get('pageable_groups')} "
        f"({ab_s} -> {aa_s}{used_s})",
    )
    reason = pin_info.get("stop_reason")
    if reason == "reserve":
        _log(
            req,
            "note: stopped pinning to protect free-RAM reserve; "
            "rest of weights stay normal (pageable). Quality unchanged.",
        )
    elif reason == "max_pin":
        _log(
            req,
            "note: hit --max-pin-gb cap (this is intentional — pinning the "
            "full ~20 GB model locks almost all physical RAM on a 64 GB box). "
            "Rest stays pageable. Quality unchanged.",
        )
    elif reason == "no_pin":
        _log(req, "note: --max-pin-gb 0 → no host pinning; pure pageable H2D.")
    return pin_info


def euler_denoise(
    model: MiniMaxH3Model,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    text: torch.Tensor,
    payload: Payload,
    sigmas: Sequence[float],
    stream_gpu: Optional[str],
    precompute_adaln: bool,
    save_every: int = 0,
    out_dir: Optional[Path] = None,
    prof: Optional[PhaseProfiler] = None,
    log: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flat-ODE Euler over ``sigmas`` (length steps+1, final 0)."""
    steps = len(sigmas) - 1
    t0 = time.perf_counter()
    if prof is not None:
        prof.start_wall()
    with torch.no_grad():
        for i in range(steps):
            sigma = float(sigmas[i])
            dt = float(sigmas[i + 1]) - sigma
            v_out, a_out = model.forward(
                video,
                audio,
                text,
                None,
                sigma,
                payload,
                stream_gpu=stream_gpu,
                precompute_adaln=precompute_adaln and stream_gpu is not None,
            )
            video = video + v_out * dt
            audio = audio + a_out * dt
            if prof is not None:
                prof.mark_step()
            if log and ((i + 1) % 5 == 0 or i + 1 == steps):
                el = time.perf_counter() - t0
                print(
                    f"step {i + 1:>3}/{steps}  sigma {sigma:.6f} -> {sigmas[i + 1]:.6f}  "
                    f"({el:.1f}s, {el / (i + 1):.2f}s/step)",
                    flush=True,
                )
            if save_every > 0 and out_dir is not None and (i + 1) % save_every == 0:
                save_safetensors(
                    out_dir / f"video_step_{i:04d}.safetensors",
                    [("video_latent", video.detach().cpu().contiguous())],
                )
                save_safetensors(
                    out_dir / f"audio_step_{i:04d}.safetensors",
                    [("audio_latent", audio.detach().cpu().contiguous())],
                )
    if prof is not None:
        prof.end_wall()
    return video, audio


def run_sample(req: SampleRequest) -> SampleResult:
    """Full production sample: load → pack → denoise → write latents."""
    cfg = req.cfg
    device = req.device
    out = Path(req.out)
    out.mkdir(parents=True, exist_ok=True)

    shape = temporal_shape(req.length)
    frame_count = shape.frame_count
    latent_t = shape.latent_t
    audio_t = shape.audio_t
    canvas = resolve_sample_canvas(req)
    lat_h, lat_w = canvas.latent_hw

    text_raw = load_text_embeds(req.text, device=device)
    _log(req, f"prompt tokens {tuple(text_raw.shape)} (loaded)")
    _log(
        req,
        f"canvas {canvas.width}x{canvas.height} ({canvas.aspect}, {canvas.megapixels:.3f} MP, "
        f"mode={canvas.mode}"
        f"{', clamped' if canvas.clamped else ''}"
        f"{'; ' + canvas.note if canvas.note else ''}) "
        f"-> video latent [1, {cfg.latents_dim}, {latent_t}, {lat_h}, {lat_w}], "
        f"audio [1, {cfg.audio_latents_dim}, 2, {audio_t}] "
        f"({frame_count} frames @ 24fps, {shape.duration_s:.2f}s)",
    )

    sigmas = flow_sigmas(int(req.steps), cfg.sigma_shift_video)
    _log(
        req,
        f"flow schedule: shift {cfg.sigma_shift_video}, {req.steps} steps, "
        f"sigma {sigmas[0]:.6f} -> {sigmas[-2]:.6f} -> 0.0",
    )

    model, _src, _compute = load_sample_model(req)
    pin_info = pin_for_sample(model, req)

    _lin0 = model.blocks[0].attn.qkv_proj
    _d0 = getattr(_lin0, "deferred_int8", None)
    if _d0 is not None:
        offload_msg = (
            "offload (deferred int8): ~20 GB weights live in system RAM as packed int8 "
            "(not all pinned/locked)"
        )
    else:
        offload_msg = "offload (full-precision host weights)"
    _log(
        req,
        f"{offload_msg}; "
        f"GPU keeps ~{max(int(req.prefetch), 1) if req.prefetch else 1} layer(s) hot while streaming "
        f"— prefetch={req.prefetch}, precompute_adaln={req.precompute_adaln}, "
        f"preproject_text={req.preproject_text}, ram_reserve_gb={req.ram_reserve_gb}, "
        f"max_pin_gb={req.max_pin_gb}, "
        "no full bf16 weight set materialized, no disk I/O during generation",
    )

    try:
        plan = build_conditioning(
            cfg=cfg,
            frame_count=frame_count,
            lat_h=lat_h,
            lat_w=lat_w,
            first_frame=req.first_frame,
            last_frame=req.last_frame,
            ref_specs=req.ref_specs,
            allow_keyframe_refs=req.allow_keyframe_refs,
            device=device,
        )
    except ConditioningError as e:
        raise SampleError(str(e)) from e

    for note in plan.notes:
        _log(req, note)

    payload = plan.to_payload(
        seed=int(req.seed),
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    if plan.keyframes:
        _log(
            req,
            f"payload FL2VA keyframes={[kf.resolved_frame_index for kf in plan.keyframes]} "
            f"frame_count={frame_count} visual_cond_noise_aug={payload.visual_cond_noise_aug}",
        )
    if plan.refs:
        _log(
            req,
            f"payload Ref2VA refs={len(plan.refs)} "
            f"(img={plan.n_img} vid={plan.n_vid} aud={plan.n_aud} files={plan.n_files}) "
            f"visual_cond_noise_aug={payload.visual_cond_noise_aug} "
            f"audio_cond_noise_aug={payload.audio_cond_noise_aug}",
        )

    g = torch.Generator(device=device).manual_seed(int(req.seed))
    video = torch.randn(
        1, cfg.latents_dim, latent_t, lat_h, lat_w, generator=g, device=device
    )
    g2 = torch.Generator(device=device).manual_seed(int(req.seed) + 1)
    audio = torch.randn(
        1, cfg.audio_latents_dim, 2, audio_t, generator=g2, device=device
    )

    prof = PhaseProfiler(enabled=bool(req.profile), use_cuda=(device == "cuda"))
    set_profiler(prof if req.profile else None)

    stream_dev = device if device == "cuda" else None
    if device != "cuda":
        model.to(device)

    text = text_raw
    if req.preproject_text:
        t_txt = time.perf_counter()
        with torch.no_grad():
            text = model.prepare_text_states(text_raw, stream_gpu=stream_dev)
        if device == "cuda":
            torch.cuda.synchronize()
        _log(
            req,
            f"pre-projected text {tuple(text_raw.shape)} -> {tuple(text.shape)} "
            f"in {time.perf_counter() - t_txt:.2f}s (reused every step)",
        )
    else:
        _log(req, f"text left as loaded {tuple(text.shape)} (proj+refiner each step)")

    t0 = time.perf_counter()
    try:
        video, audio = euler_denoise(
            model,
            video=video,
            audio=audio,
            text=text,
            payload=payload,
            sigmas=sigmas,
            stream_gpu=stream_dev,
            precompute_adaln=bool(req.precompute_adaln),
            save_every=int(req.save_every),
            out_dir=out,
            prof=prof if req.profile else None,
            log=req.log,
        )
    finally:
        set_profiler(None)

    seconds = time.perf_counter() - t0
    _log(req, f"denoising done in {seconds:.1f}s")
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
        raise SampleError("sample produced non-finite latents")

    video_path = out / "video_latent.safetensors"
    audio_path = out / "audio_latent.safetensors"
    save_safetensors(
        video_path, [("video_latent", video.detach().cpu().contiguous())]
    )
    save_safetensors(
        audio_path, [("audio_latent", audio.detach().cpu().contiguous())]
    )
    _log(req, f"wrote {video_path}")
    _log(req, f"wrote {audio_path}")

    if req.profile:
        prof.print_summary()
        extra = {
            "weights": str(req.weights),
            "canvas": [canvas.width, canvas.height],
            "canvas_spec": canvas.as_dict(),
            "latent": [latent_t, lat_h, lat_w],
            "frames": frame_count,
            "seed": int(req.seed),
            "attn": AttnImpl.normalize(req.attn),
            "sage_backend": (
                get_sage_backend()
                if AttnImpl.normalize(req.attn) == AttnImpl.SAGE
                else None
            ),
            "prefetch": int(req.prefetch),
            "precompute_adaln": bool(req.precompute_adaln),
            "preproject_text": bool(req.preproject_text),
            "ram_reserve_gb": float(req.ram_reserve_gb),
            "max_pin_gb": float(req.max_pin_gb),
            "device": device,
            "first_frame": str(req.first_frame) if req.first_frame else None,
            "last_frame": str(req.last_frame) if req.last_frame else None,
            "keyframes": [kf.resolved_frame_index for kf in plan.keyframes],
            "refs": [
                {
                    "kind": getattr(r.kind, "name", str(r.kind)),
                    "latent_t": r.latent_t,
                    "latent_h": r.latent_h,
                    "latent_w": r.latent_w,
                    "ref_audio_t": r.ref_audio_t,
                }
                for r in plan.refs
            ],
        }
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            extra["git"] = sha
        except Exception:
            extra["git"] = None
        dest = Path(req.profile_json) if req.profile_json else (out / "profile.json")
        prof.write_json(dest, extra=extra)

    return SampleResult(
        video=video,
        audio=audio,
        video_path=video_path,
        audio_path=audio_path,
        frame_count=frame_count,
        latent_t=latent_t,
        lat_h=lat_h,
        lat_w=lat_w,
        audio_t=audio_t,
        canvas=canvas,
        payload=payload,
        plan=plan,
        sigmas=list(sigmas),
        seconds=seconds,
        pin_info=pin_info,
        profile=prof if req.profile else None,
        notes=list(plan.notes),
    )
