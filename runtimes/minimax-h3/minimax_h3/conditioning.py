"""Conditioning loaders + Payload builders for T2VA / FL2VA / Ref2VA.

Validates latent shapes and Ref2VA caps once, then produces a ``Payload`` whose
``cond_*`` list order matches ``PackedLayout`` / ``_embed`` (keyframes first;
VIDEO_AUDIO pushes audio latent before video latent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .config import H3Config
from .layout import Keyframe, RefBlock, RefKind
from .model import Payload

PathLike = Union[str, Path]
RefSpec = Tuple[str, str]  # ("image"|"video"|"audio"|"av", path or "v,a")


class ConditioningError(ValueError):
    """Invalid conditioning paths, shapes, or mode combination."""


def _reject_raw_media(path: Path, *, kind: str) -> None:
    suffix = path.suffix.lower()
    media = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
        ".gif",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    }
    if suffix in media:
        raise ConditioningError(
            f"{path}: pass a DiT-normalized .safetensors {kind} latent "
            "(encode first with: uv run python scripts/h3_vae_encode.py "
            f"{path} --mode auto --width W --height H --out ref.safetensors). "
            "Sample does not load the VAE (16 GB sequencing)."
        )


def load_text_embeds(path: PathLike, device: str = "cpu") -> torch.Tensor:
    """Load text conditioning embeds from a safetensors file.

    Prefers ``text_embeds`` when present. Falls back to the sole tensor, or the
    first 2D float tensor if multiple unnamed keys exist.
    """
    import safetensors.torch

    path = Path(path)
    tensors = safetensors.torch.load_file(str(path))
    if not tensors:
        raise ConditioningError(f"no tensors in {path}")
    if "text_embeds" in tensors:
        t = tensors["text_embeds"]
    elif len(tensors) == 1:
        t = next(iter(tensors.values()))
    else:
        candidates = [
            (k, v)
            for k, v in tensors.items()
            if v.ndim >= 2 and v.dtype in (torch.float32, torch.float16, torch.bfloat16)
        ]
        if not candidates:
            raise ConditioningError(
                f"{path} has multiple tensors and no text_embeds: {list(tensors)}"
            )
        t = candidates[0][1]
    return t.to(device=device, dtype=torch.float32)


def load_cond_video_latent(
    path: PathLike,
    lat_h: int,
    lat_w: int,
    device: str = "cpu",
    *,
    require_t1: bool = True,
) -> torch.Tensor:
    """Load a DiT-normalized visual condition latent from ``.safetensors``.

    Accepts keys ``video_latent``, ``first_frame_latent``, ``last_frame_latent``,
    ``latent``, or the sole tensor.

    - FL2VA keyframes: ``require_t1=True`` → ``[1, 24, 1, lat_h, lat_w]``
    - Ref2VA image/video: ``require_t1=False`` → ``[1, 24, T, lat_h, lat_w]``
    """
    import safetensors.torch

    path = Path(path)
    _reject_raw_media(path, kind="video")
    blob = safetensors.torch.load_file(str(path))
    preferred = ("video_latent", "first_frame_latent", "last_frame_latent", "latent")
    t = None
    for k in preferred:
        if k in blob:
            t = blob[k]
            break
    if t is None:
        if len(blob) == 1:
            t = next(iter(blob.values()))
        else:
            raise ConditioningError(
                f"{path}: no video_latent/first_frame_latent key; keys={list(blob)}"
            )
    t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 4:
        t = t.unsqueeze(0)
    if t.ndim != 5:
        raise ConditioningError(f"{path}: expected 5D latent [1,C,T,H,W], got {tuple(t.shape)}")
    if t.shape[0] != 1:
        raise ConditioningError(f"{path}: expected batch=1, got {tuple(t.shape)}")
    if require_t1 and t.shape[2] != 1:
        raise ConditioningError(
            f"{path}: expected temporal=1 keyframe, got {tuple(t.shape)} "
            "(use --ref-video for multi-frame refs)"
        )
    if t.shape[3] != lat_h or t.shape[4] != lat_w:
        raise ConditioningError(
            f"{path}: spatial {t.shape[3]}x{t.shape[4]} != sample latent "
            f"{lat_h}x{lat_w} (canvas/16). Re-encode at the sample canvas."
        )
    return t.contiguous()


def load_cond_audio_latent(path: PathLike, device: str = "cpu") -> torch.Tensor:
    """Load a DiT-normalized stereo audio condition latent ``[1, 32, 2, T]``."""
    import safetensors.torch

    path = Path(path)
    _reject_raw_media(path, kind="audio")
    blob = safetensors.torch.load_file(str(path))
    preferred = ("audio_latent", "latent")
    t = None
    for k in preferred:
        if k in blob:
            t = blob[k]
            break
    if t is None:
        if len(blob) == 1:
            t = next(iter(blob.values()))
        else:
            raise ConditioningError(f"{path}: no audio_latent key; keys={list(blob)}")
    t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 3:
        if t.shape[0] == 32 and t.shape[1] == 2:
            t = t.unsqueeze(0)
        elif t.shape[0] == 2 and t.shape[1] == 32:
            t = t.permute(1, 0, 2).unsqueeze(0)
        else:
            raise ConditioningError(f"{path}: ambiguous 3D audio shape {tuple(t.shape)}")
    if t.ndim != 4:
        raise ConditioningError(f"{path}: expected [1,32,2,T], got {tuple(t.shape)}")
    if t.shape[0] != 1 or t.shape[1] != 32 or t.shape[2] != 2:
        raise ConditioningError(f"{path}: expected [1,32,2,T], got {tuple(t.shape)}")
    if t.shape[3] < 1:
        raise ConditioningError(f"{path}: empty audio temporal axis")
    return t.contiguous()


@dataclass
class ConditioningPlan:
    """Validated structure + tensors ready for ``Payload`` / logging."""

    keyframes: List[Keyframe] = field(default_factory=list)
    refs: List[RefBlock] = field(default_factory=list)
    cond_video_latents: List[torch.Tensor] = field(default_factory=list)
    cond_audio_latents: List[torch.Tensor] = field(default_factory=list)
    frame_count: Optional[int] = None
    n_img: int = 0
    n_vid: int = 0
    n_aud: int = 0
    n_files: int = 0
    notes: List[str] = field(default_factory=list)

    def to_payload(
        self,
        *,
        seed: int,
        sigma_shift_video: float,
        sigma_shift_audio: float,
    ) -> Payload:
        return Payload(
            cond_video_latents=list(self.cond_video_latents),
            cond_audio_latents=list(self.cond_audio_latents),
            keyframes=list(self.keyframes),
            refs=list(self.refs),
            frame_count=self.frame_count if self.keyframes else None,
            seed=seed,
            sigma_shift_video=sigma_shift_video,
            sigma_shift_audio=sigma_shift_audio,
        )


def build_conditioning(
    *,
    cfg: H3Config,
    frame_count: int,
    lat_h: int,
    lat_w: int,
    first_frame: Optional[PathLike] = None,
    last_frame: Optional[PathLike] = None,
    ref_specs: Optional[Sequence[RefSpec]] = None,
    allow_keyframe_refs: bool = False,
    device: str = "cpu",
    latents_dim: Optional[int] = None,
    audio_latents_dim: Optional[int] = None,
) -> ConditioningPlan:
    """Build a validated FL2VA / Ref2VA / T2VA conditioning plan.

    Modes are mutually exclusive unless ``allow_keyframe_refs`` (experimental
    Multishot-style DiT memory on a Ref2VA checkpoint).
    """
    v_ch = int(latents_dim if latents_dim is not None else cfg.latents_dim)
    a_ch = int(audio_latents_dim if audio_latents_dim is not None else cfg.audio_latents_dim)
    ref_specs = list(ref_specs or [])
    has_kf = first_frame is not None or last_frame is not None
    has_refs = bool(ref_specs)

    if has_kf and has_refs and not allow_keyframe_refs:
        raise ConditioningError(
            "cannot combine FL2VA first/last frame with Ref2VA refs "
            "(use one mode, or allow_keyframe_refs for experimental Multishot DiT memory)"
        )

    plan = ConditioningPlan()
    if allow_keyframe_refs and has_kf and has_refs:
        plan.notes.append(
            "packing FL2VA keyframe(s) + Ref2VA refs "
            "(experimental Multishot DiT memory; prefer Ref2VA weights)"
        )

    if first_frame is not None:
        z0 = load_cond_video_latent(first_frame, lat_h, lat_w, device=device)
        if z0.shape[1] != v_ch:
            raise ConditioningError(f"first-frame channels {z0.shape[1]} != {v_ch}")
        plan.keyframes.append(Keyframe(0))
        plan.cond_video_latents.append(z0)
        plan.notes.append(f"FL2VA first-frame cond {tuple(z0.shape)} from {first_frame}")

    if last_frame is not None:
        z1 = load_cond_video_latent(last_frame, lat_h, lat_w, device=device)
        if z1.shape[1] != v_ch:
            raise ConditioningError(f"last-frame channels {z1.shape[1]} != {v_ch}")
        plan.keyframes.append(Keyframe(int(frame_count) - 1))
        plan.cond_video_latents.append(z1)
        plan.notes.append(f"FL2VA last-frame cond {tuple(z1.shape)} from {last_frame}")

    if plan.keyframes:
        plan.frame_count = int(frame_count)

    n_img = n_vid = n_aud = n_files = 0
    for kind, val in ref_specs:
        if kind == "image":
            n_img += 1
            n_files += 1
            if n_img > 9:
                raise ConditioningError("Ref2VA allows ≤9 image refs")
            z = load_cond_video_latent(
                val, lat_h, lat_w, device=device, require_t1=False
            )
            if z.shape[1] != v_ch:
                raise ConditioningError(f"ref-image channels {z.shape[1]} != {v_ch}")
            if z.shape[2] != 1:
                raise ConditioningError(
                    f"ref-image must be T=1 latent, got T={z.shape[2]} "
                    "(use video refs for clips)"
                )
            plan.refs.append(RefBlock(RefKind.IMAGE, 0, z.shape[3], z.shape[4], 0))
            plan.cond_video_latents.append(z)
            plan.notes.append(f"Ref2VA image ref {tuple(z.shape)} from {val}")
        elif kind == "video":
            n_vid += 1
            n_files += 1
            if n_vid > 3:
                raise ConditioningError("Ref2VA allows ≤3 video refs")
            z = load_cond_video_latent(
                val, lat_h, lat_w, device=device, require_t1=False
            )
            if z.shape[1] != v_ch:
                raise ConditioningError(f"ref-video channels {z.shape[1]} != {v_ch}")
            vt = int(z.shape[2])
            if vt < 1:
                raise ConditioningError("ref-video empty temporal axis")
            plan.refs.append(RefBlock(RefKind.VIDEO, vt, z.shape[3], z.shape[4], 0))
            plan.cond_video_latents.append(z)
            plan.notes.append(f"Ref2VA video ref {tuple(z.shape)} from {val}")
        elif kind == "audio":
            n_aud += 1
            n_files += 1
            if n_aud > 3:
                raise ConditioningError("Ref2VA allows ≤3 audio refs")
            za = load_cond_audio_latent(val, device=device)
            if za.shape[1] != a_ch:
                raise ConditioningError(f"ref-audio channels {za.shape[1]} != {a_ch}")
            rt = int(za.shape[3])
            plan.refs.append(RefBlock(RefKind.AUDIO, 0, 0, 0, rt))
            plan.cond_audio_latents.append(za)
            plan.notes.append(f"Ref2VA audio ref {tuple(za.shape)} from {val}")
        elif kind == "av":
            parts = str(val).split(",")
            if len(parts) != 2:
                raise ConditioningError(f"av ref expects VIDEO,AUDIO paths, got {val!r}")
            vpath, apath = Path(parts[0].strip()), Path(parts[1].strip())
            n_vid += 1
            n_aud += 1
            n_files += 2
            if n_vid > 3:
                raise ConditioningError("Ref2VA allows ≤3 video slots")
            if n_aud > 3:
                raise ConditioningError("Ref2VA allows ≤3 audio clips")
            z = load_cond_video_latent(
                vpath, lat_h, lat_w, device=device, require_t1=False
            )
            za = load_cond_audio_latent(apath, device=device)
            if z.shape[1] != v_ch:
                raise ConditioningError(f"ref-av video channels {z.shape[1]}")
            if za.shape[1] != a_ch:
                raise ConditioningError(f"ref-av audio channels {za.shape[1]}")
            vt = int(z.shape[2])
            rt = int(za.shape[3])
            # Layout emits REF_AUDIO then REF_IMG for VIDEO_AUDIO — audio first.
            plan.refs.append(
                RefBlock(RefKind.VIDEO_AUDIO, vt, z.shape[3], z.shape[4], rt)
            )
            plan.cond_audio_latents.append(za)
            plan.cond_video_latents.append(z)
            plan.notes.append(
                f"Ref2VA av ref video {tuple(z.shape)} + audio {tuple(za.shape)} "
                f"from {vpath} , {apath}"
            )
        else:
            raise ConditioningError(f"unknown ref kind {kind}")

    if n_files > 12:
        raise ConditioningError("Ref2VA allows ≤12 input files total")
    if plan.refs and n_img + n_vid == 0:
        raise ConditioningError(
            "Ref2VA audio cannot be the sole input — add an image or video ref"
        )

    plan.n_img, plan.n_vid, plan.n_aud, plan.n_files = n_img, n_vid, n_aud, n_files
    return plan
