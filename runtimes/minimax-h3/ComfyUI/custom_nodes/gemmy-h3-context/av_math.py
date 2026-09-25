"""AV-safe H3 context math. No torch — used by Comfy nodes and Gemmy workers.

Architecture follows the published MultiRef / Context Loop contract:
- snap requested context *down* to a shared video+audio boundary
- 0 = preserve, 1 = generate on per-stream masks
- decoded audio must match absolute frame→sample timing or fail closed
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
MAX_FRACTIONAL_CHANGE = 0.005
LATENT_SCHEMA = "gemmy-h3-av-latent-v1"

# Shared AV-safe lengths: 39 + 51k (and the official 17k+5 grid members that
# also land on an integer 40 Hz audio step). 39 frames = 1.625 s = 65 audio steps.
AV_SAFE_FRAMES = (39, 90, 141, 192, 243, 294, 345, 396)


def align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    frame_count = int(frame_count)
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def audio_latent_t(frame_count: int) -> int:
    return int(round(int(frame_count) / FPS * AUDIO_LATENT_FPS))


def fit_audio_time(src_t: int, target_t: int) -> tuple[int, int]:
    """Keep/pad counts so an encoded wav matches the target AV audio T.

    Returns (keep_from_source, pad_right). Crop is always from the left;
    pad is zeros on the right (trailing silence).
    """
    src_t = int(src_t)
    target_t = int(target_t)
    if src_t < 1:
        raise ValueError(f"encoded audio T must be >= 1, got {src_t}")
    if target_t < 1:
        raise ValueError(f"target audio T must be >= 1, got {target_t}")
    if src_t >= target_t:
        return target_t, 0
    return src_t, target_t - src_t


def frames_from_video_tokens(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def native_guide_tail_spec(
    requested_frames: int, available_frames: int, target_frames: int
) -> tuple[int, int, int]:
    """AV-safe context frames plus video/audio token counts for a native tail guide.

    The guide must fit inside the new clip (MiniMaxH3AddGuide frame-range rule).
    """
    ctx = snap_context_frames(
        int(requested_frames), min(int(available_frames), int(target_frames))
    )
    return ctx, video_latent_t(ctx), audio_latent_t(ctx)


def snap_context_frames(requested: int, available: int) -> int:
    """Largest AV-safe length that fits in both the request and the predecessor."""
    cap = min(int(requested), int(available))
    chosen = 0
    for n in AV_SAFE_FRAMES:
        if n <= cap:
            chosen = n
        else:
            break
    if chosen == 0:
        # Still snap to 17k+5 if the cap is below 39 (imported short tail).
        n = cap
        while n >= 5 and n % 17 != 5:
            n -= 1
        if n >= 5 and audio_latent_t(n) * 3 == n * 5:
            return n
        raise ValueError(
            f"no AV-safe context length fits requested={requested} available={available}"
        )
    return chosen


def fingerprint(
    *,
    width: int,
    height: int,
    frames: int | None = None,
    video_shape: Sequence[int] | None = None,
    audio_shape: Sequence[int] | None = None,
    vae: str = "minimax_h3_video_vae_int8_convrot",
    dit: str = "",
    lora: Iterable[str] | None = None,
    mode: str = "",
    context_frames: int = 39,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": LATENT_SCHEMA,
        "width": int(width),
        "height": int(height),
        "frames": None if frames is None else int(frames),
        "video_shape": list(video_shape) if video_shape is not None else None,
        "audio_shape": list(audio_shape) if audio_shape is not None else None,
        "vae": str(vae),
        "dit": str(dit),
        "lora": sorted(str(x) for x in (lora or ()) if x),
        "mode": str(mode),
        "context_frames": int(context_frames),
    }
    if extra:
        body.update(extra)
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    body["digest"] = digest
    return body


def fingerprints_match(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    keys = ("width", "height", "vae", "dit", "lora", "mode")
    for key in keys:
        if a.get(key) != b.get(key):
            return False
    return True


def expected_audio_samples(frame_count: int, sample_rate: int) -> int:
    return int(round(int(frame_count) / FPS * int(sample_rate)))


def audio_tail_slice(actual_samples: int, frame_count: int, sample_rate: int) -> tuple[int, int]:
    """Sample [start, end) covering the last `frame_count` frames.

    Imported MP4s are usually longer than the snapped VAE-tail. Align audio to
    the same last-N-frame window as the video. A shorter file keeps [0, actual).
    """
    expected = expected_audio_samples(frame_count, sample_rate)
    actual = int(actual_samples)
    if actual >= expected:
        return actual - expected, actual
    return 0, actual


def audio_fractional_change(actual_samples: int, expected_samples: int) -> float:
    if expected_samples <= 0:
        return 1.0
    return abs(int(actual_samples) - int(expected_samples)) / float(expected_samples)


def conform_audio_ok(actual_samples: int, expected_samples: int) -> bool:
    return audio_fractional_change(actual_samples, expected_samples) <= MAX_FRACTIONAL_CHANGE
