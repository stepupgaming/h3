"""H3 temporal grid + flow schedule helpers.

Single home for the 24 fps / 17k+5 frame snap and latent_t / audio_t mapping
used by sample, continue, VAE encode, and prompt IR. Keep callers off local
copies of the while-loop snap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

FPS = 24
AUDIO_HZ = 40  # audio latent temporal rate (Hz)
FRAME_MOD = 17
FRAME_REM = 5  # frame_count ≡ 5 (mod 17); min length 5


@dataclass(frozen=True)
class TemporalShape:
    """Pixel frames + DiT latent axes derived from a snapped frame count."""

    frame_count: int
    latent_t: int
    audio_t: int
    duration_s: float

    @property
    def fps(self) -> int:
        return FPS


def snap_frames(n: int) -> int:
    """Snap *up* to the next valid H3 length (``17k+5``, min 5)."""
    fc = max(int(n), FRAME_REM)
    while fc % FRAME_MOD != FRAME_REM:
        fc += 1
    return fc


def floor_frames(n: int) -> int:
    """Snap *down* to the previous valid H3 length (min 5)."""
    fc = max(int(n), FRAME_REM)
    while fc % FRAME_MOD != FRAME_REM:
        fc -= 1
        if fc < FRAME_REM:
            return FRAME_REM
    return fc


def latent_t_from_frames(frame_count: int) -> int:
    """Video latent temporal length for a snapped frame count."""
    fc = int(frame_count)
    if fc <= FRAME_REM:
        return 2
    return ((fc - FRAME_REM) // FRAME_MOD) * 5 + 2


def audio_t_from_frames(frame_count: int, *, fps: int = FPS) -> int:
    """Audio latent temporal length (``round(duration * 40)``)."""
    return int(round(float(frame_count) / float(fps) * float(AUDIO_HZ)))


def temporal_shape(length: int, *, fps: int = FPS) -> TemporalShape:
    """Snap ``length`` frames and return latent/audio axes + duration."""
    frame_count = snap_frames(length)
    return TemporalShape(
        frame_count=frame_count,
        latent_t=latent_t_from_frames(frame_count),
        audio_t=audio_t_from_frames(frame_count, fps=fps),
        duration_s=frame_count / float(fps),
    )


def flow_sigmas(steps: int, shift: float) -> List[float]:
    """Comfy ``normal_scheduler`` on ``ModelSamplingDiscreteFlow(shift)``.

    ``snr(t) = shift*t / (1 + (shift-1)*t)``, timesteps linear in
    ``[1000, sigma_min*1000]``, append zero. Mirrors ``src/main.rs``.
    """

    def snr(t: float) -> float:
        return shift * t / (1.0 + (shift - 1.0) * t)

    sigma_min = snr(1.0 / 1000.0)
    start = 1000.0
    end = sigma_min * 1000.0
    sigmas: List[float] = []
    for i in range(int(steps)):
        frac = i / (max(int(steps), 2) - 1)
        ts = start + (end - start) * frac
        sigmas.append(snr(ts / 1000.0))
    sigmas.append(0.0)
    return sigmas


def canvas_latent_hw(width: int, height: int) -> Tuple[int, int]:
    """Pixel canvas → latent H/W (VAE 16× spatial)."""
    return int(height) // 16, int(width) // 16
