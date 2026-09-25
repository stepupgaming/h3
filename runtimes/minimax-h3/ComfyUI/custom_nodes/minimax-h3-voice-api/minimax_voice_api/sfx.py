"""First-class MiniMax H3 SFX / Foley generation (not music, not speech)."""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import JOB_ROOT
from .core import concat_audio
from .engine import FPS, ComfyClient, frames_for
from .sfx_presets import SFX_PRESETS

# H3 snaps to the 17k+5 frame grid; short one-shots still need a few seconds of
# timeline so the event is not crushed into a single noisy frame cluster.
MIN_SECONDS = 1.5
MAX_SECONDS = 30.0
ONESHOT_DEFAULT = 3.0
LOOP_DEFAULT = 12.0

KINDS = ("oneshot", "loop_bed")
PROMPT_PROFILES = ("diegetic_v1",)


def _plain_direction(value: str, fallback: str) -> str:
    value = re.sub(r"</?d(?:\s[^>]*)?>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<(?:scenetrans|cutoff)\s*/?>", "", value, flags=re.IGNORECASE)
    # Strip accidental lyric/section markup so SFX never inherits song grammar.
    value = re.sub(r"^\s*\[[^\]]+]\s*", "", value)
    return " ".join(value.split()).strip() or fallback


def resolve_preset(preset_id: str | None) -> dict[str, object] | None:
    if not preset_id:
        return None
    return SFX_PRESETS.get(preset_id)


def recommend_sfx_duration(
    description: str,
    *,
    kind: str = "oneshot",
    maximum: float = MAX_SECONDS,
) -> dict:
    """Heuristic length for one-shots vs continuous beds (no H3 call)."""
    if kind not in KINDS:
        raise ValueError("kind must be oneshot or loop_bed")
    text = description.casefold()
    words = len(re.findall(r"\b[\w’'-]+\b", description))
    multi = any(
        term in text
        for term in (
            "several", "multiple", "sequence", "then ", "followed by",
            "walk", "footsteps", "pass-by", "pass by", "open and close",
            "cycle", "pattern",
        )
    )
    longish = any(
        term in text
        for term in (
            "roll", "decay", "settle", "sustain", "hold", "continuous",
            "steady", "bed", "ambience", "atmosphere", "loop",
        )
    )
    tiny = any(
        term in text
        for term in (
            "click", "tick", "blip", "snap", "tap", "ding", "shutter",
            "toggle", "single",
        )
    )
    if kind == "loop_bed":
        raw = LOOP_DEFAULT + (2.0 if longish else 0.0) + min(6.0, words * 0.05)
        recommended = max(6.0, min(maximum, 1.0 * math.floor(raw + 0.5)))
    else:
        raw = ONESHOT_DEFAULT
        if tiny and not multi:
            raw = 2.0
        if multi:
            raw = max(raw, 5.0)
        if longish:
            raw = max(raw, 5.0)
        if words > 24:
            raw += 1.5
        recommended = max(MIN_SECONDS, min(maximum, 0.5 * math.floor(raw * 2 + 0.5)))
    return {
        "kind": kind,
        "words": words,
        "raw_seconds": round(raw, 2),
        "recommended_seconds": float(recommended),
        "capped": raw > maximum,
        "multi_event": multi,
        "tiny_event": tiny and not multi,
    }


def sfx_prompt(
    description: str,
    *,
    kind: str = "oneshot",
    space: str = "a neutral dry recording stage with no visible source",
    intensity: float = 0.7,
    profile: str = "diegetic_v1",
) -> str:
    """Build an audio-first H3 prompt that forbids speech and score."""
    if kind not in KINDS:
        raise ValueError("kind must be oneshot or loop_bed")
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"unknown sfx prompt profile: {profile}")
    description = _plain_direction(
        description,
        "A single clean diegetic sound effect with a natural attack and short decay",
    )
    space = _plain_direction(space, "a neutral dry recording stage with no visible source")
    intensity = min(1.0, max(0.0, float(intensity)))
    if intensity < 0.34:
        energy = "soft, understated, and low in the mix"
    elif intensity < 0.67:
        energy = "clear and present without exaggeration"
    else:
        energy = "bold, close, and highly intelligible as a designed effect"
    if kind == "loop_bed":
        timing = (
            "The soundscape is continuous and even for the entire clip so it can "
            "serve as a bed or loop source. Avoid a single climactic hit at the end."
        )
    else:
        timing = (
            "The primary event begins promptly after a tiny natural pre-roll of air, "
            "completes once (or as a short intentional sequence if described), then "
            "decays into clean silence or faint residual air. Do not fill unused "
            "timeline with extra unrelated events, music, or speech."
        )
    body = (
        f"[Shot 1] A plain, static, featureless dark field in {space.rstrip('.')}. "
        "The camera never moves and nothing is visible; this is an audio-only capture. "
        "There are no people on screen, no mouths, no dialogue blocks, and no readable text. "
        f"Diegetic sound only: {description.rstrip('.')}. "
        f"Presentation is {energy}. {timing} "
        "Absolutely no singing, spoken words, whispered words, vocal samples, choir, "
        "rap, lyrics, announcer, or language of any kind. "
        "Absolutely no non-diegetic music, melody, harmony, beat, bassline, chord stinger "
        "as score, or song arrangement—only the described effect and honest acoustic space."
    )
    return (
        f"integrated_multimodal_description: {body}\n\n"
        f"overall_soundscape: {description.rstrip('.')}. {energy.capitalize()}. "
        f"Recorded in {space.rstrip('.')}.\n\n"
        "non_diegetic_music: N/A"
    )


@dataclass
class SfxResult:
    job_id: str
    path: Path
    prompt: str
    duration_seconds: float
    render_seconds: float
    kind: str
    seed: int
    prompt_profile: str
    duration_mode: str
    duration_recommendation: dict | None
    preset: str | None
    intensity: float


class SfxEngine:
    def __init__(self, comfy: ComfyClient) -> None:
        self.comfy = comfy

    def generate(
        self,
        *,
        description: str,
        kind: str = "oneshot",
        space: str = "a neutral dry recording stage with no visible source",
        intensity: float = 0.7,
        duration_seconds: float = ONESHOT_DEFAULT,
        duration_mode: str = "auto",
        steps: int = 20,
        seed: int | None = None,
        response_format: str = "flac",
        prompt_profile: str = "diegetic_v1",
        resolution: int = 32,
        preset: str | None = None,
    ) -> SfxResult:
        recommendation = recommend_sfx_duration(description, kind=kind)
        if duration_mode == "auto":
            duration_seconds = float(recommendation["recommended_seconds"])
        elif duration_mode != "manual":
            raise ValueError("duration_mode must be auto or manual")

        duration_seconds = min(MAX_SECONDS, max(MIN_SECONDS, float(duration_seconds)))
        frames = frames_for(duration_seconds)
        duration = frames / FPS
        prompt = sfx_prompt(
            description,
            kind=kind,
            space=space,
            intensity=intensity,
            profile=prompt_profile,
        )
        job_id = uuid.uuid4().hex
        prefix = f"voice_api/sfx/{job_id}/take"
        actual_seed = seed if seed is not None else random.randrange(1, 2**63)
        from .engine import t2va_graph
        graph = t2va_graph(
            prompt, duration, steps, actual_seed, prefix, resolution=resolution)
        raw, elapsed = self.comfy.run(graph, {
            "kind": "sfx", "job_id": job_id, "sfx_kind": kind,
            "duration_seconds": round(duration, 3), "steps": steps,
        })
        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        native = job_dir / "sfx.flac"
        shutil.copy2(raw, native)
        output = native
        if response_format != "flac":
            output = job_dir / f"sfx.{response_format}"
            concat_audio([native], output, response_format, channels=2)
        report = {
            "job_id": job_id,
            "kind": kind,
            "preset": preset,
            "duration_seconds": round(duration, 3),
            "steps": steps,
            "seed": actual_seed,
            "intensity": intensity,
            "prompt_profile": prompt_profile,
            "duration_mode": duration_mode,
            "duration_recommendation": recommendation,
            "render_seconds": round(elapsed, 3),
            "output": str(output),
            "description": description,
            "space": space,
            "prompt": prompt,
        }
        (job_dir / "sfx_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return SfxResult(
            job_id, output, prompt, duration, elapsed, kind, actual_seed,
            prompt_profile, duration_mode, recommendation, preset, intensity,
        )
