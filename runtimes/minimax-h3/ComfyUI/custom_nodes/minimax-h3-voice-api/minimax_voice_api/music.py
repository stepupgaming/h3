"""Prompt-safe MiniMax H3 instrumental and song generation."""
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
from .engine import FPS, ComfyClient, frames_for, t2va_graph

SECTION_KINDS = (
    "pre-chorus", "breakdown", "chorus", "verse", "bridge", "intro",
    "outro", "solo", "hook",
)
SECTION_WEIGHT = {
    "intro": 1.0, "verse": 2.4, "pre-chorus": 1.2, "chorus": 2.2,
    "bridge": 1.5, "solo": 1.6, "breakdown": 1.4, "outro": 1.2,
    "hook": 1.5,
}
LEAD_IN = {
    "intro": "an instrumental introduction with no vocal",
    "verse": "the verse",
    "pre-chorus": "the pre-chorus, building",
    "chorus": "the chorus, opening up",
    "bridge": "the bridge, stripped back",
    "solo": "an instrumental solo with no vocal",
    "breakdown": "an instrumental breakdown with no vocal",
    "outro": "the outro, falling away",
    "hook": "the hook",
}
SUNG_LEAD_IN = {
    "intro": "the opening vocal hook",
    "solo": "a featured vocal passage",
    "breakdown": "the sung breakdown",
    "outro": "the sung outro, resolving",
}
PROMPT_PROFILES = ("natural_song_sheet_v2", "petrock_timed_v1")


def _plain_direction(value: str, fallback: str) -> str:
    value = re.sub(r"</?d(?:\s[^>]*)?>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<(?:scenetrans|cutoff)\s*/?>", "", value,
                   flags=re.IGNORECASE)
    return " ".join(value.split()).strip() or fallback


def parse_lyrics(text: str) -> list[tuple[str, list[str]]]:
    """Parse section-tagged plain lyrics; no caller-supplied dialogue markup."""
    if re.search(r"</?d(?:\s[^>]*)?>", text, flags=re.IGNORECASE):
        raise ValueError("lyrics must be plain text; the API adds dialogue tags")
    result: list[tuple[str, list[str]]] = []
    section, lines, seen = "verse", [], False
    for raw in text.strip().splitlines():
        line = raw.strip()
        match = re.fullmatch(r"\[(.+?)]", line)
        if match:
            if seen or lines:
                result.append((section, lines))
            section, lines, seen = match.group(1).strip().lower(), [], True
        elif line:
            lines.append(line)
    if seen or lines:
        result.append((section, lines))
    return result


def kind_of(section: str) -> str:
    lowered = section.lower()
    return next((kind for kind in SECTION_KINDS if kind in lowered), "verse")


def plan_sections(lyrics: str, seconds: float, *, word_aware: bool = False) -> list[dict]:
    blocks = parse_lyrics(lyrics)
    if not blocks or not any(lines for _, lines in blocks):
        raise ValueError("song mode requires at least one lyric line")
    weights = []
    for section, lines in blocks:
        kind = kind_of(section)
        if word_aware:
            words = sum(len(re.findall(r"\b[\w’'-]+\b", line)) for line in lines)
            if lines:
                weight = len(lines) * 2.5 + words * 0.35
                if kind == "chorus":
                    weight *= 1.05
                elif kind in {"intro", "pre-chorus"}:
                    weight *= 0.9
            else:
                weight = 3.2 if kind in {"intro", "outro"} else 4.0
        else:
            weight = SECTION_WEIGHT.get(kind, 2.0)
            if lines:
                weight *= 0.55 + 0.45 * len(lines)
        weights.append(weight)
    total = sum(weights) or 1.0
    cursor, result = 0.0, []
    for (section, lines), weight in zip(blocks, weights):
        duration = seconds * weight / total
        result.append({
            "section": section, "kind": kind_of(section), "lines": lines,
            "start": cursor, "end": min(seconds, cursor + duration),
        })
        cursor += duration
    return result


def song_metrics(style: str, lyrics: str) -> dict:
    """Describe lyric density and estimate a style-aware sung-word rate."""
    blocks = parse_lyrics(lyrics)
    lines = [line for _, section_lines in blocks for line in section_lines]
    words = sum(len(re.findall(r"\b[\w’'-]+\b", line)) for line in lines)
    vocal_sections = sum(bool(section_lines) for _, section_lines in blocks)
    instrumental_sections = sum(not section_lines for _, section_lines in blocks)
    bpm_match = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*bpm\b", style, re.IGNORECASE)
    bpm = float(bpm_match.group(1)) if bpm_match else 105.0
    bpm = min(220.0, max(45.0, bpm))
    lowered = style.casefold()
    rap = any(term in lowered for term in (
        "rap", "hip-hop", "hip hop", "trap", "drill", "spoken flow",
    ))
    fast = any(term in lowered for term in (
        "fast-paced", "fast paced", "double-time", "double time", "punk",
        "drum and bass", "dnb", "hyperpop",
    ))
    slow = any(term in lowered for term in (
        "slow", "ballad", "lullaby", "sparse", "drawn-out", "drawn out",
    ))
    words_per_second = 2.15 + (bpm - 100.0) * 0.006
    words_per_second += 0.65 if rap else 0.0
    words_per_second += 0.20 if fast else 0.0
    words_per_second -= 0.22 if slow and not rap else 0.0
    words_per_second = min(3.45, max(1.75, words_per_second))
    return {
        "words": words,
        "lines": len(lines),
        "vocal_sections": vocal_sections,
        "instrumental_sections": instrumental_sections,
        "bpm": round(bpm, 1),
        "words_per_second": round(words_per_second, 3),
        "rap_or_fast_flow": rap or fast,
        "slow_delivery": slow and not rap,
    }


def recommend_song_duration(style: str, lyrics: str, *, maximum: float = 60.0) -> dict:
    """Return a conservative duration recommendation without overruling the caller."""
    metrics = song_metrics(style, lyrics)
    if not metrics["words"]:
        raise ValueError("song mode requires at least one lyric line")
    vocal_time = metrics["words"] / metrics["words_per_second"]
    average_line_words = metrics["words"] / max(1, metrics["lines"])
    line_pause = 0.04 + min(8.0, average_line_words) * 0.01
    if not metrics["rap_or_fast_flow"]:
        line_pause = 0.08 + min(8.0, average_line_words) * 0.02
    if metrics["slow_delivery"]:
        line_pause = 0.12 + min(8.0, average_line_words) * 0.02
    instrumental_time = metrics["instrumental_sections"] * 2.0
    phrasing_time = metrics["lines"] * line_pause
    transition_time = max(0, metrics["vocal_sections"] - 1) * 0.6
    raw = vocal_time + phrasing_time + instrumental_time + transition_time + 1.0
    recommended = max(
        10.0, min(maximum, 5.0 * math.floor(raw / 5.0 + 0.5)))
    return {
        **metrics,
        "raw_seconds": round(raw, 2),
        "recommended_seconds": recommended,
        "capped": raw > maximum,
    }


def _stamp(seconds: float) -> str:
    if seconds < 60:
        return f"00:{seconds:06.3f}"
    return f"{int(seconds) // 60:02d}:{seconds % 60:06.3f}"


def instrumental_prompt(style: str, instrumentation: str,
                        soundscape: str = "N/A") -> str:
    style = _plain_direction(style, "A coherent original instrumental cue")
    instrumentation = _plain_direction(
        instrumentation, "A restrained arrangement with a clear melodic identity")
    soundscape = _plain_direction(soundscape, "N/A")
    music = f"{style.rstrip('.')}. {instrumentation.rstrip('.')}. No singing, speech, spoken words, vocal samples, or choir."
    return (
        "integrated_multimodal_description: [Shot 1] A plain, static, featureless "
        "dark field. The camera remains completely static and nothing moves for the "
        "entire clip. There are no people, voices, dialogue blocks, or audible words.\n\n"
        f"overall_soundscape: {soundscape}\n\n"
        f"non_diegetic_music: {music}"
    )


def _legacy_timed_song_body(style: str, instrumentation: str, vocalist: str,
                            plan: list[dict], language: str, scene: str) -> str:
    """The exact prompt grammar used by the successful original Pet Rock take."""
    body = (
        f"[Shot 1] A static, close, dimly lit shot in {scene.rstrip('.')} that never changes. "
        "This is one original musical performance in one take. Vocal identity and performance "
        f"direction only, never sung or spoken as lyrics: {vocalist.rstrip('.')}. "
        f"The musical style is {style.rstrip('.')}. The arrangement uses "
        f"{instrumentation.rstrip('.')}. One vocalist, Speaker 1 (S1), sings while "
        "the arrangement plays in time. Only words inside the marked lyric blocks "
        "are audible lyrics; section names, timestamps, descriptions, instruments, "
        "directions, and all other prompt prose remain inaudible."
    )
    for index, section in enumerate(plan):
        lead = (
            SUNG_LEAD_IN.get(section["kind"], LEAD_IN.get(section["kind"], "the next section"))
            if section["lines"] else LEAD_IN.get(section["kind"], "the next section")
        )
        if not section["lines"]:
            if index == 0:
                clause = (
                    f" {lead}: the vocalist remains completely silent "
                    "and the instruments carry this passage alone."
                )
            else:
                clause = (
                    f" [Shot {index + 1}] At {_stamp(section['start'])}, {lead}: the vocalist "
                    "remains completely silent and the instruments carry this passage alone."
                )
        else:
            sung = " ".join(f"<d>[{language}] {line}</d>" for line in section["lines"])
            if index == 0:
                clause = (
                    f" The vocalist sings {lead}, spacing the lyric lines "
                    f"naturally across this section (S1): {sung}"
                )
            else:
                clause = (
                    f" [Shot {index + 1}] At {_stamp(section['start'])}, the vocalist "
                    f"sings {lead}, spacing the lyric lines naturally across "
                    f"this section (S1): {sung}"
                )
        body += clause
    return body


def _natural_song_sheet_body(style: str, instrumentation: str, vocalist: str,
                             plan: list[dict], language: str, scene: str) -> str:
    """Music-first H3 prompt: one take, with song sections rather than fake cuts."""
    body = (
        f"[Shot 1] One uninterrupted original musical performance in {scene.rstrip('.')}. "
        "The camera and recording perspective remain stable for the entire take; there are no "
        "cuts and no change of performer, band, room, tempo, key, mix, or microphone position. "
        f"Speaker 1 (S1) is {vocalist.rstrip('.')}. Musical direction: {style.rstrip('.')}. "
        f"Arrangement: {instrumentation.rstrip('.')}. S1 performs the following song form in "
        "order with connected musical phrasing and natural transitions. Section names, time "
        "cues, performance notes, and all text outside <d> blocks are production directions "
        "only and remain completely inaudible."
    )
    for section in plan:
        lead = (
            SUNG_LEAD_IN.get(section["kind"], LEAD_IN.get(section["kind"], "the next section"))
            if section["lines"] else LEAD_IN.get(section["kind"], "the next section")
        )
        timing = f" From about {_stamp(section['start'])} to {_stamp(section['end'])},"
        if not section["lines"]:
            body += (
                f"{timing} {lead} is instrumental; S1 remains silent while the same "
                "arrangement continues without a cut."
            )
            continue
        # One block per musical section keeps short lyric lines connected instead
        # of turning every line into a separate stop-start utterance.
        sung = "\n".join(section["lines"])
        body += (
            f"{timing} S1 sings {lead} as one connected vocal passage with clear natural "
            f"English diction and no spoken introduction: <d>[{language}] {sung}</d>"
        )
    return body


def song_prompt(style: str, instrumentation: str, vocalist: str, lyrics: str,
                seconds: float, soundscape: str = "N/A",
                language: str = "English",
                scene: str = "a small intimate live room recorded in one take",
                profile: str = "natural_song_sheet_v2") -> tuple[str, list[dict]]:
    style = _plain_direction(style, "An original song with a coherent arrangement")
    instrumentation = _plain_direction(
        instrumentation, "A complete band arrangement supporting one vocalist")
    vocalist = _plain_direction(vocalist, "a natural expressive adult singer")
    soundscape = _plain_direction(soundscape, "N/A")
    scene = _plain_direction(scene, "a small intimate live room recorded in one take")
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"unknown song prompt profile: {profile}")
    plan = plan_sections(
        lyrics, seconds, word_aware=profile == "natural_song_sheet_v2")
    if profile == "petrock_timed_v1":
        body = _legacy_timed_song_body(
            style, instrumentation, vocalist, plan, language, scene)
    else:
        body = _natural_song_sheet_body(
            style, instrumentation, vocalist, plan, language, scene)
    body += (
        " The final musical note resolves cleanly. No voice description, style note, "
        "instrument name, section label, timestamp, instruction, or unmarked prompt "
        "text is ever sung or spoken."
    )
    return (
        f"integrated_multimodal_description: {body}\n\n"
        f"overall_soundscape: {soundscape}\n\n"
        "non_diegetic_music: N/A",
        plan,
    )


@dataclass
class MusicResult:
    job_id: str
    path: Path
    prompt: str
    duration_seconds: float
    render_seconds: float
    mode: str
    seed: int
    section_plan: list[dict]
    prompt_profile: str
    duration_mode: str
    duration_recommendation: dict | None


class MusicEngine:
    def __init__(self, comfy: ComfyClient) -> None:
        self.comfy = comfy

    def generate(self, *, mode: str, style: str, instrumentation: str,
                 lyrics: str = "", vocalist: str = "", soundscape: str = "N/A",
                 scene: str = "a small intimate live room recorded in one take",
                 duration_seconds: float = 15.0, steps: int = 20,
                 seed: int | None = None, language: str = "English",
                 response_format: str = "flac", duration_mode: str = "manual",
                 prompt_profile: str = "natural_song_sheet_v2") -> MusicResult:
        recommendation = None
        if mode == "song":
            recommendation = recommend_song_duration(style, lyrics)
            if duration_mode == "auto":
                duration_seconds = recommendation["recommended_seconds"]
        if duration_mode not in {"auto", "manual"}:
            raise ValueError("duration_mode must be auto or manual")
        frames = frames_for(duration_seconds)
        duration = frames / FPS
        if mode == "song":
            prompt, plan = song_prompt(
                style, instrumentation, vocalist, lyrics, duration,
                soundscape, language, scene, prompt_profile,
            )
        else:
            prompt = instrumental_prompt(style, instrumentation, soundscape)
            plan = []
        job_id = uuid.uuid4().hex
        prefix = f"voice_api/music/{job_id}/take"
        actual_seed = seed if seed is not None else random.randrange(1, 2**63)
        graph = t2va_graph(
            prompt, duration, steps, actual_seed, prefix, resolution=32)
        raw, elapsed = self.comfy.run(graph, {
            "kind": "music", "job_id": job_id, "mode": mode,
            "duration_seconds": round(duration, 3), "steps": steps,
        })
        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        native = job_dir / "music.flac"
        shutil.copy2(raw, native)
        output = native
        if response_format != "flac":
            output = job_dir / f"music.{response_format}"
            concat_audio([native], output, response_format, channels=2)
        report = {
            "job_id": job_id, "mode": mode, "duration_seconds": round(duration, 3),
            "steps": steps, "seed": actual_seed,
            "prompt_profile": prompt_profile if mode == "song" else "instrumental_v1",
            "duration_mode": duration_mode,
            "duration_recommendation": recommendation,
            "render_seconds": round(elapsed, 3),
            "output": str(output), "section_plan": plan, "prompt": prompt,
        }
        (job_dir / "music_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return MusicResult(
            job_id, output, prompt, duration, elapsed, mode, actual_seed, plan,
            prompt_profile if mode == "song" else "instrumental_v1",
            duration_mode, recommendation)
