from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import COMFY_INPUT, DATA_ROOT, JOB_ROOT, VOICE_ROOT, require_comfy_layout

BUILTIN_VOICES = {
    "warm_narrator": {
        "name": "Warm Narrator",
        "description": "An adult man with a natural warm baritone, measured pacing, gentle authority, and clean studio diction.",
        "gender": "male", "age_group": "middle_aged", "style_tags": ["warm", "narration", "grounded"],
    },
    "velvet_goth": {
        "name": "Velvet Goth",
        "description": "An adult woman with a realistic low velvety voice, restrained dark humor, intimate delivery, and natural conversational imperfections.",
        "gender": "female", "age_group": "adult", "style_tags": ["velvety", "intimate", "wry"],
    },
    "bright_assistant": {
        "name": "Bright Assistant",
        "description": "An adult woman with a clear, lively, friendly voice, medium pitch, precise diction, and an easy conversational rhythm.",
        "gender": "female", "age_group": "adult", "style_tags": ["bright", "friendly", "conversational"],
    },
    "young_hero": {
        "name": "Young Hero",
        "description": "A young adult man with an earnest clear tenor, energetic but controlled delivery, and natural modern American speech.",
        "gender": "male", "age_group": "young_adult", "style_tags": ["earnest", "energetic", "heroic"],
    },
    "elder_storyteller_f": {
        "name": "Grandmother Storyteller",
        "description": "An older adult woman with a rich weathered voice, warm humor, unhurried pacing, and expressive storytelling cadence.",
        "gender": "female", "age_group": "senior", "style_tags": ["weathered", "storytelling", "warm"],
    },
    "elder_storyteller_m": {
        "name": "Grandfather Storyteller",
        "description": "An older adult man with a textured mellow bass voice, patient pacing, gentle wit, and highly intelligible diction.",
        "gender": "male", "age_group": "senior", "style_tags": ["mellow", "storytelling", "patient"],
    },
    "soft_asmr": {
        "name": "Soft ASMR",
        "description": "An adult woman speaking softly and naturally very close to a studio microphone, warm breathy detail, relaxed pacing, never theatrical or robotic.",
        "gender": "female", "age_group": "adult", "style_tags": ["soft", "asmr", "close"],
    },
    "dry_detective": {
        "name": "Dry Detective",
        "description": "An adult man with a dry slightly gravelly mid-low voice, understated noir wit, deliberate phrasing, and realistic close-mic presence.",
        "gender": "male", "age_group": "adult", "style_tags": ["gravelly", "noir", "dry"],
    },
    "calm_professor": {
        "name": "Calm Professor",
        "description": "A middle-aged woman with a grounded articulate contralto, patient explanatory cadence, subtle warmth, and crisp natural diction.",
        "gender": "female", "age_group": "middle_aged", "style_tags": ["contralto", "calm", "articulate"],
    },
    "radio_host": {
        "name": "Radio Host",
        "description": "A middle-aged man with a polished resonant broadcast voice, confident timing, friendly energy, and clear contemporary diction.",
        "gender": "male", "age_group": "middle_aged", "style_tags": ["broadcast", "resonant", "confident"],
    },
    "gentle_caregiver": {
        "name": "Gentle Caregiver",
        "description": "An adult nonbinary speaker with a soft centered mid-range voice, reassuring warmth, relaxed pacing, and natural conversational delivery.",
        "gender": "nonbinary", "age_group": "adult", "style_tags": ["gentle", "centered", "reassuring"],
    },
    "comic_best_friend": {
        "name": "Comic Best Friend",
        "description": "A young adult woman with a lively slightly husky voice, quick comic timing, expressive reactions, and casual realistic speech.",
        "gender": "female", "age_group": "young_adult", "style_tags": ["comic", "husky", "casual"],
    },
    "grounded_actress": {
        "name": "Grounded Actress",
        "description": "An adult woman with a natural mid-low register, emotionally available professional acting, tiny breaths and hesitations, and completely unforced contemporary speech.",
        "gender": "female", "age_group": "adult", "style_tags": ["realistic", "dramatic", "grounded"],
    },
    "witty_british_f": {
        "name": "Witty British Woman",
        "description": "An adult English woman with a believable modern southern British accent, agile dry wit, precise but relaxed articulation, and intimate conversational timing.",
        "gender": "female", "age_group": "adult", "style_tags": ["british", "witty", "conversational"],
    },
    "smoky_jazz_f": {
        "name": "Smoky Jazz Woman",
        "description": "A middle-aged woman with a smoky textured lower voice, lived-in warmth, measured phrasing, and subtle late-night club performer presence without caricature.",
        "gender": "female", "age_group": "middle_aged", "style_tags": ["smoky", "textured", "late_night"],
    },
    "earnest_young_f": {
        "name": "Earnest Young Woman",
        "description": "A young adult woman with a natural clear voice, spontaneous emotional reactions, contemporary casual diction, and sincere energetic professional acting.",
        "gender": "female", "age_group": "young_adult", "style_tags": ["earnest", "youthful", "emotional"],
    },
    "mature_executive_f": {
        "name": "Mature Executive Woman",
        "description": "A mature woman with a composed resonant contralto, intelligent authority, nuanced warmth, and the relaxed confidence of a real person speaking one-to-one.",
        "gender": "female", "age_group": "middle_aged", "style_tags": ["authoritative", "mature", "contralto"],
    },
    "natural_neighbor_m": {
        "name": "Natural Neighbor",
        "description": "An adult man with an approachable mid-range voice, ordinary contemporary American speech, easy warmth, small natural hesitations, and no announcer polish.",
        "gender": "male", "age_group": "adult", "style_tags": ["natural", "casual", "friendly"],
    },
    "british_stage_m": {
        "name": "British Stage Actor",
        "description": "A middle-aged English man with a natural modern British accent, resonant but human tone, controlled theatrical skill, and subtle close-range emotional detail.",
        "gender": "male", "age_group": "middle_aged", "style_tags": ["british", "dramatic", "resonant"],
    },
    "soft_spoken_m": {
        "name": "Soft-Spoken Man",
        "description": "A young adult man with a gentle low-volume tenor, intimate realistic breath detail, thoughtful pauses, and calm contemporary conversational phrasing.",
        "gender": "male", "age_group": "young_adult", "style_tags": ["soft", "intimate", "thoughtful"],
    },
    "rugged_western_m": {
        "name": "Rugged Western Man",
        "description": "A mature American man with a lightly weathered baritone, understated western inflection, economical phrasing, dry warmth, and restrained realistic acting.",
        "gender": "male", "age_group": "middle_aged", "style_tags": ["rugged", "western", "understated"],
    },
    "animated_comic_m": {
        "name": "Animated Comic Man",
        "description": "A young adult man with an expressive flexible voice, excellent comic timing, spontaneous reactions, and energetic speech that remains recognizably human.",
        "gender": "male", "age_group": "young_adult", "style_tags": ["comic", "animated", "energetic"],
    },
    "cinematic_ai_companion_f": {
        "name": "Cinematic AI Companion",
        "description": "An adult woman with a low warm husky voice, close conversational presence, dry understated amusement, emotional intelligence, and nuanced cinematic acting; intimate and human rather than synthetic.",
        "gender": "female", "age_group": "adult", "style_tags": ["husky", "cinematic", "intimate", "ai_companion"],
    },
    "velvet_siren_f": {
        "name": "Velvet Siren",
        "description": "A mature adult woman with a smooth dark mezzo voice, confident sensual warmth, patient phrasing, subtle smiles in her tone, and realistic professional close-mic acting without breathy caricature.",
        "gender": "female", "age_group": "adult", "style_tags": ["sensual", "velvet", "confident", "mature"],
    },
    "attached_girlfriend_f": {
        "name": "Devoted Girlfriend",
        "description": "A young adult woman with a pretty natural contemporary voice, intensely affectionate focus, quick playful reactions, possessive little undercurrents, and emotionally believable romantic-comedy acting.",
        "gender": "female", "age_group": "young_adult", "style_tags": ["devoted", "playful", "intense", "romantic_comedy"],
    },
    "commanding_boss_f": {
        "name": "Commanding Boss",
        "description": "A middle-aged woman with a controlled resonant contralto, crisp decisive diction, amused authority, impeccable timing, and the grounded confidence of an elite dramatic actor speaking at close range.",
        "gender": "female", "age_group": "middle_aged", "style_tags": ["bossy", "authoritative", "contralto", "controlled"],
    },
}


@dataclass
class VoiceProfile:
    id: str
    name: str
    description: str
    builtin: bool = False
    reference_path: str | None = None
    transcript: str | None = None
    created_at: float | None = None
    gender: str | None = None
    age_group: str | None = None
    style_tags: list[str] | None = None

    @property
    def ready(self) -> bool:
        return bool(self.reference_path and Path(self.reference_path).is_file())

    def public(self) -> dict:
        result = asdict(self)
        result["ready"] = self.ready
        return result


def ensure_directories() -> None:
    for path in (DATA_ROOT, VOICE_ROOT, JOB_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def safe_id(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not value:
        raise ValueError("voice name must contain letters or numbers")
    return value[:64]


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower().replace("’", "'"))


def word_similarity(expected: str, heard: str) -> float:
    a, b = normalize_words(expected), normalize_words(heard)
    if not a:
        return 1.0 if not b else 0.0
    # ASR tokenizers legitimately disagree on closed compounds such as
    # "streetlights" / "street lights". A zero-cost two-token merge handles
    # that narrow case without forgiving an inserted, omitted, or changed word.
    rows, cols = len(a) + 1, len(b) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            options = [
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            ]
            if i >= 2 and a[i - 2] + a[i - 1] == b[j - 1]:
                options.append(distance[i - 2][j - 1])
            if j >= 2 and a[i - 1] == b[j - 2] + b[j - 1]:
                options.append(distance[i - 1][j - 2])
            distance[i][j] = min(options)
    return max(0.0, 1.0 - distance[-1][-1] / max(len(a), len(b), 1))


def _split_long_piece(piece: str, max_words: int) -> list[str]:
    words = piece.split()
    if len(words) <= max_words:
        return [piece.strip()] if piece.strip() else []
    result = []
    while words:
        take = min(max_words, len(words))
        if len(words) > max_words:
            floor = max(4, max_words // 2)
            for idx in range(take - 1, floor - 1, -1):
                if words[idx - 1].endswith((",", ";", ":", "—", "-")):
                    take = idx
                    break
        result.append(" ".join(words[:take]))
        words = words[take:]
    return result


def chunk_text(text: str, target_words: int = 18, max_words: int = 24) -> list[str]:
    """Split prose into short speakable clauses without losing any words."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    target_words = max(5, min(target_words, max_words))
    max_words = max(target_words, max_words)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence.split()) <= max_words:
            pieces.append(sentence)
            continue
        clauses = re.split(r"(?<=[,;:—])\s+", sentence)
        for clause in clauses:
            pieces.extend(_split_long_piece(clause, max_words))

    chunks: list[str] = []
    pending = ""
    for piece in pieces:
        candidate = f"{pending} {piece}".strip()
        if pending and len(candidate.split()) > max_words:
            chunks.append(pending)
            pending = piece.strip()
        else:
            pending = candidate
        if pending and len(pending.split()) >= target_words and pending[-1:] in ".!?":
            chunks.append(pending)
            pending = ""
    if pending:
        chunks.append(pending)
    return chunks


def run_checked(args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(args, input=input_bytes, capture_output=True)
    if proc.returncode:
        message = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(message or f"command failed: {args[0]}")
    return proc.stdout


def preprocess_reference(source: Path, destination: Path, seconds: float = 12.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-vn",
        "-t", str(min(seconds, 14.0)), "-ac", "2", "-ar", "32000",
        "-af", "highpass=f=60,loudnorm=I=-20:TP=-2:LRA=7",
        "-c:a", "flac", str(destination),
    ])


def preprocess_generated(source: Path, destination: Path, expected_seconds: float,
                         channels: int = 1) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # H3 often fills unused clip time with room tone. Keep a small natural pause,
    # but remove long silent heads/tails and cap pathological generations.
    cap = max(2.5, min(14.5, expected_seconds + 2.0))
    filters = (
        "highpass=f=55,"
        "silenceremove=start_periods=1:start_duration=0.04:start_threshold=-50dB:"
        "stop_periods=-1:stop_duration=0.65:stop_threshold=-50dB,"
        "afade=t=in:st=0:d=0.015,adelay=80:all=1,apad=pad_dur=0.30"
    )
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-t", f"{cap:.3f}", "-af", filters, "-ac", str(channels), "-ar", "32000",
        "-c:a", "flac", str(destination),
    ])


def _read_analysis_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read common audio directly, with FFmpeg fallback for odd containers."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError):
        decoded = run_checked([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2",
            "-ar", "32000", "pipe:1",
        ])
        values = np.frombuffer(decoded, dtype=np.float32)
        return values[:len(values) // 2 * 2].reshape(-1, 2), 32000


def onset_orphan_cut(path: Path, threshold_db: float = -45.0) -> float | None:
    """Locate a short startup burst separated from real speech by silence."""
    audio, rate = _read_analysis_audio(path)
    audio = audio[:round(rate * 3.0)]
    frame = max(1, round(rate * 0.010))
    usable = audio[:len(audio) // frame * frame]
    if not len(usable):
        return None
    framed = usable.reshape(-1, frame, audio.shape[1])
    rms = np.sqrt(np.mean(framed ** 2, axis=1) + 1e-12).max(axis=1)
    active = rms > 10 ** (threshold_db / 20)
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    substantial = [(a, b) for a, b in runs if b - a >= 10]
    if len(substantial) < 2:
        return None
    first_start, first_end = substantial[0]
    next_start, _ = substantial[1]
    if (first_start <= 15 and first_end - first_start <= 35
            and next_start - first_end >= 25):
        return max(0.0, next_start * 0.010 - 0.080)
    return None


def speech_activity_margins(path: Path, threshold_db: float = -42.0
                            ) -> tuple[float | None, float | None]:
    """Return audible head and tail silence using channel-aware RMS."""
    audio, rate = _read_analysis_audio(path)
    frame = max(1, round(rate * 0.005))
    usable = audio[:len(audio) // frame * frame]
    if not len(usable):
        return None, None
    framed = usable.reshape(-1, frame, audio.shape[1])
    rms = np.sqrt(np.mean(framed ** 2, axis=1) + 1e-12).max(axis=1)
    indices = np.flatnonzero(rms > 10 ** (threshold_db / 20))
    if not len(indices):
        return None, None
    return (float(indices[0] * frame / rate),
            float((len(rms) - 1 - indices[-1]) * frame / rate))


def audio_signal_metrics(path: Path) -> dict:
    """Cheap non-linguistic checks for artifacts ASR cannot perceive."""
    audio, _rate = _read_analysis_audio(path)
    if not len(audio):
        return {"peak_dbfs": -240.0, "clipped_sample_fraction": 0.0,
                "dc_offset": 0.0, "onset_orphan_cut_seconds": None}
    peak = max(float(np.max(np.abs(audio))), 1e-12)
    return {
        "peak_dbfs": round(20 * math.log10(peak), 2),
        "clipped_sample_fraction": round(
            float((np.abs(audio) >= 0.999).mean()), 8),
        "dc_offset": round(float(np.max(np.abs(audio.mean(axis=0)))), 8),
        "onset_orphan_cut_seconds": onset_orphan_cut(path),
    }


def remove_onset_orphan(source: Path, destination: Path,
                        channels: int = 2) -> float:
    cut = onset_orphan_cut(source)
    if cut is None:
        return 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{cut:.3f}",
        "-i", str(source), "-af",
        "afade=t=in:st=0:d=0.02,adelay=80:all=1,apad=pad_dur=0.30",
        "-ac", str(channels), "-ar", "32000", "-c:a", "flac",
        str(destination),
    ])
    return cut


def audio_to_pcm(path: Path, channels: int = 1) -> bytes:
    return run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le",
        "-acodec", "pcm_s16le", "-ac", str(channels), "-ar", "32000", "pipe:1",
    ])


def decoded_audio_duration(path: Path) -> float:
    decoded = run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
        "-ar", "16000", "pipe:1",
    ])
    return len(decoded) / (2 * 16000)


def trim_aligned_audio(source: Path, destination: Path, start: float,
                       end: float | None, channels: int = 1) -> None:
    """Cut ASR-detected lead-in/tail speech and make the new seam click-free."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}"]
    if end is not None:
        command += ["-to", f"{end:.3f}"]
    command += [
        "-i", str(source), "-af",
        "afade=t=in:st=0:d=0.02,adelay=80:all=1,apad=pad_dur=0.30",
        "-ac", str(channels), "-ar", "32000", "-c:a", "flac", str(destination),
    ]
    run_checked(command)


def trim_leading_audio(source: Path, destination: Path, start: float,
                       channels: int = 2) -> None:
    """Remove generated pre-roll without changing the tail or adding padding."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}",
        "-i", str(source), "-af", "afade=t=in:st=0:d=0.012",
        "-ac", str(channels), "-ar", "32000", "-c:a", "flac",
        str(destination),
    ])


def retime_audio(source: Path, destination: Path, tempo: float,
                 channels: int = 2) -> None:
    """Slow rushed H3 speech without changing pitch, using two gentle stages."""
    tempo = max(0.55, min(1.0, tempo))
    stage = math.sqrt(tempo)
    filters = (
        f"atempo={stage:.6f},atempo={stage:.6f},"
        "afade=t=in:st=0:d=0.012,apad=pad_dur=0.25"
    )
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-af", filters, "-ac", str(channels), "-ar", "32000",
        "-c:a", "flac", str(destination),
    ])


def fit_audio_duration(source: Path, destination: Path, target_seconds: float,
                       channels: int = 2) -> float:
    """Pitch-preserving whole-file duration fit with conservative limits."""
    actual = float(run_checked([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(source),
    ]).decode().strip())
    tempo = actual / target_seconds
    if not 0.75 <= tempo <= 1.35:
        raise ValueError(
            "target duration requires an unsafe tempo change "
            f"({actual:.3f}s to {target_seconds:.3f}s; allowed 0.75x-1.35x)")
    suffix = destination.suffix.casefold()
    render_seconds = (max(0.1, target_seconds - 1024 / 32000)
                      if suffix == ".aac" else target_seconds)
    filters = (f"atempo={tempo:.8f},apad=pad_dur=0.10,"
               f"atrim=duration={render_seconds:.6f},"
               f"afade=t=out:st={max(0.0, render_seconds - 0.015):.6f}:d=0.015")
    rate = "48000" if suffix == ".opus" else "32000"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
               "-af", filters, "-ar", rate, "-ac", str(channels)]
    codecs = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
              ".opus": ["-c:a", "libopus", "-b:a", "128k"],
              ".aac": ["-c:a", "aac", "-b:a", "192k"],
              ".wav": ["-c:a", "pcm_s16le"]}
    command += codecs.get(suffix, ["-c:a", "flac"])
    command.append(str(destination))
    run_checked(command)
    return tempo


def concat_audio(paths: Iterable[Path], destination: Path, fmt: str,
                 channels: int = 1, crossfade_seconds: float = 0.06) -> None:
    paths = list(paths)
    if not paths:
        raise ValueError("no chunks to concatenate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for path in paths:
        command += ["-i", str(path)]
    if len(paths) == 1:
        audio_filter = "[0:a]anull[out]"
    elif crossfade_seconds <= 0:
        inputs = "".join(f"[{index}:a]" for index in range(len(paths)))
        audio_filter = f"{inputs}concat=n={len(paths)}:v=0:a=1[out]"
    else:
        stages = []
        previous = "0:a"
        for index in range(1, len(paths)):
            label = "out" if index == len(paths) - 1 else f"join{index}"
            stages.append(
                f"[{previous}][{index}:a]acrossfade=d={crossfade_seconds:.4f}:"
                f"c1=tri:c2=tri[{label}]"
            )
            previous = label
        audio_filter = ";".join(stages)
    output_rate = "48000" if fmt == "opus" else "32000"
    command += ["-filter_complex", audio_filter, "-map", "[out]",
                "-ar", output_rate, "-ac", str(channels)]
    if fmt == "mp3":
        command += ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif fmt == "opus":
        command += ["-c:a", "libopus", "-b:a", "128k"]
    elif fmt == "aac":
        command += ["-c:a", "aac", "-b:a", "192k"]
    elif fmt == "wav":
        command += ["-c:a", "pcm_s16le"]
    else:
        command += ["-c:a", "flac"]
    command.append(str(destination))
    run_checked(command)


def speech_join_pause_seconds(text: str) -> float:
    ending = text.rstrip()[-1:] if text.strip() else ""
    if ending in ".!?":
        return 0.42
    if ending in ",;:—-":
        return 0.24
    return 0.16


def trim_audio_for_join(source: Path, destination: Path, *, start: float = 0.0,
                        end: float | None = None, channels: int = 2) -> None:
    filters = [f"atrim=start={max(0.0, start):.6f}", "asetpts=PTS-STARTPTS"]
    if end is not None:
        filters[0] += f":end={max(start + 0.05, end):.6f}"
    if start > 0:
        filters.append("afade=t=in:st=0:d=0.010")
    if end is not None:
        duration = max(0.05, end - start)
        filters.append(
            f"afade=t=out:st={max(0.0, duration - 0.010):.6f}:d=0.010")
    run_checked([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
        "-af", ",".join(filters), "-ar", "32000", "-ac", str(channels),
        "-c:a", "flac", str(destination),
    ])



class VoiceRegistry:
    def __init__(self) -> None:
        ensure_directories()
        self._lock = threading.RLock()

    def _manifest(self, voice_id: str) -> Path:
        return VOICE_ROOT / voice_id / "voice.json"

    def save(self, profile: VoiceProfile) -> VoiceProfile:
        with self._lock:
            manifest = self._manifest(profile.id)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
        return profile

    def get(self, voice_id: str) -> VoiceProfile:
        voice_id = safe_id(voice_id)
        manifest = self._manifest(voice_id)
        if manifest.is_file():
            profile = VoiceProfile(**json.loads(manifest.read_text(encoding="utf-8")))
            preset = BUILTIN_VOICES.get(voice_id)
            if preset:
                profile.builtin = True
                profile.gender = profile.gender or preset.get("gender")
                profile.age_group = profile.age_group or preset.get("age_group")
                profile.style_tags = profile.style_tags or preset.get("style_tags")
            return profile
        if voice_id in BUILTIN_VOICES:
            preset = BUILTIN_VOICES[voice_id]
            reference = VOICE_ROOT / voice_id / "reference.flac"
            return VoiceProfile(
                id=voice_id, name=preset["name"], description=preset["description"],
                builtin=True,
                reference_path=str(reference) if reference.is_file() else None,
                gender=preset.get("gender"), age_group=preset.get("age_group"),
                style_tags=preset.get("style_tags"))
        raise KeyError(voice_id)

    def list(self) -> list[VoiceProfile]:
        ids = set(BUILTIN_VOICES)
        ids.update(path.parent.name for path in VOICE_ROOT.glob("*/voice.json"))
        return [self.get(voice_id) for voice_id in sorted(ids)]

    def create(self, name: str, description: str, source: Path,
               transcript: str | None = None) -> VoiceProfile:
        require_comfy_layout()
        voice_id = safe_id(name)
        if voice_id in BUILTIN_VOICES:
            raise ValueError(f"'{voice_id}' is reserved for a built-in voice")
        reference = VOICE_ROOT / voice_id / "reference.flac"
        preprocess_reference(source, reference)
        comfy_copy = COMFY_INPUT / voice_id / "reference.flac"
        preprocess_reference(reference, comfy_copy)
        return self.save(VoiceProfile(voice_id, name.strip(), description.strip(), False,
                                     str(reference), transcript, time.time()))

    def from_description(self, description: str, name: str | None = None) -> VoiceProfile:
        description = description.strip()
        if not description:
            raise ValueError("voice description cannot be empty")
        digest = hashlib.sha256(description.casefold().encode("utf-8")).hexdigest()[:12]
        voice_id = f"described_{digest}"
        try:
            return self.get(voice_id)
        except KeyError:
            return self.save(VoiceProfile(
                voice_id, (name or "Described Voice").strip(), description,
                False, None, None, time.time()))

    def attach_reference(self, profile: VoiceProfile, source: Path,
                         transcript: str | None = None) -> VoiceProfile:
        require_comfy_layout()
        reference = VOICE_ROOT / profile.id / "reference.flac"
        preprocess_reference(source, reference)
        comfy_copy = COMFY_INPUT / profile.id / "reference.flac"
        preprocess_reference(reference, comfy_copy)
        profile.reference_path = str(reference)
        profile.transcript = transcript
        profile.created_at = profile.created_at or time.time()
        return self.save(profile)

    def comfy_reference(self, profile: VoiceProfile) -> str:
        require_comfy_layout()
        if not profile.ready:
            raise ValueError(f"voice '{profile.id}' has no reference yet")
        relative = Path(profile.id) / "reference.flac"
        destination = COMFY_INPUT / relative
        if not destination.is_file():
            preprocess_reference(Path(profile.reference_path), destination)
        return str(Path("voice_api") / relative)
