from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from itertools import chain
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import JOB_ROOT, MAX_TRANSCRIPTION_BYTES, MAX_TRANSCRIPTION_SECONDS
from .core import VoiceRegistry, audio_to_pcm
from .engine import SpeechFidelityError, VoiceEngine, audio_duration
from .music import MusicEngine, recommend_song_duration
from .music_presets import MUSIC_PRESETS
from .sfx import (
    MAX_SECONDS as SFX_MAX_SECONDS,
)
from .sfx import (
    MIN_SECONDS as SFX_MIN_SECONDS,
)
from .sfx import (
    SfxEngine,
    recommend_sfx_duration,
)
from .sfx_presets import SFX_PRESETS
from .transcription import subtitles, verbose_transcription

app = FastAPI(
    title="MiniMax H3 Voice API",
    version=__version__,
    description=(
        "Long-form speech, music, and first-class SFX/Foley over MiniMax H3 in "
        "ComfyUI, with Parakeet verification for speech."
    ),
)
registry = VoiceRegistry()
engine = VoiceEngine(registry=registry)
music_engine = MusicEngine(engine.comfy)
sfx_engine = SfxEngine(engine.comfy)
WEB_ROOT = Path(__file__).resolve().parent / "web"
ASR_MODEL = "nemo-parakeet-tdt-0.6b-v2-int8"
DEFAULT_SFX_DESCRIPTION = (
    "A single clean diegetic sound effect with a natural attack and short decay"
)
DEFAULT_SFX_SPACE = "a neutral dry recording stage with no visible source"
app.mount("/studio-assets", StaticFiles(directory=WEB_ROOT), name="studio-assets")


class SpeechRequest(BaseModel):
    model: str = "minimax-h3-ref2va"
    speech_mode: Literal["reference", "continuous"] = "reference"
    input: str = Field(min_length=1)
    voice: str = "warm_narrator"
    voice_description: str | None = None
    response_format: Literal["flac", "wav", "mp3", "opus", "aac", "pcm"] = "flac"
    stream: bool = False
    instructions: str | None = None
    direction: str = "natural conversational delivery with clear English diction"
    emotion: str = "neutral"
    emotion_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str = "English"
    speed: float = Field(default=1.0, ge=0.65, le=1.5)
    steps: int = Field(default=30, ge=10, le=50)
    resolution: Literal[32, 64, 128] = 32
    continuity_context_seconds: float = Field(default=10.0, ge=0.92, le=10.0)
    spatial_preset: Literal[
        "studio", "close", "across_table", "living_room", "bedside", "stage"
    ] = "close"
    channels: Literal[1, 2] = 2
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    target_words: int = Field(default=18, ge=5, le=24)
    max_words: int = Field(default=24, ge=7, le=30)
    verify: bool = True
    fidelity_mode: Literal["exact", "similarity", "unverified"] = "exact"
    max_retries: int = Field(default=3, ge=0, le=5)
    candidate_pool_size: int = Field(default=1, ge=1, le=3)
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    strict_fidelity: bool = True
    max_words_per_minute: float = Field(default=225.0, ge=100.0, le=400.0)
    generation_padding_seconds: float = Field(default=0.10, ge=0.0, le=3.0)
    audio_ref_keep: float = Field(default=1.0, ge=0.90, le=1.0)
    target_duration_seconds: float | None = Field(default=None, ge=1.0, le=300.0)
    dit: Literal["stock", "eros"] = "stock"
    video_vae: str | None = None


class VoiceSettings(BaseModel):
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.5, ge=0.0, le=1.0)
    speed: float = Field(default=1.0, ge=0.65, le=1.5)
    use_speaker_boost: bool = True


class ElevenSpeechRequest(BaseModel):
    text: str = Field(min_length=1)
    model_id: str = "minimax-h3-ref2va"
    language_code: str = "English"
    voice_settings: VoiceSettings = Field(default_factory=VoiceSettings)
    emotion: str = "neutral"
    instructions: str = "natural conversational delivery with clear diction"
    # Also accepts ElevenLabs-style values such as mp3_44100_128 and
    # pcm_32000; H3 is natively returned at 32 kHz.
    output_format: str = "mp3_44100_128"
    verify: bool = True


class MusicRequest(BaseModel):
    mode: Literal["instrumental", "song"] = "song"
    title: str = "H3 music"
    preset: str | None = None
    style: str = Field(default="An original coherent musical performance", min_length=1)
    instrumentation: str = Field(
        default="A restrained arrangement with a clear melodic identity",
        min_length=1,
    )
    vocalist: str = "a natural expressive adult singer"
    lyrics: str = ""
    soundscape: str = "N/A"
    scene: str = "a small intimate live room recorded in one take"
    language: str = "English"
    duration_seconds: float = Field(default=60.0, ge=5.0, le=60.0)
    duration_mode: Literal["auto", "manual"] = "auto"
    prompt_profile: Literal["natural_song_sheet_v2", "petrock_timed_v1"] = (
        "natural_song_sheet_v2"
    )
    steps: int = Field(default=20, ge=10, le=50)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    response_format: Literal["flac", "wav", "mp3", "opus", "aac"] = "flac"


class SfxRequest(BaseModel):
    title: str = "H3 sfx"
    preset: str | None = None
    description: str = Field(default=DEFAULT_SFX_DESCRIPTION, min_length=1)
    kind: Literal["oneshot", "loop_bed"] = "oneshot"
    space: str = DEFAULT_SFX_SPACE
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)
    duration_seconds: float = Field(default=3.0, ge=SFX_MIN_SECONDS, le=SFX_MAX_SECONDS)
    duration_mode: Literal["auto", "manual"] = "auto"
    prompt_profile: Literal["diegetic_v1"] = "diegetic_v1"
    steps: int = Field(default=20, ge=10, le=50)
    resolution: Literal[32, 64, 128] = 32
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    response_format: Literal["flac", "wav", "mp3", "opus", "aac"] = "flac"


EMOTIONS = [
    "neutral", "happy", "sad", "angry", "excited", "fearful", "tender",
    "playful", "authoritative", "tired", "sarcastic", "intimate", "whispering",
]


@app.get("/", include_in_schema=False)
def studio() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "comfyui": "online" if engine.comfy.health() else "offline",
        "parakeet": "cpu-hot" if engine.verifier._model is not None else "cpu-lazy",
        "gpu_policy": "H3 jobs serialized; ASR uses CPUExecutionProvider",
        "queue": engine.comfy.queue.public(),
        "warmth": engine.comfy.warmth(asr_loaded=engine.verifier._model is not None),
    }


@app.get("/v1/warmth")
def warmth() -> dict:
    """Show whether H3 and the CPU verifier are resident for the next call."""
    return engine.comfy.warmth(asr_loaded=engine.verifier._model is not None)


@app.get("/v1/voices")
def voices() -> dict:
    return {"data": [profile.public() for profile in registry.list()]}


@app.get("/v1/models")
def models() -> dict:
    return {"data": [
        {
            "id": "minimax-h3-ref2va", "object": "model", "owned_by": "local",
            "capabilities": ["speech", "voice-cloning", "streaming", "long-form",
                             "emotion", "spatial-presence", "resolution-control"],
            "dit": "stock",
        },
        {
            "id": "minimax-h3-ref2va-eros", "object": "model", "owned_by": "local",
            "capabilities": ["speech", "voice-cloning", "streaming", "long-form",
                             "emotion", "spatial-presence", "resolution-control"],
            "dit": "eros",
        },
        {
            "id": "minimax-h3-fl2va-continuous", "object": "model",
            "owned_by": "local",
            "capabilities": ["speech", "described-voice", "latent-continuation",
                             "long-form", "emotion", "spatial-presence"],
        },
        {
            "id": "minimax-h3-sfx", "object": "model", "owned_by": "local",
            "capabilities": ["sfx", "foley", "ambience", "oneshot", "loop-bed",
                             "presets"],
        },
        {
            "id": ASR_MODEL, "object": "model", "owned_by": "local",
            "capabilities": ["transcription", "word-timestamps", "subtitles",
                             "audio-input", "video-input", "cpu-only"],
        },
    ]}


async def _save_transcription_upload(file: UploadFile, destination: Path) -> int:
    """Stream an upload to disk while enforcing a deterministic size limit."""
    written = 0
    with destination.open("wb") as output:
        while block := await file.read(1024 * 1024):
            written += len(block)
            if written > MAX_TRANSCRIPTION_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=("transcription upload exceeds "
                            f"{MAX_TRANSCRIPTION_BYTES // (1024 * 1024)} MB"),
                )
            output.write(block)
    if written == 0:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")
    return written


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(ASR_MODEL),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: Literal["json", "text", "verbose_json", "srt", "vtt"] = Form("json"),
    temperature: float = Form(0.0),
):
    """Transcribe uploaded audio or video with CPU-only Parakeet int8."""
    del prompt  # Accepted for OpenAI client compatibility; Parakeet has no prompt bias.
    if model not in {ASR_MODEL, "parakeet-tdt-0.6b-v2", "whisper-1"}:
        raise HTTPException(status_code=422, detail=f"unsupported transcription model: {model}")
    if language and language.strip().casefold() not in {"en", "eng", "english", "auto"}:
        raise HTTPException(
            status_code=422,
            detail="the installed Parakeet TDT 0.6B v2 model transcribes English only",
        )
    if temperature != 0:
        raise HTTPException(
            status_code=422,
            detail="Parakeet decoding is deterministic; temperature must be 0",
        )

    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="h3_asr_upload_") as temp_dir:
        source = Path(temp_dir) / f"upload{suffix}"
        await _save_transcription_upload(file, source)
        try:
            duration = audio_duration(source)
            if duration <= 0:
                raise ValueError("the uploaded file contains no decodable audio")
            if duration > MAX_TRANSCRIPTION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail=(f"audio duration exceeds the {MAX_TRANSCRIPTION_SECONDS:g} "
                            "second transcription limit"),
                )
            started = time.perf_counter()
            # Keep FastAPI responsive to health/queue/browser polling while
            # CPU inference runs. The verifier itself serializes ONNX calls.
            result = await asyncio.to_thread(
                engine.verifier.transcribe_timestamped, source)
            elapsed = time.perf_counter() - started
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"could not transcribe upload: {exc}") from exc

    verbose = verbose_transcription(result, duration, ASR_MODEL)
    headers = {
        "X-ASR-Model": ASR_MODEL,
        "X-ASR-Device": "cpu-int8",
        "X-ASR-Seconds": f"{elapsed:.3f}",
        "X-Audio-Duration-Seconds": f"{duration:.3f}",
        "X-ASR-RTFx": f"{duration / max(elapsed, 0.001):.3f}",
    }
    if response_format == "text":
        return PlainTextResponse(verbose["text"] + "\n", headers=headers)
    if response_format in {"srt", "vtt"}:
        media_type = "application/x-subrip" if response_format == "srt" else "text/vtt"
        return PlainTextResponse(
            subtitles(verbose["segments"], response_format),
            media_type=media_type, headers=headers,
        )
    if response_format == "verbose_json":
        return JSONResponse(verbose, headers=headers)
    return JSONResponse({"text": verbose["text"]}, headers=headers)


@app.get("/v1/audio/emotions")
def emotions() -> dict:
    return {"data": EMOTIONS}


@app.get("/v1/audio/music/presets")
def music_presets() -> dict:
    return {"data": [{"id": key, **value} for key, value in MUSIC_PRESETS.items()]}


@app.get("/v1/audio/music/jobs/{job_id}/prompt")
def music_prompt(job_id: str) -> dict:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="music job not found")
    report_path = JOB_ROOT / job_id / "music_report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="music job not found")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="music report is unreadable") from exc
    return {
        "job_id": job_id,
        "prompt": report.get("prompt", ""),
        "section_plan": report.get("section_plan", []),
        "duration_seconds": report.get("duration_seconds"),
        "mode": report.get("mode"),
        "seed": report.get("seed"),
        "prompt_profile": report.get("prompt_profile"),
        "duration_mode": report.get("duration_mode"),
        "duration_recommendation": report.get("duration_recommendation"),
    }


@app.post("/v1/audio/music/duration")
def music_duration(request: MusicRequest) -> dict:
    """Plan lyric timing without loading or running H3."""
    preset = MUSIC_PRESETS.get(request.preset or "")
    style = (
        preset["style"]
        if preset and request.style == MusicRequest.model_fields["style"].default
        else request.style
    )
    if request.mode != "song":
        return {
            "recommended_seconds": request.duration_seconds,
            "mode": "instrumental",
            "reason": "Instrumentals use the selected manual duration.",
        }
    try:
        return {"mode": "song", **recommend_song_duration(style, request.lyrics)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/audio/music")
def music(request: MusicRequest):
    preset = MUSIC_PRESETS.get(request.preset or "")
    style = preset["style"] if preset and request.style == MusicRequest.model_fields["style"].default else request.style
    instrumentation = (preset["instrumentation"]
                       if preset and request.instrumentation == MusicRequest.model_fields["instrumentation"].default
                       else request.instrumentation)
    vocalist = (preset["vocalist"]
                if preset and request.vocalist == MusicRequest.model_fields["vocalist"].default
                else request.vocalist)
    try:
        result = music_engine.generate(
            mode=request.mode, style=style,
            instrumentation=instrumentation, lyrics=request.lyrics,
            vocalist=vocalist, soundscape=request.soundscape,
            scene=request.scene,
            duration_seconds=request.duration_seconds, steps=request.steps,
            seed=request.seed, language=request.language,
            response_format=request.response_format,
            duration_mode=request.duration_mode,
            prompt_profile=request.prompt_profile,
        )
        media = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg",
                 "opus": "audio/ogg", "aac": "audio/aac"}[request.response_format]
        safe_title = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in request.title
        ).strip("_")[:64] or "h3_music"
        return FileResponse(
            result.path, media_type=media,
            filename=f"{safe_title}.{request.response_format}",
            headers={
                "X-H3-Mode": result.mode,
                "X-H3-Job-Id": result.job_id,
                "X-H3-Duration-Seconds": f"{result.duration_seconds:.3f}",
                "X-H3-Render-Seconds": f"{result.render_seconds:.3f}",
                "X-H3-Sections": str(len(result.section_plan)),
                "X-H3-Seed": str(result.seed),
                "X-H3-Prompt-Profile": result.prompt_profile,
                "X-H3-Duration-Mode": result.duration_mode,
                "X-H3-Recommended-Seconds": str(
                    (result.duration_recommendation or {}).get(
                        "recommended_seconds", request.duration_seconds)
                ),
                "X-H3-Warm": "true",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _merge_sfx_request(request: SfxRequest) -> tuple[str, str, str, float]:
    """Fill description/kind/space/duration from a preset when caller left defaults."""
    preset = SFX_PRESETS.get(request.preset or "")
    description = request.description
    kind = request.kind
    space = request.space
    duration_seconds = request.duration_seconds
    using_stock_description = description == DEFAULT_SFX_DESCRIPTION
    if preset and using_stock_description:
        description = str(preset["description"])
        if preset.get("kind") in {"oneshot", "loop_bed"}:
            kind = str(preset["kind"])
        if preset.get("space"):
            space = str(preset["space"])
        if (
            request.duration_mode == "auto"
            and request.duration_seconds
            == SfxRequest.model_fields["duration_seconds"].default
            and preset.get("default_seconds") is not None
        ):
            duration_seconds = float(preset["default_seconds"])
    elif preset and space == DEFAULT_SFX_SPACE and preset.get("space"):
        space = str(preset["space"])
    return description, kind, space, duration_seconds


@app.get("/v1/audio/sfx/presets")
def sfx_presets() -> dict:
    return {"data": [{"id": key, **value} for key, value in SFX_PRESETS.items()]}


@app.get("/v1/audio/sfx/jobs/{job_id}/prompt")
def sfx_prompt_report(job_id: str) -> dict:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="sfx job not found")
    report_path = JOB_ROOT / job_id / "sfx_report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="sfx job not found")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="sfx report is unreadable") from exc
    return {
        "job_id": job_id,
        "prompt": report.get("prompt", ""),
        "duration_seconds": report.get("duration_seconds"),
        "kind": report.get("kind"),
        "preset": report.get("preset"),
        "seed": report.get("seed"),
        "intensity": report.get("intensity"),
        "prompt_profile": report.get("prompt_profile"),
        "duration_mode": report.get("duration_mode"),
        "duration_recommendation": report.get("duration_recommendation"),
        "description": report.get("description"),
        "space": report.get("space"),
    }


@app.post("/v1/audio/sfx/duration")
def sfx_duration(request: SfxRequest) -> dict:
    """Plan SFX length without loading or running H3."""
    description, kind, _space, duration_seconds = _merge_sfx_request(request)
    try:
        plan = recommend_sfx_duration(description, kind=kind)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.duration_mode == "manual":
        return {
            **plan,
            "recommended_seconds": duration_seconds,
            "mode": "manual",
            "reason": "Manual duration preserves the caller-selected length.",
        }
    if (
        request.preset
        and request.description == DEFAULT_SFX_DESCRIPTION
        and SFX_PRESETS.get(request.preset or "", {}).get("default_seconds") is not None
        and request.duration_seconds == SfxRequest.model_fields["duration_seconds"].default
    ):
        plan = {
            **plan,
            "recommended_seconds": float(
                SFX_PRESETS[request.preset]["default_seconds"]),
            "preset_default": True,
        }
    return plan


@app.post("/v1/audio/sfx")
def sfx(request: SfxRequest):
    description, kind, space, duration_seconds = _merge_sfx_request(request)
    if request.preset and request.preset not in SFX_PRESETS:
        raise HTTPException(status_code=422, detail=f"unknown sfx preset: {request.preset}")
    try:
        result = sfx_engine.generate(
            description=description,
            kind=kind,
            space=space,
            intensity=request.intensity,
            duration_seconds=duration_seconds,
            duration_mode=request.duration_mode,
            steps=request.steps,
            seed=request.seed,
            response_format=request.response_format,
            prompt_profile=request.prompt_profile,
            resolution=request.resolution,
            preset=request.preset,
        )
        media = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg",
                 "opus": "audio/ogg", "aac": "audio/aac"}[request.response_format]
        safe_title = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in request.title
        ).strip("_")[:64] or "h3_sfx"
        return FileResponse(
            result.path, media_type=media,
            filename=f"{safe_title}.{request.response_format}",
            headers={
                "X-H3-Mode": "sfx",
                "X-H3-Sfx-Kind": result.kind,
                "X-H3-Job-Id": result.job_id,
                "X-H3-Duration-Seconds": f"{result.duration_seconds:.3f}",
                "X-H3-Render-Seconds": f"{result.render_seconds:.3f}",
                "X-H3-Seed": str(result.seed),
                "X-H3-Prompt-Profile": result.prompt_profile,
                "X-H3-Duration-Mode": result.duration_mode,
                "X-H3-Intensity": f"{result.intensity:.3f}",
                "X-H3-Recommended-Seconds": str(
                    (result.duration_recommendation or {}).get(
                        "recommended_seconds", request.duration_seconds)
                ),
                "X-H3-Warm": "true",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    description: str = Form("A natural expressive adult speaker."),
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "reference.bin").suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="h3_voice_upload_") as temp_dir:
        source = Path(temp_dir) / f"upload{suffix}"
        with source.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        try:
            # ffmpeg accepts both audio and video; it extracts and normalizes speech.
            profile = registry.create(name, description, source)
            profile.transcript = engine.verifier.transcribe(Path(profile.reference_path))
            registry.save(profile)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return profile.public()


@app.post("/v1/voices/add")
async def add_voice_elevenlabs(
    name: str = Form(...),
    description: str = Form("A natural expressive adult speaker."),
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="at least one audio or video file is required")
    # H3 accepts a single <=15-second voice anchor. The first supplied file is
    # normalized into that canonical reference; additional samples are ignored.
    result = await create_voice(name=name, description=description, file=files[0])
    return {"voice_id": result["id"], "requires_verification": False, "voice": result}


@app.get("/v1/voices/{voice_id}")
def get_voice(voice_id: str) -> dict:
    try:
        return registry.get(voice_id).public()
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="voice not found") from exc


@app.get("/v1/audio/speech/jobs/{job_id}/report")
def speech_report(job_id: str) -> dict:
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise HTTPException(status_code=404, detail="speech job not found")
    job = JOB_ROOT / job_id
    report = job / "report.json"
    quality = job / "final_quality.json"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="speech job not found")
    return {
        "job_id": job_id,
        "chunks": json.loads(report.read_text(encoding="utf-8")),
        "final_quality": (json.loads(quality.read_text(encoding="utf-8"))
                          if quality.is_file() else None),
    }


def _resolve_speech_dit(request: SpeechRequest) -> str:
    if request.dit == "eros" or "eros" in request.model.casefold():
        return "eros"
    return "stock"


def _speech_response(request: SpeechRequest):
    if request.candidate_pool_size > request.max_retries + 1:
        raise HTTPException(
            status_code=422,
            detail="candidate_pool_size cannot exceed max_retries + 1 attempts")
    if request.stream and request.target_duration_seconds is not None:
        raise HTTPException(
            status_code=422,
            detail="target_duration_seconds requires non-streaming output")
    if request.speech_mode == "continuous" and request.target_duration_seconds is not None:
        raise HTTPException(
            status_code=422,
            detail="target_duration_seconds currently requires reference speech mode")
    try:
        if request.voice_description:
            profile = registry.from_description(request.voice_description, request.voice)
        else:
            profile = registry.get(request.voice)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="voice not found; enroll it with POST /v1/voices") from exc

    base_direction = request.instructions or request.direction
    emotion_direction = (
        f"{base_direction.rstrip('.')}; a {request.emotion} emotional reading at "
        f"{request.emotion_intensity:.0%} expressive intensity"
    )
    fidelity_mode = request.fidelity_mode
    if not request.verify:
        fidelity_mode = "unverified"
    if (request.language.strip().casefold() != "english"
            and fidelity_mode != "unverified"):
        raise HTTPException(
            status_code=422,
            detail="exact/similarity verification currently requires English; "
                   "select fidelity_mode=unverified explicitly")
    verify = fidelity_mode != "unverified"
    kwargs = dict(
        text=request.input, voice=profile.id, direction=emotion_direction,
        steps=request.steps, speed=request.speed, target_words=request.target_words,
        max_words=request.max_words, verify=verify,
        max_retries=request.max_retries, min_similarity=request.min_similarity,
        strict_fidelity=request.strict_fidelity, language=request.language,
        candidate_pool_size=request.candidate_pool_size,
        fidelity_mode=fidelity_mode,
        max_words_per_minute=request.max_words_per_minute,
        generation_padding_seconds=request.generation_padding_seconds,
        audio_ref_keep=request.audio_ref_keep,
        resolution=request.resolution, space=request.spatial_preset,
        seed=request.seed, channels=request.channels,
        dit=_resolve_speech_dit(request),
        video_vae=request.video_vae,
    )
    try:
        if request.speech_mode == "continuous":
            if request.stream:
                raise HTTPException(
                    status_code=422,
                    detail="continuous FL2VA currently returns after its latent chain completes; use stream=false",
                )
            completed = engine.generate_continuous(
                text=request.input, voice=profile.id, direction=emotion_direction,
                steps=request.steps, speed=request.speed,
                target_words=min(request.target_words, 20),
                max_words=min(request.max_words, 22), verify=verify,
                min_similarity=request.min_similarity,
                strict_fidelity=request.strict_fidelity,
                language=request.language, resolution=request.resolution,
                space=request.spatial_preset,
                context_seconds=request.continuity_context_seconds,
                seed=request.seed,
                channels=request.channels,
                video_vae=request.video_vae,
            )
            total_render = sum(item.render_seconds for item in completed)
            minimum_similarity = min(item.similarity for item in completed)
            fmt = request.response_format
            output_format = "flac" if fmt == "pcm" else fmt
            # MotionContextTrim removes the repeated latent head.  A straight
            # sample join preserves that continuation better than a crossfade.
            output = engine.assemble(
                completed, output_format, crossfade_seconds=0.0)
            final_quality = engine.validate_final_output(
                completed, output, request.input,
                fidelity_mode=fidelity_mode,
                min_similarity=request.min_similarity,
                strict_fidelity=request.strict_fidelity,
                target_duration_seconds=None)
            headers = {
                "X-Voice-Chunks": str(len(completed)),
                "X-Voice-Id": profile.id,
                "X-H3-Job-Id": completed[0].path.parent.name,
                "X-Voice-Mode": "fl2va-motion-context",
                "X-H3-Warm": "true",
                "X-H3-Render-Seconds": f"{total_render:.3f}",
                "X-ASR-Verification": "enabled" if verify else "disabled",
                "X-ASR-Min-Similarity": f"{minimum_similarity:.4f}",
                "X-Voice-Exact-Match": str(
                    final_quality["exact_match"]).lower(),
                "X-Voice-Leading-Trim-Seconds": (
                    f"{completed[0].leading_trim_seconds:.3f}"),
            }
            if fmt == "pcm":
                headers.update({
                    "X-Audio-Format": "pcm_s16le",
                    "X-Audio-Sample-Rate": "32000",
                    "X-Audio-Channels": str(request.channels),
                })
                return Response(
                    content=audio_to_pcm(output, request.channels),
                    media_type=("audio/x-pcm;codec=pcm_s16le;rate=32000;"
                                f"channels={request.channels}"),
                    headers=headers,
                )
            media = {"flac": "audio/flac", "wav": "audio/wav",
                     "mp3": "audio/mpeg", "opus": "audio/ogg",
                     "aac": "audio/aac"}[fmt]
            return FileResponse(
                output, media_type=media, filename=output.name, headers=headers)

        results = engine.generate(**kwargs)
        if request.stream:
            # Generate and validate the first chunk before committing HTTP
            # headers. Later accepted chunks continue rendering while it plays.
            first = next(results)
            results = chain((first,), results)
            # Raw signed 16-bit little-endian PCM is genuinely streamable:
            # each accepted chunk is yielded while the next chunk begins rendering.
            return StreamingResponse(
                engine.stream_pcm(results),
                media_type=f"audio/x-pcm;codec=pcm_s16le;rate=32000;channels={request.channels}",
                headers={
                    "X-Audio-Format": "pcm_s16le",
                    "X-Audio-Sample-Rate": "32000",
                    "X-Audio-Channels": str(request.channels),
                    "X-Streaming-Behavior": "chunk-ready",
                    "X-Voice-Id": profile.id,
                    "X-H3-Job-Id": first.path.parent.name,
                    "X-ASR-Verification": "enabled" if verify else "disabled",
                },
            )
        completed = list(results)
        if request.response_format == "pcm":
            output = engine.assemble(
                completed, "flac",
                target_duration_seconds=request.target_duration_seconds)
            final_quality = engine.validate_final_output(
                completed, output, request.input,
                fidelity_mode=fidelity_mode,
                min_similarity=request.min_similarity,
                strict_fidelity=request.strict_fidelity,
                target_duration_seconds=request.target_duration_seconds)
            total_render = sum(item.render_seconds for item in completed)
            return Response(
                content=audio_to_pcm(output, request.channels),
                media_type=f"audio/x-pcm;codec=pcm_s16le;rate=32000;channels={request.channels}",
                headers={"X-Audio-Format": "pcm_s16le",
                         "X-Audio-Sample-Rate": "32000",
                         "X-Audio-Channels": str(request.channels),
                         "X-Voice-Chunks": str(len(completed)),
                         "X-Voice-Id": profile.id,
                         "X-H3-Job-Id": completed[0].path.parent.name,
                         "X-Voice-Exact-Match": str(
                             final_quality["exact_match"]).lower(),
                         "X-H3-Warm": "true",
                         "X-H3-Render-Seconds": f"{total_render:.3f}",
                         "X-ASR-Verification": "enabled" if verify else "disabled"},
            )
        fmt = request.response_format
        output = engine.assemble(
            completed, fmt,
            target_duration_seconds=request.target_duration_seconds)
        final_quality = engine.validate_final_output(
            completed, output, request.input,
            fidelity_mode=fidelity_mode,
            min_similarity=request.min_similarity,
            strict_fidelity=request.strict_fidelity,
            target_duration_seconds=request.target_duration_seconds)
        total_render = sum(item.render_seconds for item in completed)
        media = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg",
                 "opus": "audio/ogg", "aac": "audio/aac"}[fmt]
        return FileResponse(output, media_type=media, filename=output.name,
                            headers={"X-Voice-Chunks": str(len(completed)),
                                     "X-Voice-Id": profile.id,
                                     "X-H3-Job-Id": completed[0].path.parent.name,
                                     "X-Voice-Exact-Match": str(
                                         final_quality["exact_match"]).lower(),
                                     "X-H3-Warm": "true",
                                     "X-H3-Render-Seconds": f"{total_render:.3f}",
                                     "X-ASR-Verification": "enabled" if verify else "disabled"})
    except HTTPException:
        raise
    except SpeechFidelityError as exc:
        raise HTTPException(status_code=502, detail={
            "error": "speech_quality_rejected",
            "message": str(exc),
            "expected": exc.text,
            "transcript": exc.transcript,
            "similarity": exc.similarity,
            "exact_match": exc.exact_match,
            "attempts": exc.attempts,
            "rejection_reasons": exc.rejection_reasons,
        }) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="voice not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/speech")
def speech(request: SpeechRequest):
    """OpenAI-compatible speech route with optional MiniMax extensions."""
    return _speech_response(request)


def _eleven_to_openai(voice_id: str, body: ElevenSpeechRequest,
                      stream: bool) -> SpeechRequest:
    settings = body.voice_settings
    format_prefix = body.output_format.casefold().split("_", 1)[0]
    format_aliases = {"ogg": "opus", "ulaw": "wav", "alaw": "wav"}
    response_format = format_aliases.get(format_prefix, format_prefix)
    if response_format not in {"flac", "wav", "mp3", "opus", "aac", "pcm"}:
        raise HTTPException(status_code=422, detail=f"unsupported output_format: {body.output_format}")
    # ElevenLabs' stability/similarity controls do not have direct H3 sampler
    # equivalents. Speaker boost maps to ASR retries; style maps to expressivity.
    return SpeechRequest(
        model=body.model_id, input=body.text, voice=voice_id,
        speech_mode="reference",
        response_format=response_format, stream=stream,
        instructions=body.instructions, emotion=body.emotion,
        emotion_intensity=settings.style, language=body.language_code,
        speed=settings.speed, verify=body.verify,
        max_retries=2 if settings.use_speaker_boost else 0,
        min_similarity=0.75 + 0.20 * settings.similarity_boost,
        strict_fidelity=True,
        resolution=32, spatial_preset="close", channels=2,
        steps=20 + round(15 * settings.stability),
    )


@app.post("/v1/text-to-speech/{voice_id}")
def eleven_speech(voice_id: str, body: ElevenSpeechRequest):
    return _speech_response(_eleven_to_openai(voice_id, body, False))


@app.post("/v1/text-to-speech/{voice_id}/stream")
def eleven_stream(voice_id: str, body: ElevenSpeechRequest):
    return _speech_response(_eleven_to_openai(voice_id, body, True))


@app.get("/v1/queue")
def queue() -> dict:
    return engine.comfy.queue.public()
