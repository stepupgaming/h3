from __future__ import annotations

import bisect
import json
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .config import (
    CLIP,
    COMFY_INPUT,
    COMFY_OUTPUT,
    COMFY_URL,
    JOB_ROOT,
    PARAKEET_MODEL,
    UNET_REF2VA,
    UNET_T2VA,
    VAE_AUDIO,
    VAE_VIDEO,
    require_comfy_layout,
    unet_for_dit,
)
from .core import (
    VoiceProfile,
    VoiceRegistry,
    audio_signal_metrics,
    audio_to_pcm,
    chunk_text,
    concat_audio,
    decoded_audio_duration,
    fit_audio_duration,
    normalize_words,
    preprocess_generated,
    remove_onset_orphan,
    speech_activity_margins,
    speech_join_pause_seconds,
    trim_aligned_audio,
    trim_audio_for_join,
    trim_leading_audio,
    word_similarity,
)

FPS = 24


def _direction_text(value: str) -> str:
    """Keep user direction inert instead of allowing dialogue-tag injection."""
    value = re.sub(r"</?d(?:\s[^>]*)?>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<(?:scenetrans|cutoff)\s*/?>", "", value,
                   flags=re.IGNORECASE)
    return " ".join(value.split()).strip()


def _validate_spoken_text(value: str) -> str:
    if re.search(r"</?d(?:\s[^>]*)?>|<(?:scenetrans|cutoff)\s*/?>",
                 value, flags=re.IGNORECASE):
        raise ValueError("spoken text must be plain text; dialogue tags are added by the API")
    return value.strip()


def frames_for(seconds: float) -> int:
    requested = max(5, round(seconds * FPS))
    lower = max(5, requested - ((requested - 5) % 17))
    upper = lower + 17
    return lower if requested - lower <= upper - requested else upper


def speech_seconds(text: str, speed: float = 1.0) -> float:
    words = max(1, len(text.split()))
    punctuation = sum(text.count(mark) for mark in ".,;:!?—")
    return max(3.0, min(13.0,
                        words / (2.65 * max(0.65, speed))
                        + 0.10 * punctuation + 0.18))


def transcript_exact_match(expected: str, observed: str) -> bool:
    """Require the complete ASR transcript, not an embedded word run."""
    return word_similarity(expected, observed) >= 1.0 - 1e-9


def performance_text(text: str) -> str:
    """Add one actable beat to a clause without changing its spoken words."""
    words = text.split()
    if len(words) <= 5 or "..." in text or "…" in text:
        return text
    middle = len(words) // 2
    # Existing midpoint punctuation already gives H3 a delivery beat.
    nearby = words[max(0, middle - 1):min(len(words), middle + 1)]
    if any(word.endswith((",", ";", ":", "—")) for word in nearby):
        return text
    return " ".join(words[:middle] + ["..."] + words[middle:])


def audio_duration(path: Path) -> float:
    value = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True).strip()
    return float(value)


def speech_prompt(profile: VoiceProfile, text: str, direction: str = "",
                  language: str = "English", space: str = "close",
                  include_picture: bool = True,
                  audio_ref_keep: float = 1.0) -> str:
    text = _validate_spoken_text(text)
    direction = (_direction_text(direction)
                 or "natural conversational delivery with clear English diction")
    identity = (_direction_text(profile.description)
                or "a natural expressive adult speaker")
    spaces = {
        "studio": "a quiet professional voice booth with dry close-mic acoustics",
        "close": "a real furnished room, physically close to the listener, with subtle natural early reflections and intimate microphone perspective",
        "across_table": "a quiet furnished room seated across a small table from the listener, with believable conversational distance and restrained room reflections",
        "living_room": "a comfortable furnished living room near the listener, with soft realistic room tone and short warm reflections",
        "bedside": "a quiet bedroom beside the listener, very close but naturally voiced, with soft fabric absorption and tiny spatial reflections",
        "stage": "a small treated performance stage facing the listener, with controlled professional room reflections and clear direct sound",
    }
    space_description = spaces.get(space, spaces["close"])
    picture_definition = (
        "<Picture 1> is a blank dark frame and carries no content.\n"
        if include_picture else "")
    picture_retention = (
        "\n<Picture 1>: weak_reference - contributes nothing to the target audio."
        if include_picture else "")
    if audio_ref_keep >= 0.9995:
        reuse_summary = (
            "Treat <Audio 1> as a strict voice-identity reference. Preserve the "
            "same adult speaker: timbre, breath, natural pace, microphone "
            "distance, accent, and human realism. Do not repeat its words.")
        retention = (
            "<Audio 1>: voice_timbre_and_recording_style - preserve this exact "
            "speaker and their natural human micro-pauses for a new line.")
    else:
        reuse_summary = (
            "Treat <Audio 1> as immutable speaker identity, not a performance "
            "template. Preserve timbre, age, and accent while the new acting "
            "direction controls cadence and emotion. Do not repeat its words.")
        retention = (
            "<Audio 1>: voice_identity_only - preserve this exact speaker while "
            "the target acting direction overrides reference cadence.")
    return f"""subject_definitions:
{picture_definition}<Audio 1> is the canonical voice identity of Speaker 1 (S1).

summary:
[audio reuse] {reuse_summary}

retention_analysis:
{retention}{picture_retention}

detailed_description:
[Shot 1] A still, dark scene. Speaker 1 (S1) is the same adult speaker from <Audio 1>. Voice identity and acting direction only, never spoken aloud: {identity.rstrip('.')}; {direction.rstrip('.')}. The speaker is in {space_description}. The delivery is emotionally present, connected, and naturally conversational. The only audible words in the entire target are inside the following single dialogue block. Speaker 1 (S1) says exactly and only: <d>[{language}] {text}</d> After the final word, Speaker 1 closes their mouth. No voice description, acting note, instruction, label, metadata, or other prompt text is spoken before or after this dialogue.

overall_soundscape:
Subtle believable non-verbal ambience and microphone perspective from the specified space. No music, effects, other voices, or speech outside the marked dialogue.

non_diegetic_music:
N/A"""


def continuous_speech_prompt(profile: VoiceProfile, text: str,
                             direction: str = "", language: str = "English",
                             space: str = "close", index: int = 0,
                             total: int = 1) -> str:
    """Build the compact scene prompt used by FL2VA motion-context speech.

    Unlike Ref2VA, this path asks H3 to perform one continuing scene.  The
    persona and acting notes stay outside the sole dialogue block so they
    cannot leak into speech.
    """
    text = _validate_spoken_text(text)
    direction = (_direction_text(direction)
                 or "natural conversational delivery with clear diction")
    identity = (_direction_text(profile.description)
                or "a natural expressive adult speaker")
    spaces = {
        "studio": "a quiet professional voice booth with dry close-mic acoustics",
        "close": "a real furnished room close to the listener with subtle natural early reflections",
        "across_table": "a quiet furnished room across a small table from the listener",
        "living_room": "a comfortable furnished living room near the listener",
        "bedside": "a quiet bedroom beside the listener with soft fabric absorption",
        "stage": "a small treated performance stage facing the listener",
    }
    location = spaces.get(space, spaces["close"])
    continuity = (
        "This opening establishes the one speaker and recording perspective."
        if index == 0 else
        "Continue the same uninterrupted conversation with exactly the same "
        "speaker, vocal identity, microphone distance, room, and emotional state."
    )
    chapter = f"part {index + 1} of {total}" if total > 1 else "one complete take"
    return f"""integrated_multimodal_description: [Shot 1] A static dark scene in {location}. This is {chapter} of one continuous live-action voice performance. {continuity} Speaker 1 (S1) is {identity.rstrip('.')}. Acting direction only, never spoken aloud: {direction.rstrip('.')}. The performance is emotionally present, spontaneous, and unmistakably human, with connected phrasing, natural breaths, small timing variations, and no announcer or TTS cadence. Only the words inside the following dialogue block are audible. Speaker 1 (S1) says exactly and only: <d>[{language}] {text}</d> After the final word, the speaker closes their mouth. No label, voice description, acting note, instruction, or other prompt prose is spoken.

overall_soundscape: Subtle continuous room tone and believable close-microphone presence from the specified space. No music, effects, other voices, or speech outside the marked dialogue.

non_diegetic_music: N/A"""


def anchor_prompt(profile: VoiceProfile, sample_text: str) -> str:
    identity = (_direction_text(profile.description)
                or "a natural expressive adult speaker")
    sample_text = _validate_spoken_text(sample_text)
    return f"""integrated_multimodal_description: [Shot 1] A static featureless dark frame. A single speaker records a clean voice reference in a quiet professional studio. Voice identity direction only, never spoken aloud: {identity.rstrip('.')}. The delivery is natural, relaxed, expressive, and unmistakably human. The only audible words in the target are inside this single dialogue block. The speaker (S1) says exactly and only: <d>[English] {sample_text}</d> No voice description, direction, label, or prompt text is spoken before or after the sentence.

overall_soundscape: Clean close-microphone speech with barely audible studio room tone. No music, sound effects, background activity, echoes, or long pauses.

non_diegetic_music: N/A"""


def _save_audio_ids(graph: dict) -> list[str]:
    ids = [
        nid
        for nid, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == "SaveAudio"
    ]
    ids.sort(key=lambda x: int(x) if str(x).isdigit() else str(x))
    if not ids:
        raise ValueError("compiled speech graph has no SaveAudio nodes")
    return ids


def _materialize(package: str, params: dict) -> dict:
    import sys
    root = Path(__file__).resolve().parents[3] / "comfy-workflows"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from materialize import materialize_package  # type: ignore
    return materialize_package(package, params)


def _video_vae(override: str | None = None) -> str:
    raw = (override or VAE_VIDEO or "").strip()
    name = Path(raw).name if raw else "minimax_h3_video_vae_int8_convrot.safetensors"
    if not name.endswith(".safetensors"):
        name = f"{name}.safetensors"
    return name


def t2va_graph(prompt: str, seconds: float, steps: int, seed: int, prefix: str,
               resolution: int = 32, video_vae: str | None = None) -> dict:
    return _materialize("comfy-workflow-h3-speech-t2va", {
        "prompt": prompt,
        "length": frames_for(seconds),
        "steps": steps,
        "seed": seed,
        "output_prefix": prefix,
        "width": resolution,
        "height": resolution,
        "unet": UNET_T2VA,
        "clip": CLIP,
        "video_vae": _video_vae(video_vae),
        "audio_vae": VAE_AUDIO,
    })


def ref2va_graph(prompt: str, reference: str, image: str, seconds: float,
                 steps: int, seed: int, prefix: str, resolution: int = 32,
                 audio_ref_keep: float = 1.0, unet: str | None = None,
                 video_vae: str | None = None) -> dict:
    if abs(audio_ref_keep - 1.0) > 1e-9:
        return _materialize("comfy-workflow-h3-speech-ref2va-keep", {
            "prompt": prompt,
            "length": frames_for(seconds),
            "steps": steps,
            "seed": seed,
            "output_prefix": prefix,
            "width": resolution,
            "height": resolution,
            "ref_image": image,
            "reference_audio": reference,
            "audio_keep": audio_ref_keep,
            "unet": unet or UNET_REF2VA,
            "clip": CLIP,
            "video_vae": _video_vae(video_vae),
            "audio_vae": VAE_AUDIO,
        })
    return _materialize("comfy-workflow-h3-speech-ref2va", {
        "prompt": prompt,
        "length": frames_for(seconds),
        "steps": steps,
        "seed": seed,
        "output_prefix": prefix,
        "width": resolution,
        "height": resolution,
        "image": image,
        "reference_audio": reference,
        "unet": unet or UNET_REF2VA,
        "clip": CLIP,
        "video_vae": _video_vae(video_vae),
        "audio_vae": VAE_AUDIO,
    })


def motion_context_speech_graph(prompts: list[str], seconds: list[float],
                                steps: int, seeds: list[int], prefix: str,
                                resolution: int = 32,
                                context_frames: int = 22,
                                audio_context_frames: int = 240,
                                video_vae: str | None = None,
                                ) -> tuple[dict, list[str]]:
    """Build one FL2VA graph whose later clips inherit the previous AV latent."""
    if not prompts or len(prompts) != len(seconds) or len(prompts) != len(seeds):
        raise ValueError("prompts, seconds, and seeds must have the same non-zero length")
    overlap_seconds = context_frames / FPS
    n = len(prompts)
    if n == 1:
        graph = t2va_graph(prompts[0], seconds[0], steps, seeds[0], f"{prefix}/chunk_0000", resolution, video_vae=video_vae)
        return graph, _save_audio_ids(graph)
    if n == 2:
        graph = _materialize("comfy-workflow-h3-speech-motion-context", {
            "prompt_0": prompts[0],
            "length_0": frames_for(seconds[0]),
            "seed_0": seeds[0],
            "prompt_1": prompts[1],
            "length_1": frames_for(seconds[1] + overlap_seconds),
            "seed_1": seeds[1],
            "steps": steps,
            "unet": UNET_T2VA,
            "clip": CLIP,
            "video_vae": _video_vae(video_vae),
            "audio_vae": VAE_AUDIO,
            "output_prefix_0": f"{prefix}/chunk_0000",
            "output_prefix_1": f"{prefix}/chunk_0001",
        })
        return graph, _save_audio_ids(graph)
    if n == 3:
        graph = _materialize("comfy-workflow-h3-speech-motion-3", {
            "prompt": prompts[0],
            "length": frames_for(seconds[0]),
            "seed": seeds[0],
            "prompt_1": prompts[1],
            "length_1": frames_for(seconds[1] + overlap_seconds),
            "seed_1": seeds[1],
            "prompt_2": prompts[2],
            "length_2": frames_for(seconds[2] + overlap_seconds),
            "seed_2": seeds[2],
            "steps": steps,
            "unet": UNET_T2VA,
            "clip": CLIP,
            "video_vae": _video_vae(video_vae),
            "audio_vae": VAE_AUDIO,
            "output_prefix": f"{prefix}/chunk_0000",
            "output_prefix_1": f"{prefix}/chunk_0001",
            "output_prefix_2": f"{prefix}/chunk_0002",
        })
        return graph, _save_audio_ids(graph)
    raise ValueError(
        f"continuous speech graphs are packaged for 1–3 chunks; got {n}. "
        "The caller should batch longer jobs."
    )



class QueueState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.waiting = 0
        self.active: dict | None = None
        self.completed = 0
        self.last_finished_at: float | None = None
        self.last_render_seconds: float | None = None
        self.last_success = False

    def acquire(self, detail: dict) -> None:
        with self._state_lock:
            self.waiting += 1
        self.lock.acquire()
        with self._state_lock:
            self.waiting -= 1
            self.active = detail

    def release(self, *, success: bool = False,
                render_seconds: float | None = None) -> None:
        with self._state_lock:
            self.active = None
            self.last_success = success
            if success:
                self.completed += 1
                self.last_finished_at = time.time()
                self.last_render_seconds = render_seconds
        self.lock.release()

    def public(self) -> dict:
        with self._state_lock:
            return {
                "waiting": self.waiting, "active": self.active,
                "serialized_for_vram": True,
                "completed_since_api_start": self.completed,
                "last_finished_at": self.last_finished_at,
                "last_render_seconds": self.last_render_seconds,
            }


class ComfyClient:
    def __init__(self, base_url: str = COMFY_URL, queue: QueueState | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.queue = queue or QueueState()

    def health(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=2).read(1)
            return True
        except Exception:
            return False

    def system_stats(self) -> dict | None:
        try:
            return json.load(urllib.request.urlopen(
                f"{self.base_url}/system_stats", timeout=3))
        except Exception:
            return None

    def warmth(self, *, asr_loaded: bool = False) -> dict:
        """Report the persistent model cache used by consecutive API calls."""
        stats = self.system_stats()
        devices = (stats or {}).get("devices", [])
        device = devices[0] if devices else {}
        total = int(device.get("vram_total", 0) or 0)
        free = int(device.get("vram_free", total) or 0)
        reserved_gb = max(0.0, (total - free) / (1024 ** 3))
        queue = self.queue.public()
        # The H3 transformer itself occupies roughly 20 GB on this machine.
        # A conservative 8 GB threshold distinguishes a resident H3 stack from
        # an otherwise idle ComfyUI process without claiming exact model IDs.
        resident = reserved_gb >= 8.0
        if queue["active"]:
            state = "warming" if queue["completed_since_api_start"] == 0 else "hot"
        elif resident:
            state = "hot"
        else:
            state = "cold"
        return {
            "enabled": True,
            "state": state,
            "h3_gpu_resident": resident,
            "gpu_reserved_gb": round(reserved_gb, 2),
            "asr_cpu_resident": asr_loaded,
            "idle_timeout_seconds": None,
            "policy": "retain model and loader caches until VRAM pressure or manual unload",
            "completed_since_api_start": queue["completed_since_api_start"],
            "last_finished_at": queue["last_finished_at"],
            "last_render_seconds": queue["last_render_seconds"],
        }

    def run_audio_nodes(self, graph: dict, detail: dict,
                        output_node_ids: list[str] | None = None,
                        timeout: float = 3600) -> tuple[list[Path], float]:
        """Run a graph and return audio outputs in caller-specified node order."""
        require_comfy_layout()
        self.queue.acquire(detail)
        started = time.time()
        success = False
        try:
            request = urllib.request.Request(
                f"{self.base_url}/api/prompt",
                json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode(),
                {"Content-Type": "application/json"},
            )
            prompt_id = json.load(urllib.request.urlopen(request, timeout=30))["prompt_id"]
            while time.time() - started < timeout:
                time.sleep(1.0)
                try:
                    history = json.load(urllib.request.urlopen(
                        f"{self.base_url}/api/history/{prompt_id}", timeout=15))
                except (urllib.error.URLError, TimeoutError):
                    continue
                if prompt_id not in history:
                    continue
                entry = history[prompt_id]
                if entry.get("status", {}).get("status_str") != "success":
                    messages = entry.get("status", {}).get("messages", [])
                    errors = [str(x[1].get("exception_message", "")) for x in messages
                              if x[0] == "execution_error"]
                    raise RuntimeError(errors[0] if errors else "ComfyUI generation failed")
                outputs = entry.get("outputs", {})
                ordered = output_node_ids or list(outputs)
                paths: list[Path] = []
                for node_id in ordered:
                    output = outputs.get(str(node_id), {})
                    for audio in output.get("audio", []) or []:
                        path = COMFY_OUTPUT / audio.get("subfolder", "") / audio["filename"]
                        if not path.is_file():
                            raise RuntimeError(
                                f"ComfyUI reported {path}, but it is not locally visible. "
                                "Check H3_COMFY_ROOT."
                            )
                        paths.append(path)
                if not paths:
                    raise RuntimeError("ComfyUI completed without audio")
                if output_node_ids and len(paths) != len(output_node_ids):
                    raise RuntimeError(
                        f"ComfyUI returned {len(paths)} of {len(output_node_ids)} "
                        "expected audio chunks")
                success = True
                return paths, time.time() - started
            raise TimeoutError("timed out waiting for ComfyUI")
        finally:
            self.queue.release(success=success,
                               render_seconds=round(time.time() - started, 3))

    def run(self, graph: dict, detail: dict,
            timeout: float = 3600) -> tuple[Path, float]:
        paths, elapsed = self.run_audio_nodes(graph, detail, timeout=timeout)
        return paths[0], elapsed


class ParakeetVerifier:
    """Lazy CPU-only ASR. Loading it never claims CUDA VRAM."""
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                import onnx_asr
                self._model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v2", path=PARAKEET_MODEL,
                    quantization="int8", providers=["CPUExecutionProvider"])
                self._model.recognize(np.zeros(16000, np.float32), sample_rate=16000)
            return self._model

    def transcribe(self, path: Path) -> str:
        return self.transcribe_timestamped(path).text

    def transcribe_timestamped(self, path: Path):
        proc = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le",
            "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
        ], capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        with self._inference_lock:
            return self._get_model().with_timestamps().recognize(
                audio, sample_rate=16000)


class SpeechFidelityError(RuntimeError):
    """Raised when every MiniMax attempt fails the requested speech check."""

    def __init__(self, text: str, transcript: str, similarity: float,
                 attempts: int, *, rejection_reasons: list[str] | None = None,
                 exact_match: bool = False) -> None:
        reasons = rejection_reasons or []
        super().__init__(
            f"speech fidelity check failed after {attempts} attempt(s): "
            f"similarity={similarity:.3f}; exact={exact_match}; "
            f"reasons={', '.join(reasons) or 'quality threshold'}; "
            f"expected={text!r}; heard={transcript!r}"
        )
        self.text = text
        self.transcript = transcript
        self.similarity = similarity
        self.attempts = attempts
        self.exact_match = exact_match
        self.rejection_reasons = reasons


FidelityMode = Literal["exact", "similarity", "unverified"]


def exact_speech_bounds(result, expected: str) -> tuple[float, float | None] | None:
    """Find an exact expected word run and return safe audio cut boundaries."""
    expected_words = normalize_words(expected)
    heard_matches = list(re.finditer(r"[a-z0-9]+(?:'[a-z0-9]+)?",
                                     result.text.lower().replace("’", "'")))
    heard_words = [match.group(0) for match in heard_matches]
    if not expected_words or len(heard_words) < len(expected_words):
        return None
    # Locate the requested run while allowing only exact split/closed compounds
    # ("streetlights" == "street lights"). This is stricter than fuzzy text
    # matching and keeps the timestamp cut anchored to known requested words.
    word_span = None
    for start_index in range(len(heard_words)):
        expected_index, heard_index = 0, start_index
        while expected_index < len(expected_words) and heard_index < len(heard_words):
            if expected_words[expected_index] == heard_words[heard_index]:
                expected_index += 1
                heard_index += 1
            elif (heard_index + 1 < len(heard_words)
                  and expected_words[expected_index]
                  == heard_words[heard_index] + heard_words[heard_index + 1]):
                expected_index += 1
                heard_index += 2
            elif (expected_index + 1 < len(expected_words)
                  and expected_words[expected_index] + expected_words[expected_index + 1]
                  == heard_words[heard_index]):
                expected_index += 2
                heard_index += 1
            else:
                break
        if expected_index == len(expected_words):
            word_span = (start_index, heard_index)
            break
    if word_span is None or not result.tokens or not result.timestamps:
        return None
    word_index, after_index = word_span

    joined = "".join(result.tokens)
    base = joined.find(result.text)
    if base < 0:
        base = len(joined) - len(joined.lstrip())
    token_ends = []
    length = 0
    for token in result.tokens:
        length += len(token)
        token_ends.append(length)

    start_char = base + heard_matches[word_index].start()
    start_token = min(bisect.bisect_right(token_ends, start_char),
                      len(result.timestamps) - 1)
    start = max(0.0, float(result.timestamps[start_token]) - 0.12)

    end = None
    if after_index < len(heard_matches):
        last_char = base + heard_matches[after_index - 1].end() - 1
        last_token = min(bisect.bisect_right(token_ends, last_char),
                         len(result.timestamps) - 1)
        after_char = base + heard_matches[after_index].start()
        after_token = min(bisect.bisect_right(token_ends, after_char),
                          len(result.timestamps) - 1)
        last_time = float(result.timestamps[last_token])
        next_time = float(result.timestamps[after_token])
        end = max(start + 0.25, (last_time + next_time) / 2.0)
    return start, end


def requested_speech_start(result, expected: str) -> float | None:
    """Locate the requested opening phrase even if later words are imperfect."""
    opening = normalize_words(expected)[:4]
    if not opening:
        return None
    bounds = exact_speech_bounds(result, " ".join(opening))
    return bounds[0] if bounds else None


@dataclass
class ChunkResult:
    index: int
    text: str
    path: Path
    transcript: str
    similarity: float
    attempts: int
    render_seconds: float
    channels: int = 1
    words_per_minute: float | None = None
    tail_margin_seconds: float | None = None
    tempo_factor: float = 1.0
    leading_trim_seconds: float = 0.0
    exact_match: bool = False
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    attempt_history: list[dict] = field(default_factory=list)
    head_margin_seconds: float | None = None
    raw_transcript: str = ""
    raw_similarity: float = 0.0
    raw_exact_match: bool = False


class VoiceEngine:
    SAMPLE_TEXT = "Hello. I'm glad you're here. Let's take our time and make every word feel natural."

    def __init__(self, registry: VoiceRegistry | None = None,
                 comfy: ComfyClient | None = None,
                 verifier: ParakeetVerifier | None = None) -> None:
        self.registry = registry or VoiceRegistry()
        self.comfy = comfy or ComfyClient()
        self.verifier = verifier or ParakeetVerifier()
        self.dark_image = COMFY_INPUT / "dark.png"

    def _ensure_dark_image(self) -> None:
        require_comfy_layout()
        if not self.dark_image.is_file():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                            "-i", "color=c=0x08080a:s=64x64", "-frames:v", "1",
                            str(self.dark_image)], check=True)

    def _align_verified(self, source: Path, expected: str, stem: str,
                        expected_seconds: float) -> tuple[Path, str, float, bool]:
        """Normalize an anchor without hiding invented words via cropping."""
        processed = source.parent / f"{stem}.flac"
        preprocess_generated(source, processed, expected_seconds, channels=2)
        timestamped = self.verifier.transcribe_timestamped(processed)
        raw_exact = transcript_exact_match(expected, timestamped.text)
        if raw_exact:
            bounds = exact_speech_bounds(timestamped, expected)
            if bounds and (bounds[0] > 0.08 or bounds[1] is not None):
                aligned = processed.with_name(processed.stem + "_aligned.flac")
                trim_aligned_audio(processed, aligned, *bounds, channels=2)
                processed = aligned
        transcript = self.verifier.transcribe(processed).strip()
        exact = raw_exact and transcript_exact_match(expected, transcript)
        return processed, transcript, word_similarity(expected, transcript), exact

    def ensure_voice(self, profile: VoiceProfile, steps: int = 30) -> VoiceProfile:
        if profile.ready:
            self.registry.comfy_reference(profile)
            return profile
        seconds = speech_seconds(self.SAMPLE_TEXT)
        best: tuple[Path, str, float, bool] | None = None
        for attempt in range(3):
            prefix = f"voice_api/raw_unverified/anchors/{profile.id}_try_{attempt}"
            raw, _ = self.comfy.run(
                t2va_graph(anchor_prompt(profile, self.SAMPLE_TEXT), seconds, steps,
                            random.randrange(1, 2**63), prefix, resolution=32),
                {"kind": "voice_anchor", "voice": profile.id,
                 "attempt": attempt + 1})
            candidate = self._align_verified(
                raw, self.SAMPLE_TEXT,
                f"{profile.id}_anchor_try_{attempt}", seconds)
            if best is None or (candidate[3], candidate[2]) > (best[3], best[2]):
                best = candidate
            if candidate[3] and candidate[2] >= 0.92:
                break
        assert best is not None
        if not best[3] or best[2] < 0.85:
            raise SpeechFidelityError(
                self.SAMPLE_TEXT, best[1], best[2], 3,
                rejection_reasons=["voice anchor contains words outside its script"],
                exact_match=best[3])
        return self.registry.attach_reference(profile, best[0], best[1])

    def render_chunk(self, profile: VoiceProfile, text: str, job_id: str, index: int,
                     *, steps: int, speed: float, direction: str, verify: bool,
                     max_retries: int, min_similarity: float,
                     strict_fidelity: bool,
                     candidate_pool_size: int = 1,
                     fidelity_mode: FidelityMode = "exact",
                     max_words_per_minute: float = 225.0,
                     generation_padding_seconds: float = 0.10,
                     audio_ref_keep: float = 1.0,
                     language: str = "English", resolution: int = 32,
                     space: str = "close", seed: int | None = None,
                     channels: int = 2, unet: str | None = None,
                     video_vae: str | None = None) -> ChunkResult:
        # Surplus Ref2VA timeline is often filled with invented speech. Estimate
        # the spoken span tightly and leave only explicit, tunable headroom.
        seconds = max(3.5, min(
            13.5, speech_seconds(text, speed) + generation_padding_seconds))
        if speed < 0.9:
            pace_direction = "unhurried natural pacing"
        elif speed > 1.1:
            pace_direction = "brisk but still natural pacing"
        else:
            pace_direction = "natural conversational pacing"
        paced_direction = f"{direction.rstrip('.')}; {pace_direction}"
        self._ensure_dark_image()
        reference = self.registry.comfy_reference(profile)
        best = None
        total_render = 0.0
        history: list[dict] = []
        accepted_count = 0
        for attempt in range(max_retries + 1):
            attempt_seed = (seed + index * 1009 + attempt if seed is not None
                            else random.randrange(1, 2**63))
            prefix = (f"voice_api/raw_unverified/{job_id}/"
                      f"chunk_{index:04d}_try_{attempt}")
            prompt = speech_prompt(
                profile, text, paced_direction, language, space,
                audio_ref_keep=audio_ref_keep)
            graph = ref2va_graph(
                prompt, reference, "voice_api/dark.png", seconds, steps,
                attempt_seed, prefix, resolution, audio_ref_keep,
                unet=unet, video_vae=video_vae)
            raw, elapsed = self.comfy.run(graph, {
                "kind": "speech", "job_id": job_id, "chunk": index,
                "attempt": attempt + 1, "voice": profile.id,
            })
            total_render += elapsed
            processed = JOB_ROOT / job_id / f"chunk_{index:04d}_try_{attempt}.flac"
            preprocess_generated(raw, processed, seconds, channels=channels)
            deblipped = processed.with_name(processed.stem + "_deblipped.flac")
            onset_cleanup = remove_onset_orphan(
                processed, deblipped, channels=channels)
            if onset_cleanup:
                processed = deblipped
            transcript = ""
            raw_transcript = ""
            words_per_minute = None
            tail_margin = None
            head_margin = None
            tempo_factor = 1.0
            exact_match = not verify
            if verify:
                timestamps = self.verifier.transcribe_timestamped(processed)
                raw_transcript = timestamps.text.strip()
                raw_exact = transcript_exact_match(text, raw_transcript)
                transcript = raw_transcript
                # Do not rehabilitate a take that contains the script surrounded
                # by invented words. Clean takes may still lose silent margin.
                if raw_exact:
                    bounds = exact_speech_bounds(timestamps, text)
                    if bounds and (bounds[0] > 0.08 or bounds[1] is not None):
                        aligned = processed.with_name(processed.stem + "_aligned.flac")
                        trim_aligned_audio(processed, aligned, *bounds,
                                           channels=channels)
                        processed = aligned
                        timestamps = self.verifier.transcribe_timestamped(processed)
                        transcript = timestamps.text.strip()
                exact_match = raw_exact and transcript_exact_match(text, transcript)
                final_timestamps = timestamps
                if len(final_timestamps.timestamps) >= 2:
                    head_margin = max(0.0, float(final_timestamps.timestamps[0]))
                    spoken_span = max(
                        0.1, float(final_timestamps.timestamps[-1])
                        - float(final_timestamps.timestamps[0]))
                    words_per_minute = (len(normalize_words(text)) * 60.0
                                        / spoken_span)
                    tail_margin = max(
                        0.0, audio_duration(processed)
                        - float(final_timestamps.timestamps[-1]))
            else:
                raw_exact = True
            score = word_similarity(text, transcript) if verify else 1.0
            raw_score = word_similarity(text, raw_transcript) if verify else 1.0
            signal = audio_signal_metrics(processed)
            reasons: list[str] = []
            if fidelity_mode == "exact" and not exact_match:
                reasons.append("raw take is not an exact full-script ASR match")
            elif fidelity_mode == "similarity" and raw_score < min_similarity:
                reasons.append(
                    f"raw word similarity {raw_score:.3f} is below {min_similarity:.3f}")
            effective_max_wpm = max_words_per_minute * speed
            if words_per_minute and words_per_minute > effective_max_wpm:
                reasons.append(
                    f"speech rate {words_per_minute:.1f} WPM exceeds "
                    f"{effective_max_wpm:.1f} WPM")
            if signal["onset_orphan_cut_seconds"] is not None:
                reasons.append("isolated onset artifact remains after cleanup")
            if signal["clipped_sample_fraction"] > 0.0001:
                reasons.append("clipped sample fraction exceeds 0.01%")
            accepted = not reasons
            record = {
                "attempt": attempt + 1, "seed": attempt_seed,
                "raw_path": str(raw), "path": str(processed),
                "raw_transcript": raw_transcript,
                "raw_similarity": round(raw_score, 4),
                "raw_exact_match": raw_exact,
                "transcript": transcript, "similarity": round(score, 4),
                "exact_match": exact_match,
                "words_per_minute": (round(words_per_minute, 1)
                                     if words_per_minute is not None else None),
                "onset_cleanup_seconds": round(onset_cleanup, 3),
                "signal_metrics": signal, "accepted": accepted,
                "rejection_reasons": reasons,
                "render_seconds": round(elapsed, 3),
            }
            history.append(record)
            acoustic_head, acoustic_tail = speech_activity_margins(processed)
            head_margin = acoustic_head if acoustic_head is not None else head_margin
            tail_margin = acoustic_tail if acoustic_tail is not None else tail_margin
            candidate = ChunkResult(index, text, processed, transcript, score,
                                    attempt + 1, total_render, channels,
                                    words_per_minute, tail_margin, tempo_factor,
                                    0.0, exact_match, accepted, reasons,
                                    list(history), head_margin, raw_transcript,
                                    raw_score, raw_exact)
            candidate_rank = (
                candidate.accepted, candidate.exact_match,
                candidate.similarity,
                -max(0.0, (candidate.words_per_minute or 0.0)
                     - effective_max_wpm),
                candidate.tail_margin_seconds or 0.0,
            )
            best_rank = ((best.accepted, best.exact_match, best.similarity,
                          -max(0.0, (best.words_per_minute or 0.0)
                               - effective_max_wpm),
                          best.tail_margin_seconds or 0.0)
                         if best is not None else None)
            if best is None or candidate_rank > best_rank:
                best = candidate
            if accepted:
                accepted_count += 1
                if accepted_count >= candidate_pool_size:
                    break
        assert best is not None
        for record in history:
            record["selected"] = record["path"] == str(best.path)
        best.attempts = len(history)
        best.render_seconds = total_render
        best.attempt_history = history
        if best.rejection_reasons and (
                fidelity_mode == "exact" or strict_fidelity):
            raise SpeechFidelityError(
                text, best.raw_transcript or best.transcript,
                best.raw_similarity, best.attempts,
                rejection_reasons=best.rejection_reasons,
                exact_match=best.exact_match)
        return best

    def generate(self, *, text: str, voice: str, direction: str = "",
                 steps: int = 30, speed: float = 1.0, target_words: int = 18,
                 max_words: int = 24, verify: bool = True, max_retries: int = 2,
                 min_similarity: float = 0.72,
                 strict_fidelity: bool = True,
                 candidate_pool_size: int = 1,
                 fidelity_mode: FidelityMode = "exact",
                 max_words_per_minute: float = 225.0,
                 generation_padding_seconds: float = 0.10,
                 audio_ref_keep: float = 1.0,
                 language: str = "English",
                 resolution: int = 32, space: str = "close",
                 seed: int | None = None, channels: int = 2,
                 dit: str = "stock",
                 video_vae: str | None = None,
                 event: Callable[[dict], None] | None = None) -> Iterator[ChunkResult]:
        profile = self.ensure_voice(self.registry.get(voice), steps)
        chunks = chunk_text(text, target_words, max_words)
        if not chunks:
            raise ValueError("input text is empty")
        job_id = uuid.uuid4().hex
        for index, chunk in enumerate(chunks):
            if event:
                event({"event": "chunk_started", "job_id": job_id,
                       "index": index, "count": len(chunks), "text": chunk})
            result = self.render_chunk(
                profile, chunk, job_id, index, steps=steps, speed=speed,
                direction=direction, verify=verify, max_retries=max_retries,
                min_similarity=min_similarity, strict_fidelity=strict_fidelity,
                candidate_pool_size=candidate_pool_size,
                fidelity_mode=fidelity_mode,
                max_words_per_minute=max_words_per_minute,
                generation_padding_seconds=generation_padding_seconds,
                audio_ref_keep=audio_ref_keep,
                language=language, resolution=resolution, space=space, seed=seed,
                channels=channels, unet=unet_for_dit(dit), video_vae=video_vae)
            if event:
                event({"event": "chunk_ready", "job_id": job_id,
                       "index": index, "count": len(chunks),
                       "transcript": result.transcript,
                       "similarity": result.similarity, "attempts": result.attempts})
            yield result

    def generate_continuous(self, *, text: str, voice: str,
                            direction: str = "", steps: int = 30,
                            speed: float = 1.0, target_words: int = 16,
                            max_words: int = 20, verify: bool = True,
                            min_similarity: float = 0.72,
                            strict_fidelity: bool = False,
                            language: str = "English",
                            resolution: int = 32, space: str = "close",
                            context_seconds: float = 10.0,
                            seed: int | None = None,
                            channels: int = 2,
                            video_vae: str | None = None) -> list[ChunkResult]:
        """Render expressive FL2VA speech as one latent-context chain.

        This mode intentionally does not use the profile's reference anchor.
        The profile description establishes a new interpretation in the first
        clip; motion context carries that actual performed identity forward.
        """
        profile = self.registry.get(voice)
        chunks = chunk_text(text, target_words, max_words)
        if not chunks:
            raise ValueError("input text is empty")
        job_id = uuid.uuid4().hex
        base_seed = seed if seed is not None else random.randrange(1, 2**63)
        prompts = [
            continuous_speech_prompt(
                profile, chunk, direction, language, space, index, len(chunks))
            for index, chunk in enumerate(chunks)
        ]
        durations = [
            max(9.0, min(14.0, speech_seconds(chunk, speed) + 1.5))
            for chunk in chunks
        ]
        seeds = [base_seed + index * 1009 for index in range(len(chunks))]
        prefix = f"voice_api/continuous/{job_id}"
        audio_ctx = max(22, min(240, round(context_seconds * FPS)))
        raw_paths: list[Path] = []
        elapsed = 0.0
        offset = 0
        batch_no = 0
        while offset < len(chunks):
            take = min(3, len(chunks) - offset)
            graph, output_nodes = motion_context_speech_graph(
                prompts[offset:offset + take],
                durations[offset:offset + take],
                steps,
                seeds[offset:offset + take],
                f"{prefix}/b{batch_no:02d}",
                resolution=resolution,
                context_frames=22,
                audio_context_frames=audio_ctx,
                video_vae=video_vae,
            )
            batch_paths, batch_elapsed = self.comfy.run_audio_nodes(
                graph,
                {"kind": "continuous_speech", "job_id": job_id,
                 "chunks": len(chunks), "batch": batch_no, "voice": profile.id,
                 "mode": "fl2va_motion_context",
                 "audio_context_seconds": context_seconds},
                output_nodes,
            )
            raw_paths.extend(batch_paths)
            elapsed += batch_elapsed
            offset += take
            batch_no += 1
        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        per_chunk_render = elapsed / len(chunks)
        results: list[ChunkResult] = []
        for index, (chunk, raw_path) in enumerate(zip(chunks, raw_paths, strict=True)):
            # FLAC-to-FLAC copy preserves the exact decoded H3 waveform.  Do not
            # loudness-normalize, silence-strip, or crossfade a latent-matched
            # seam; those operations were part of the flatter Ref2VA sound.
            path = job_dir / f"chunk_{index:04d}.flac"
            shutil.copy2(raw_path, path)
            timestamped = self.verifier.transcribe_timestamped(path) if verify else None
            transcript = timestamped.text.strip() if timestamped is not None else ""
            leading_trim = 0.0
            if index == 0 and timestamped is not None:
                start = requested_speech_start(timestamped, chunk)
                if start is not None and start > 0.08:
                    aligned = job_dir / "chunk_0000_start_aligned.flac"
                    trim_leading_audio(path, aligned, start, channels=channels)
                    path = aligned
                    leading_trim = start
            similarity = word_similarity(chunk, transcript) if verify else 1.0
            results.append(ChunkResult(
                index=index, text=chunk, path=path, transcript=transcript,
                similarity=similarity, attempts=1,
                render_seconds=per_chunk_render, channels=channels,
                leading_trim_seconds=leading_trim,
            ))
        if verify and strict_fidelity:
            worst = min(results, key=lambda item: item.similarity)
            if worst.similarity < min_similarity:
                raise SpeechFidelityError(
                    worst.text, worst.transcript, worst.similarity, 1)
        return results

    @staticmethod
    def stream_pcm(results: Iterator[ChunkResult]) -> Iterator[bytes]:
        """Stream accepted chunks with a 60 ms crossfade at each boundary.

        The tail is held back while the next MiniMax chunk renders. This lets the
        first chunk begin playing immediately without giving up smooth seams.
        """
        overlap_frames = round(0.060 * 32000)
        pending: np.ndarray | None = None
        channels: int | None = None
        for result in results:
            channels = result.channels
            overlap_samples = overlap_frames * channels
            current = np.frombuffer(
                audio_to_pcm(result.path, channels), dtype="<i2").copy()
            if pending is None:
                pending = current
                if len(pending) > overlap_samples:
                    yield pending[:-overlap_samples].astype("<i2", copy=False).tobytes()
                    pending = pending[-overlap_samples:]
                continue

            count = min(overlap_samples, len(pending), len(current))
            if count:
                fade_out = np.linspace(1.0, 0.0, count, endpoint=False)
                fade_in = 1.0 - fade_out
                mixed = np.clip(
                    pending[-count:].astype(np.float32) * fade_out
                    + current[:count].astype(np.float32) * fade_in,
                    -32768, 32767).astype("<i2")
                if len(pending) > count:
                    yield pending[:-count].astype("<i2", copy=False).tobytes()
                yield mixed.tobytes()
                current = current[count:]
            elif len(pending):
                yield pending.astype("<i2", copy=False).tobytes()

            if len(current) > overlap_samples:
                yield current[:-overlap_samples].astype("<i2", copy=False).tobytes()
                pending = current[-overlap_samples:]
            else:
                pending = current
        if pending is not None and len(pending):
            yield pending.astype("<i2", copy=False).tobytes()

    @staticmethod
    def assemble(results: list[ChunkResult], fmt: str,
                 crossfade_seconds: float = 0.06,
                 target_duration_seconds: float | None = None) -> Path:
        job_dir = results[0].path.parent
        output = job_dir / f"speech.{fmt}"
        paths: list[Path] = []
        for index, item in enumerate(results):
            start, end = 0.0, None
            if crossfade_seconds > 0:
                if index and item.head_margin_seconds is not None:
                    pause = speech_join_pause_seconds(results[index - 1].text)
                    start = max(
                        0.0, item.head_margin_seconds
                        - (pause + crossfade_seconds) / 2.0)
                if index + 1 < len(results) and item.tail_margin_seconds is not None:
                    pause = speech_join_pause_seconds(item.text)
                    retained = (pause + crossfade_seconds) / 2.0
                    end = (audio_duration(item.path)
                           - max(0.0, item.tail_margin_seconds - retained))
            if start > 0 or end is not None:
                trimmed = job_dir / f"chunk_{index:04d}_join.flac"
                trim_audio_for_join(
                    item.path, trimmed, start=start, end=end,
                    channels=results[0].channels)
                paths.append(trimmed)
            else:
                paths.append(item.path)
        untimed = (output if target_duration_seconds is None else
                   job_dir / "speech_untimed.flac")
        concat_audio(paths, untimed, fmt if target_duration_seconds is None else "flac",
                     channels=results[0].channels,
                     crossfade_seconds=crossfade_seconds)
        if target_duration_seconds is not None:
            tempo = fit_audio_duration(
                untimed, output, target_duration_seconds, results[0].channels)
            for item in results:
                item.tempo_factor = tempo
        report = job_dir / "report.json"
        report.write_text(json.dumps([{
            "index": x.index, "text": x.text, "transcript": x.transcript,
            "similarity": round(x.similarity, 4), "attempts": x.attempts,
            "render_seconds": round(x.render_seconds, 2), "path": str(x.path),
            "channels": x.channels,
            "words_per_minute": (round(x.words_per_minute, 1)
                                 if x.words_per_minute is not None else None),
            "tail_margin_seconds": (round(x.tail_margin_seconds, 3)
                                    if x.tail_margin_seconds is not None else None),
            "tempo_factor": round(x.tempo_factor, 4),
            "leading_trim_seconds": round(x.leading_trim_seconds, 3),
            "exact_match": x.exact_match,
            "accepted": x.accepted,
            "rejection_reasons": x.rejection_reasons,
            "raw_transcript": x.raw_transcript,
            "raw_similarity": round(x.raw_similarity, 4),
            "raw_exact_match": x.raw_exact_match,
            "attempt_history": x.attempt_history,
        } for x in results], indent=2), encoding="utf-8")
        return output

    def validate_final_output(
        self, results: list[ChunkResult], output: Path, expected: str, *,
        fidelity_mode: FidelityMode, min_similarity: float,
        strict_fidelity: bool, target_duration_seconds: float | None = None,
    ) -> dict:
        transcript = None
        similarity = 1.0
        exact_match = fidelity_mode == "unverified"
        if fidelity_mode != "unverified":
            transcript = self.verifier.transcribe_timestamped(output).text.strip()
            similarity = word_similarity(expected, transcript)
            exact_match = transcript_exact_match(expected, transcript)
        reasons: list[str] = []
        if fidelity_mode == "exact" and not exact_match:
            reasons.append("final encoded output is not an exact full-script ASR match")
        elif fidelity_mode == "similarity" and similarity < min_similarity:
            reasons.append(
                f"final word similarity {similarity:.3f} is below {min_similarity:.3f}")
        signal = audio_signal_metrics(output)
        if signal["onset_orphan_cut_seconds"] is not None:
            reasons.append("isolated onset artifact exists in final output")
        if signal["clipped_sample_fraction"] > 0.0001:
            reasons.append("final clipped sample fraction exceeds 0.01%")
        duration = decoded_audio_duration(output)
        if target_duration_seconds is not None:
            tolerance = 0.020 if output.suffix.casefold() == ".aac" else 0.010
            if abs(duration - target_duration_seconds) > tolerance:
                reasons.append(
                    "final decoded duration misses target by "
                    f"{duration - target_duration_seconds:+.3f}s")
        quality = {
            "path": str(output), "expected": expected,
            "transcript": transcript, "similarity": round(similarity, 4),
            "exact_match": exact_match, "duration_seconds": round(duration, 3),
            "signal_metrics": signal, "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        (output.parent / "final_quality.json").write_text(
            json.dumps(quality, indent=2), encoding="utf-8")
        if reasons and (fidelity_mode == "exact" or strict_fidelity):
            raise SpeechFidelityError(
                expected, transcript or "", similarity,
                sum(item.attempts for item in results),
                rejection_reasons=reasons, exact_match=exact_match)
        return quality
