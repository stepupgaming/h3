"""Environment-based runtime configuration."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def load_dotenv() -> None:
    """Load simple KEY=VALUE settings without adding a runtime dependency."""
    candidates = (Path.cwd() / ".env", REPO_ROOT / ".env")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
        break


load_dotenv()


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


_xdg_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
DATA_ROOT = _path("H3_DATA_DIR", _xdg_data / "minimax-h3-voice-api")

_comfy_candidates = (REPO_ROOT.parent / "ComfyUI", Path.cwd() / "ComfyUI")
_default_comfy = next((path for path in _comfy_candidates if path.is_dir()),
                      _comfy_candidates[0])
COMFY_ROOT = _path("H3_COMFY_ROOT", _default_comfy)
COMFY_INPUT = COMFY_ROOT / "input" / "voice_api"
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_URL = os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

VOICE_ROOT = DATA_ROOT / "voices"
JOB_ROOT = DATA_ROOT / "jobs"

# Leave unset to let onnx-asr download/cache the CPU int8 model itself.
_parakeet = os.environ.get("H3_PARAKEET_MODEL")
PARAKEET_MODEL = Path(_parakeet).expanduser().resolve() if _parakeet else None
MAX_TRANSCRIPTION_BYTES = int(
    float(os.environ.get("H3_MAX_TRANSCRIPTION_MB", "250")) * 1024 * 1024)
MAX_TRANSCRIPTION_SECONDS = float(
    os.environ.get("H3_MAX_TRANSCRIPTION_SECONDS", "3600"))

UNET_T2VA = os.environ.get(
    "H3_MODEL_T2VA", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
UNET_REF2VA = os.environ.get(
    "H3_MODEL_REF2VA", "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
UNET_REF2VA_EROS = os.environ.get(
    "H3_MODEL_REF2VA_EROS",
    "10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors")


def unet_for_dit(dit: str) -> str:
    """Map speech --dit stock|eros to the UNETLoader filename."""
    if (dit or "stock").strip().casefold() == "eros":
        return UNET_REF2VA_EROS
    return UNET_REF2VA
CLIP = os.environ.get(
    "H3_MODEL_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
VAE_VIDEO = os.environ.get(
    "H3_MODEL_VIDEO_VAE", "minimax_h3_video_vae_int8_convrot.safetensors")
VAE_AUDIO = os.environ.get(
    "H3_MODEL_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")


def require_comfy_layout() -> None:
    """Fail before writing to a mistakenly configured filesystem path."""
    if not COMFY_ROOT.is_dir():
        raise RuntimeError(
            f"ComfyUI was not found at {COMFY_ROOT}. Set H3_COMFY_ROOT to the "
            "directory containing ComfyUI's main.py."
        )
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    COMFY_OUTPUT.mkdir(parents=True, exist_ok=True)
