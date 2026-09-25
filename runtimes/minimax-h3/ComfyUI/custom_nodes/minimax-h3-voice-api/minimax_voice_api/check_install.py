"""Validate the local ComfyUI integration without generating media."""
from __future__ import annotations

import json
import shutil
import urllib.request

import os
from pathlib import Path

from .config import (
    CLIP, COMFY_ROOT, COMFY_URL, UNET_REF2VA, UNET_REF2VA_EROS, UNET_T2VA,
    VAE_AUDIO, VAE_VIDEO,
)

REQUIRED_NODES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo", "BasicScheduler", "KSamplerSelect",
    "RandomNoise", "BasicGuider", "SamplerCustomAdvanced", "VAEDecodeAudio",
    "LoadImage", "LoadAudio", "SaveAudio", "VAEDecode",
    "MiniMaxH3MotionContext", "MiniMaxH3MotionContextTrim",
}


def _weight_roots() -> list[Path]:
    roots = [
        Path(os.environ.get("GEMMY_H3_CHECKPOINTS", r"C:\Projects\minimax-h3\checkpoints")),
        Path(os.environ.get("GEMMY_H3_EROS_CHECKPOINTS", r"F:\Models\minimax-h3-eros")),
        Path(os.environ.get("GEMMY_H3_REF2VA_STOCK", r"G:\Models\minimax-h3-backup")),
        COMFY_ROOT / "models",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _find_weight(kind: str, filename: str) -> Path:
    for root in _weight_roots():
        candidate = root / kind / filename
        if candidate.is_file():
            return candidate
    return COMFY_ROOT / "models" / kind / filename


def collect() -> dict:
    checks: dict[str, object] = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "comfy_root": str(COMFY_ROOT),
        "comfy_main": (COMFY_ROOT / "main.py").is_file(),
        "comfy_url": COMFY_URL,
    }
    expected_models = {
        "t2va": _find_weight("diffusion_models", UNET_T2VA),
        "ref2va_stock": _find_weight("diffusion_models", UNET_REF2VA),
        "ref2va_eros": _find_weight("diffusion_models", UNET_REF2VA_EROS),
        "clip": _find_weight("text_encoders", CLIP),
        "video_vae": _find_weight("vae", VAE_VIDEO),
        "audio_vae": _find_weight("vae", VAE_AUDIO),
    }
    checks["models"] = {name: {"path": str(path), "found": path.is_file()}
                        for name, path in expected_models.items()}
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3).read(1)
        checks["comfy_online"] = True
    except Exception as exc:
        checks["comfy_online"] = False
        checks["comfy_error"] = str(exc)
    try:
        info = json.load(urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=10))
        missing = sorted(REQUIRED_NODES - set(info))
        checks["required_nodes"] = {"found": not missing, "missing": missing}
    except Exception as exc:
        checks["required_nodes"] = {"found": False, "missing": sorted(REQUIRED_NODES),
                                    "error": str(exc)}
    return checks


def main() -> None:
    checks = collect()
    print(json.dumps(checks, indent=2))
    ok = bool(checks["ffmpeg"] and checks["ffprobe"] and checks["comfy_main"]
              and checks.get("comfy_online"))
    ok = ok and all(item["found"] for item in checks["models"].values())
    ok = ok and bool(checks["required_nodes"]["found"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
