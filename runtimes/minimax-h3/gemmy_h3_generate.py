"""Gemmy MiniMax-H3 one-shot generate worker.

Sequences TE → DiT sample → VAE decode as separate Python processes so a 16 GB
card never holds all three. Invoked by `gemmy video h3 generate` with a JSON
request path.

Request schema (subset):
  {
    "prompt": str,
    "output": str,                 # final .mp4 path
    "work_dir": str,
    "h3_root": str,                # internalized runtime code root
    "checkpoints_root": str|null,  # external weights (sets GEMMY_H3_CHECKPOINTS)
    "python": str,                 # interpreter that can import minimax_h3 + scripts
    "mode": "t2va"|"i2v"|"fl2va"|"ref2va",
    "quality": "fast"|"hq",      # sage vs fa2
    "width": int, "height": int,
    "frames": int,                 # already snapped 17k+5 preferred
    "steps": int, "seed": int,
    "compile_ir": bool,
    "first_image": str|null,       # RGB still for I2V / F+L
    "last_image": str|null,
    "refs": [                      # Ref2VA media (product paths; encoded here)
      {"kind": "image"|"video"|"audio", "path": str},
      {"kind": "av", "video": str, "audio": str},
      ...
    ],
    "allow_keyframe_refs": bool,   # experimental FL2VA keyframes + refs
    "weights": str,                # DiT safetensors
    "profile": bool
  }

Writes work_dir/result.json and copies/moves muxed MP4 to output.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    checkpoints_root: Path | None = None,
) -> None:
    print(f"[h3-worker] {label}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if checkpoints_root is not None:
        env["GEMMY_H3_CHECKPOINTS"] = str(checkpoints_root)
    # Prefer package resolution from h3_root (runtime code).
    pp = env.get("PYTHONPATH", "")
    root = str(cwd)
    env["PYTHONPATH"] = root + (os.pathsep + pp if pp else "")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit {proc.returncode}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _encode_media(
    *,
    python: Path,
    enc_script: Path,
    h3_root: Path,
    checkpoints: Path | None,
    src: Path,
    out: Path,
    mode: str,
    width: int,
    height: int,
    label: str,
) -> None:
    cmd = [
        str(python),
        str(enc_script),
        str(src),
        "--mode",
        mode,
        "--out",
        str(out),
        "--device",
        "cuda",
    ]
    if mode in ("image", "video"):
        cmd.extend(["--width", str(width), "--height", str(height)])
    _run(cmd, cwd=h3_root, label=label, checkpoints_root=checkpoints)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser(description="Gemmy MiniMax-H3 generate worker")
    ap.add_argument("--request", type=Path, required=True, help="JSON request path")
    args = ap.parse_args(argv)

    req = json.loads(args.request.read_text(encoding="utf-8"))
    video_vae = Path(str(req.get("video_vae") or "")).name
    if video_vae and video_vae != "minimax_h3_video_vae_fp16.safetensors":
        raise SystemExit(
            "--engine python accepts only the official fp16 video VAE "
            "(minimax_h3_video_vae_fp16.safetensors); int8 fused decode is Comfy-only"
        )
    h3_root = Path(req["h3_root"]).resolve()
    python = Path(req["python"]).resolve()
    work = Path(req["work_dir"]).resolve()
    output = Path(req["output"]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    ckpt_raw = req.get("checkpoints_root") or os.environ.get("GEMMY_H3_CHECKPOINTS")
    checkpoints = Path(ckpt_raw).resolve() if ckpt_raw else None
    if checkpoints is not None:
        os.environ["GEMMY_H3_CHECKPOINTS"] = str(checkpoints)

    if not h3_root.is_dir():
        raise SystemExit(f"h3_root missing: {h3_root}")
    if not python.is_file():
        raise SystemExit(f"python missing: {python}")

    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit("prompt is required")

    # Extra engine fields from the unified request contract are ignored here.
    # Comfy-only knobs (turbo/sol/cache) must not reach this worker; Gemmy CLI
    # rejects them when --engine python. Tolerate presence for forward-compat.
    _engine = str(req.get("engine") or "python").lower()
    if _engine not in ("python", "native", ""):
        print(
            f"[h3-worker] note: engine={_engine!r} on python worker; "
            "running streamed run_sample path",
            flush=True,
        )

    mode = str(req.get("mode") or "t2va").lower()
    quality = str(req.get("quality") or "fast").lower()
    attn = "fa2" if quality in ("hq", "fa2", "high") else "sage"
    width = int(req.get("width") or 864)
    height = int(req.get("height") or 480)
    frames = int(req.get("frames") or 124)
    steps = int(req.get("steps") or 20)
    seed = int(req.get("seed") or 42)
    compile_ir = bool(req.get("compile_ir", True))
    first_image = req.get("first_image")
    last_image = req.get("last_image")
    refs_raw = req.get("refs") or []
    allow_keyframe_refs = bool(req.get("allow_keyframe_refs", False))
    weights = Path(req["weights"]).resolve()
    profile = bool(req.get("profile", False))

    if not weights.is_file():
        raise SystemExit(f"DiT weights missing: {weights}")

    if mode == "ref2va" and not refs_raw:
        raise SystemExit("ref2va mode requires at least one entry in refs[]")

    t0 = time.perf_counter()
    stages: dict[str, float] = {}

    # --- optional IR compile ---
    prompt_path = work / "prompt.txt"
    ir_mode = "t2va"
    if mode in ("i2v", "fl2va", "i2va"):
        ir_mode = "fl2va"
    elif mode == "ref2va":
        ir_mode = "ref2va"

    text_to_encode = prompt
    if compile_ir:
        t = time.perf_counter()
        ir_out = work / "prompt_ir.txt"
        ir_script = h3_root / "scripts" / "h3_prompt_ir.py"
        cmd = [
            str(python),
            str(ir_script),
            prompt,
            "--mode",
            ir_mode,
            "--duration",
            f"{frames / 24.0:.4f}",
            "-o",
            str(ir_out),
        ]
        _run(cmd, cwd=h3_root, label="prompt_ir", checkpoints_root=checkpoints)
        text_to_encode = ir_out.read_text(encoding="utf-8")
        stages["prompt_ir_s"] = time.perf_counter() - t
    prompt_path.write_text(text_to_encode, encoding="utf-8")

    # --- text encode (own process; frees VRAM on exit) ---
    t = time.perf_counter()
    text_st = work / "text.safetensors"
    te_script = h3_root / "scripts" / "h3_text_encode.py"
    te_cmd = [
        str(python),
        str(te_script),
        text_to_encode,
        str(text_st),
        "--device",
        "cuda",
    ]
    _run(te_cmd, cwd=h3_root, label="text_encode", checkpoints_root=checkpoints)
    stages["text_encode_s"] = time.perf_counter() - t

    enc_script = h3_root / "scripts" / "h3_vae_encode.py"

    # --- optional keyframe encode for I2V / F+L ---
    first_latent = None
    last_latent = None
    if first_image:
        t = time.perf_counter()
        first_latent = work / "first_frame_latent.safetensors"
        _encode_media(
            python=python,
            enc_script=enc_script,
            h3_root=h3_root,
            checkpoints=checkpoints,
            src=Path(first_image).resolve(),
            out=first_latent,
            mode="image",
            width=width,
            height=height,
            label="vae_encode_first",
        )
        stages["vae_encode_first_s"] = time.perf_counter() - t
    if last_image:
        t = time.perf_counter()
        last_latent = work / "last_frame_latent.safetensors"
        _encode_media(
            python=python,
            enc_script=enc_script,
            h3_root=h3_root,
            checkpoints=checkpoints,
            src=Path(last_image).resolve(),
            out=last_latent,
            mode="image",
            width=width,
            height=height,
            label="vae_encode_last",
        )
        stages["vae_encode_last_s"] = time.perf_counter() - t

    # --- Ref2VA media → DiT latents (order preserved) ---
    # Each sample CLI flag needs a latent path; av becomes "video.st,audio.st".
    sample_ref_flags: list[tuple[str, str]] = []
    if refs_raw:
        t = time.perf_counter()
        refs_dir = work / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(refs_raw):
            if not isinstance(spec, dict):
                raise SystemExit(f"refs[{i}] must be an object")
            kind = str(spec.get("kind") or "").lower()
            if kind == "image":
                src = Path(spec["path"]).resolve()
                out = refs_dir / f"ref_{i:02d}_image.safetensors"
                _encode_media(
                    python=python,
                    enc_script=enc_script,
                    h3_root=h3_root,
                    checkpoints=checkpoints,
                    src=src,
                    out=out,
                    mode="image",
                    width=width,
                    height=height,
                    label=f"vae_encode_ref_image_{i}",
                )
                sample_ref_flags.append(("--ref-image", str(out)))
            elif kind == "video":
                src = Path(spec["path"]).resolve()
                out = refs_dir / f"ref_{i:02d}_video.safetensors"
                _encode_media(
                    python=python,
                    enc_script=enc_script,
                    h3_root=h3_root,
                    checkpoints=checkpoints,
                    src=src,
                    out=out,
                    mode="video",
                    width=width,
                    height=height,
                    label=f"vae_encode_ref_video_{i}",
                )
                sample_ref_flags.append(("--ref-video", str(out)))
            elif kind == "audio":
                src = Path(spec["path"]).resolve()
                out = refs_dir / f"ref_{i:02d}_audio.safetensors"
                _encode_media(
                    python=python,
                    enc_script=enc_script,
                    h3_root=h3_root,
                    checkpoints=checkpoints,
                    src=src,
                    out=out,
                    mode="audio",
                    width=width,
                    height=height,
                    label=f"vae_encode_ref_audio_{i}",
                )
                sample_ref_flags.append(("--ref-audio", str(out)))
            elif kind == "av":
                v_src = Path(spec["video"]).resolve()
                a_src = Path(spec["audio"]).resolve()
                v_out = refs_dir / f"ref_{i:02d}_av_video.safetensors"
                a_out = refs_dir / f"ref_{i:02d}_av_audio.safetensors"
                _encode_media(
                    python=python,
                    enc_script=enc_script,
                    h3_root=h3_root,
                    checkpoints=checkpoints,
                    src=v_src,
                    out=v_out,
                    mode="video",
                    width=width,
                    height=height,
                    label=f"vae_encode_ref_av_video_{i}",
                )
                _encode_media(
                    python=python,
                    enc_script=enc_script,
                    h3_root=h3_root,
                    checkpoints=checkpoints,
                    src=a_src,
                    out=a_out,
                    mode="audio",
                    width=width,
                    height=height,
                    label=f"vae_encode_ref_av_audio_{i}",
                )
                sample_ref_flags.append(("--ref-av", f"{v_out},{a_out}"))
            else:
                raise SystemExit(f"refs[{i}]: unknown kind {kind!r}")
        stages["vae_encode_refs_s"] = time.perf_counter() - t

    # --- DiT sample ---
    t = time.perf_counter()
    latent_dir = work / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    # --device / --attn are parent-level argparse flags (before the subcommand).
    sample_cmd = [
        str(python),
        "-m",
        "minimax_h3",
        "--device",
        "cuda",
        "--attn",
        attn,
        "sample",
        str(weights),
        str(text_st),
        "--steps",
        str(steps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--length",
        str(frames),
        "--seed",
        str(seed),
        "--out",
        str(latent_dir),
    ]
    if first_latent is not None:
        sample_cmd.extend(["--first-frame", str(first_latent)])
    if last_latent is not None:
        sample_cmd.extend(["--last-frame", str(last_latent)])
    for flag, value in sample_ref_flags:
        sample_cmd.extend([flag, value])
    if allow_keyframe_refs and (first_latent is not None or last_latent is not None) and sample_ref_flags:
        sample_cmd.append("--allow-keyframe-refs")
    if profile:
        sample_cmd.append("--profile")
    _run(sample_cmd, cwd=h3_root, label="dit_sample", checkpoints_root=checkpoints)
    stages["dit_sample_s"] = time.perf_counter() - t

    # --- VAE decode + mux ---
    t = time.perf_counter()
    decode_dir = work / "decode"
    dec_script = h3_root / "scripts" / "h3_vae_decode.py"
    _run(
        [
            str(python),
            str(dec_script),
            str(latent_dir),
            "--out",
            str(decode_dir),
            "--device",
            "cuda",
            "--fps",
            "24",
            "--frames",
            str(frames),
        ],
        cwd=h3_root,
        label="vae_decode",
        checkpoints_root=checkpoints,
    )
    stages["vae_decode_s"] = time.perf_counter() - t

    muxed = decode_dir / "output.mp4"
    if not muxed.is_file():
        # fall back to silent video if mux failed upstream
        silent = decode_dir / "video.mp4"
        if silent.is_file():
            muxed = silent
        else:
            raise SystemExit(f"decode produced no mp4 under {decode_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(muxed, output)

    wall = time.perf_counter() - t0
    result = {
        "ok": True,
        "output": str(output),
        "work_dir": str(work),
        "mode": mode,
        "quality": quality,
        "attn": attn,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
        "weights": str(weights),
        "checkpoints_root": str(checkpoints) if checkpoints else None,
        "n_refs": len(sample_ref_flags),
        "stages_s": stages,
        "wall_s": wall,
    }
    _write_json(work / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[h3-worker] wrote {output} in {wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
