"""Post-decode video upscale helper (Base 768p-class → 1080p / 2K-ish).

Official H3-Regenerate-2K is a **closed second generative pass**, not open SR.

**RTX backend** runs the vendored community node
``ComfyUI-NVIDIA-RTX-VSR-Pro`` (``RTXVideoSuperResolution``) through the
internalized H3 Comfy engine (``ComfyUI/run_h3_workflow.py``). That is the
same NVIDIA ``nvidia-vfx`` / ``nvvfx.VideoSuperRes`` SDK effect, with Pro's
own 16K hybrid planning, channel-integrity abort, and DLPack clone — **not**
a Gemmy-owned reimplementation of the node.

Graph (API format):
  LoadVideo → GetVideoComponents → RTXVideoSuperResolution → CreateVideo → SaveVideo

Owner-box preference order (``--backend auto``):
  1. **rtx** — Pro node via H3 Comfy (requires ``nvidia-vfx`` + Pro custom_node)
  2. **video2x** — Real-ESRGAN via Video2X if on PATH
  3. **ffmpeg** — lanczos (soft fallback)

Install RTX path (optional NVIDIA index wheel; not pinned in uv.lock):
  uv pip install -U --no-build-isolation nvidia-vfx --index-url https://pypi.nvidia.com
Models usually resolve via the package or ``NVVFX_SDK_PATH`` /
``C:\\Program Files\\NVIDIA Corporation\\NVIDIA Video Effects\\models``.

Upscale **final deliverables only** — never frames fed back into FL2VA continue
encode (would change cond geometry).

Examples:
  uv run python scripts/h3_upscale.py outputs/seg/video.mp4 --scale 2
  uv run python scripts/h3_upscale.py outputs/seg/video.mp4 --backend rtx --scale 2 \\
    --audio outputs/seg/audio.wav --out outputs/seg/output_2k.mp4
  uv run python scripts/h3_upscale.py outputs/seg/video.mp4 --height 1080 --backend ffmpeg
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Pro node schema caps scale-by-multiplier at 4.0; larger factors use target dims.
PRO_SCALE_MAX = 4.0
ALIGNMENT = 8


def _h3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comfy_root() -> Path:
    env = (os.environ.get("GEMMY_H3_COMFY") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "run_h3_workflow.py").is_file():
            return p
    return (_h3_root() / "ComfyUI").resolve()


def _pro_node_dir(comfy: Optional[Path] = None) -> Path:
    root = comfy or _comfy_root()
    return root / "custom_nodes" / "ComfyUI-NVIDIA-RTX-VSR-Pro"


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found") from e


def _ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def _video2x() -> Optional[str]:
    return shutil.which("video2x") or shutil.which("video2x.exe")


def _rtx_available() -> Tuple[bool, str]:
    """Return (ok, detail). Pro node + nvvfx + CUDA torch + Comfy runner."""
    comfy = _comfy_root()
    runner = comfy / "run_h3_workflow.py"
    if not runner.is_file():
        return False, f"Comfy runner missing: {runner}"
    pro = _pro_node_dir(comfy)
    if not (pro / "__init__.py").is_file():
        return False, f"Pro node missing: {pro}"
    try:
        import nvvfx  # noqa: F401
    except Exception as e:
        return False, f"nvvfx/nvidia-vfx not importable ({e})"
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda not available"
    except Exception as e:
        return False, f"torch missing ({e})"
    return True, f"Pro node + nvvfx ok ({pro.name})"


def run(cmd: Sequence[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode})")


def probe_wh_fps(src: Path) -> Tuple[int, int, float]:
    """Best-effort width, height, fps via ffprobe."""
    ffprobe = _ffprobe()
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of",
            "csv=p=0:s=x",
        ]
        try:
            out = subprocess.check_output(cmd + [str(src)], text=True).strip()
            parts = out.replace(",", "x").split("x")
            if len(parts) >= 3:
                w, h = int(parts[0]), int(parts[1])
                rate = parts[2]
                if "/" in rate:
                    a, b = rate.split("/", 1)
                    fps = float(a) / max(float(b), 1e-9)
                else:
                    fps = float(rate) if rate else 24.0
                if w > 0 and h > 0 and fps > 0:
                    return w, h, fps
        except Exception:
            pass
    raise RuntimeError(
        f"cannot probe video {src}: need ffprobe on PATH "
        "(install ffmpeg) for RTX/path geometry"
    )


def resolve_out_size(
    in_w: int,
    in_h: int,
    *,
    scale: Optional[int],
    width: Optional[int],
    height: Optional[int],
    multiple: int = ALIGNMENT,
) -> Tuple[int, int]:
    """Target pixel size (align-down to multiple). Used for planning/logging."""
    if width is not None and height is not None:
        ow, oh = int(width), int(height)
    elif height is not None:
        oh = int(height)
        ow = int(round(in_w * (oh / in_h)))
    elif width is not None:
        ow = int(width)
        oh = int(round(in_h * (ow / in_w)))
    else:
        s = int(scale or 2)
        ow, oh = in_w * s, in_h * s
    if multiple > 1:
        ow = max(multiple, (ow // multiple) * multiple)
        oh = max(multiple, (oh // multiple) * multiple)
    return ow, oh


def _stage_video(src: Path, comfy_input: Path, dest_name: str) -> str:
    """Copy/link video into Comfy input/; return basename for LoadVideo."""
    comfy_input.mkdir(parents=True, exist_ok=True)
    dest = comfy_input / dest_name
    src = src.resolve()
    if dest.exists():
        try:
            if dest.samefile(src):
                return dest_name
        except Exception:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest_name


def _find_latest_mp4(output_dir: Path, prefix: str, t_after: float) -> Optional[Path]:
    """Find SaveVideo output matching filename_prefix written after t_after."""
    prefix = prefix.replace("\\", "/").strip("/")
    candidates: List[Path] = []
    if not output_dir.is_dir():
        return None
    for p in output_dir.rglob("*.mp4"):
        try:
            if p.stat().st_mtime + 0.5 < t_after:
                continue
        except OSError:
            continue
        rel = p.relative_to(output_dir).as_posix()
        if prefix in rel or p.stem.startswith(prefix.split("/")[-1]):
            candidates.append(p)
    if not candidates:
        all_new: List[Path] = []
        for p in output_dir.rglob("*.mp4"):
            try:
                if p.stat().st_mtime + 0.5 >= t_after:
                    all_new.append(p)
            except OSError:
                pass
        candidates = all_new
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _quality_name(name: str) -> str:
    key = name.strip().upper().replace("-", "_")
    aliases = {"U": "ULTRA", "H": "HIGH", "M": "MEDIUM", "L": "LOW"}
    key = aliases.get(key, key)
    if key not in ("LOW", "MEDIUM", "HIGH", "ULTRA"):
        raise ValueError(f"unknown RTX quality {name!r}; try ULTRA/HIGH/MEDIUM/LOW")
    return key


def upscale_rtx(
    src: Path,
    dst: Path,
    *,
    scale: int,
    width: Optional[int],
    height: Optional[int],
    quality: str,
) -> None:
    """File pipeline: stage video → H3 Comfy + Pro RTXVideoSuperResolution → copy MP4."""
    ok, detail = _rtx_available()
    if not ok:
        raise FileNotFoundError(
            f"RTX VSR Pro unavailable: {detail}. "
            "Install: uv pip install -U --no-build-isolation nvidia-vfx "
            "--index-url https://pypi.nvidia.com "
            "and ensure ComfyUI-NVIDIA-RTX-VSR-Pro is under ComfyUI/custom_nodes/"
        )

    comfy = _comfy_root()
    runner = comfy / "run_h3_workflow.py"
    h3 = _h3_root()
    in_w, in_h, fps = probe_wh_fps(src)

    # Resolve planning size for logs + for scale>4 conversion to target dims.
    planned_w, planned_h = resolve_out_size(
        in_w, in_h, scale=scale, width=width, height=height, multiple=ALIGNMENT
    )

    wf_scale: Optional[float] = None
    wf_w: Optional[int] = width
    wf_h: Optional[int] = height
    keep_aspect = False
    if width is None and height is None:
        if float(scale) <= PRO_SCALE_MAX:
            wf_scale = float(scale)
        else:
            # Pro combo max is 4.0 — convert large multipliers to target dims.
            wf_w, wf_h = planned_w, planned_h
            keep_aspect = False
    elif width is not None and height is not None:
        # Exact product canvas (e.g. --canvas 1080p → 1920×1088).
        wf_w, wf_h = planned_w, planned_h
        keep_aspect = False
    else:
        # Single-side target: let Pro fit aspect inside the box.
        wf_w = planned_w if width is not None else None
        wf_h = planned_h if height is not None else None
        # Still pass both so Pro has a full box after resolve_out_size.
        wf_w, wf_h = planned_w, planned_h
        keep_aspect = True

    job = uuid.uuid4().hex[:10]
    video_name = f"gemmy_vsr_{job}{src.suffix.lower() or '.mp4'}"
    prefix = f"gemmy_vsr/{job}"
    mode = (
        f"scale×{wf_scale}"
        if wf_scale is not None
        else f"target {wf_w}x{wf_h} keep_aspect={keep_aspect}"
    )
    print(
        f"rtx-vsr-pro: {in_w}x{in_h} @ {fps:.3f}fps → ~{planned_w}x{planned_h} "
        f"({mode}, quality={_quality_name(quality)}) via Comfy Pro node",
        flush=True,
    )

    comfy_input = comfy / "input"
    staged = _stage_video(src, comfy_input, video_name)
    sys.path.insert(0, str(_h3_root().parent.parent / "comfy-workflows"))
    from materialize import materialize_package  # type: ignore

    params: dict = {
        "video": staged,
        "quality": _quality_name(quality),
        "output_prefix": prefix,
    }
    if wf_scale is None:
        w = int(wf_w or 1920)
        h = int(wf_h or 1080)
        w = max(ALIGNMENT, (w // ALIGNMENT) * ALIGNMENT)
        h = max(ALIGNMENT, (h // ALIGNMENT) * ALIGNMENT)
        params["resize_type"] = "target dimensions"
        params["target_width"] = w
        params["target_height"] = h
        params["keep_aspect"] = bool(keep_aspect)
    else:
        params["resize_type"] = "scale by multiplier"
        params["scale"] = float(wf_scale)
    graph = materialize_package("comfy-workflow-h3-rtx-vsr", params)

    work = h3 / "outputs" / "_vsr_pro_work" / job
    work.mkdir(parents=True, exist_ok=True)
    graph_path = work / "workflow_api.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    timing_path = work / "timing.json"

    py_candidates = [
        h3 / ".venv" / "Scripts" / "python.exe",
        h3 / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    comfy_python = Path(sys.executable)
    for cand in py_candidates:
        if cand.is_file():
            comfy_python = cand
            break

    cmd = [
        str(comfy_python),
        "-u",
        str(runner),
        str(graph_path),
        "--out-note",
        str(timing_path),
        "--attn",
        "sage",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    # Upscale does not need DiT weights, but the shared runner resolves checkpoints.
    # Rust launcher already sets GEMMY_H3_CHECKPOINTS; keep whatever is present.

    print(f"rtx-vsr-pro: comfy execute ({comfy_python.name})", flush=True)
    t0 = time.perf_counter()
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(comfy), env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Comfy Pro VSR runner failed with exit {proc.returncode} "
            f"(workflow={graph_path})"
        )

    outputs_dir = comfy / "output"
    if timing_path.is_file():
        try:
            note = json.loads(timing_path.read_text(encoding="utf-8"))
            if note.get("outputs_dir"):
                outputs_dir = Path(note["outputs_dir"])
            if note.get("success") is False:
                raise RuntimeError(f"Comfy Pro VSR reported success=false: {note}")
        except RuntimeError:
            raise
        except Exception:
            pass

    mp4 = _find_latest_mp4(outputs_dir, prefix, t0 - 1.0)
    if mp4 is None:
        raise RuntimeError(
            f"no Pro VSR mp4 under {outputs_dir} for prefix={prefix}"
        )

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mp4, dst)
    print(
        f"rtx-vsr-pro: wrote {dst} from {mp4.name} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    # Best-effort cleanup of staged input (keep work/ for debug).
    try:
        staged_path = comfy_input / staged
        if staged_path.is_file():
            staged_path.unlink()
    except OSError:
        pass


def upscale_video2x(
    src: Path,
    dst: Path,
    *,
    scale: int,
    width: Optional[int],
    height: Optional[int],
    model: str,
) -> None:
    exe = _video2x()
    if not exe:
        raise FileNotFoundError("video2x not on PATH")
    cmd: List[str] = [
        exe,
        "-i",
        str(src),
        "-o",
        str(dst),
        "-p",
        "realesrgan",
        "-s",
        str(scale),
        "--realesrgan-model",
        model,
    ]
    if width is not None:
        cmd.extend(["-w", str(width)])
    if height is not None:
        cmd.extend(["-h", str(height)])
    run(cmd)


def upscale_ffmpeg(
    src: Path,
    dst: Path,
    *,
    scale: Optional[float],
    width: Optional[int],
    height: Optional[int],
) -> None:
    if width is not None and height is not None:
        vf = f"scale={width}:{height}:flags=lanczos"
    elif height is not None:
        vf = f"scale=-2:{height}:flags=lanczos"
    elif width is not None:
        vf = f"scale={width}:-2:flags=lanczos"
    else:
        s = float(scale or 2.0)
        vf = f"scale=iw*{s}:ih*{s}:flags=lanczos"
    cmd = [
        _ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "fast",
        "-an",
        str(dst),
    ]
    run(cmd)


def mux(video: Path, audio: Path, out: Path) -> None:
    cmd = [
        _ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out),
    ]
    run(cmd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Post-decode upscale for H3 deliverables "
            "(RTX VSR Pro Comfy node / Video2X / ffmpeg)"
        )
    )
    ap.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="input mp4 (usually silent video.mp4 or output.mp4)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: <input>_up_<tag>.mp4)",
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=2,
        help="scale factor (Pro node multiplier max 4; larger uses target dims)",
    )
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument(
        "--height", type=int, default=None, help="e.g. 1080 / 1440 for short-edge targets"
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "rtx", "video2x", "ffmpeg"],
        default="auto",
        help="auto: rtx (Comfy Pro node) → video2x → ffmpeg lanczos",
    )
    ap.add_argument(
        "--rtx-quality",
        default="ULTRA",
        help="Pro node quality (default ULTRA; also HIGH/MEDIUM/LOW)",
    )
    ap.add_argument(
        "--model",
        default="realesrgan-plus",
        help="Video2X RealESRGAN model name (default realesrgan-plus)",
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="optional wav/aac to mux after upscale (reuses decode audio.wav)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="print backend availability and exit",
    )
    args = ap.parse_args(argv)

    rtx_ok, rtx_detail = _rtx_available()
    if args.check:
        print(f"rtx: {'yes' if rtx_ok else 'no'}  ({rtx_detail})")
        print(f"video2x: {'yes' if _video2x() else 'no'}")
        ff_ok = False
        try:
            print(f"ffmpeg: {_ffmpeg()}")
            ff_ok = True
        except Exception as e:
            print(f"ffmpeg: no ({e})")
        return 0 if rtx_ok or _video2x() or ff_ok else 1

    if args.input is None:
        print("FAIL: input video required (or pass --check)", file=sys.stderr)
        return 2
    src = Path(args.input)
    if not src.is_file():
        print(f"FAIL: missing {src}", file=sys.stderr)
        return 1

    if args.out is None:
        tag = f"{args.height}p" if args.height else f"x{args.scale}"
        args.out = src.with_name(f"{src.stem}_up_{tag}{src.suffix}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        if rtx_ok:
            backend = "rtx"
        elif _video2x():
            backend = "video2x"
        else:
            backend = "ffmpeg"
        print(f"auto backend → {backend}", flush=True)

    tmp = out if args.audio is None else out.with_name(out.stem + "_vonly" + out.suffix)
    try:
        if backend == "rtx":
            try:
                upscale_rtx(
                    src,
                    tmp,
                    scale=args.scale,
                    width=args.width,
                    height=args.height,
                    quality=args.rtx_quality,
                )
            except FileNotFoundError as e:
                print(f"{e}; falling back", flush=True)
                backend = "video2x" if _video2x() else "ffmpeg"
            except RuntimeError as e:
                # Pro integrity abort / Comfy execute failure: fail closed for
                # channel-collapse style errors; other runtime errors may fall back.
                msg = str(e)
                if "channel" in msg.lower() and "integrity" in msg.lower():
                    raise
                if "success=false" in msg or "Comfy Pro VSR" in msg:
                    # Real Pro/Comfy failure — do not silently ship lanczos as rtx.
                    raise
                print(f"rtx failed ({e}); falling back", flush=True)
                backend = "video2x" if _video2x() else "ffmpeg"
            except Exception as e:
                print(f"rtx failed ({e}); falling back", flush=True)
                backend = "video2x" if _video2x() else "ffmpeg"

        if backend == "video2x":
            try:
                upscale_video2x(
                    src,
                    tmp,
                    scale=args.scale,
                    width=args.width,
                    height=args.height,
                    model=args.model,
                )
            except FileNotFoundError:
                print("video2x not found; falling back to ffmpeg lanczos", flush=True)
                backend = "ffmpeg"

        if backend == "ffmpeg":
            upscale_ffmpeg(
                src,
                tmp,
                scale=float(args.scale),
                width=args.width,
                height=args.height,
            )
    except Exception as e:
        print(f"FAIL: upscale: {e}", file=sys.stderr)
        return 1

    if args.audio is not None:
        try:
            mux(tmp, Path(args.audio), out)
            if tmp != out and tmp.is_file():
                tmp.unlink(missing_ok=True)
        except Exception as e:
            print(f"FAIL: mux: {e}", file=sys.stderr)
            return 1
    elif tmp != out:
        tmp.replace(out)

    print(f"wrote {out}  (backend={backend})")
    print(
        "note: file RTX VSR via ComfyUI-NVIDIA-RTX-VSR-Pro "
        "(not App playback toggle; not a Gemmy-owned nvvfx port). "
        "Official H3-Regenerate-2K remains closed generative API — this is post-SR."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
