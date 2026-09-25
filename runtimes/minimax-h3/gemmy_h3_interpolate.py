"""H3 DLSS frame interpolation post via pinned ComfyUI-NVIDIA-DLSS-Frame-Interpolation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

H3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(H3_ROOT))

from h3_comfy_common import (  # noqa: E402
    find_latest,
    force_utf8_stdio,
    materialize_workflow,
    resolve_comfy_root,
    run_comfy_graph,
    stage_file,
    write_json,
)

PACKAGE = "comfy-workflow-h3-dlss-interpolate"
PACK_DIR = "ComfyUI-NVIDIA-DLSS-Frame-Interpolation"

FPS_CHOICES = (
    "23.976",
    "25",
    "29.97",
    "30",
    "48",
    "50",
    "59.94",
    "60",
    "90",
    "120",
)
ENGINE_MAP = {
    "auto": "Auto",
    "native": "Native DLSSG",
    "cascade": "Cascade",
}
QUALITY_MAP = {
    "auto": "Auto (Default)",
    "max": "Max",
    "best": "Best",
    "good": "Good",
}
CODEC_MAP = {
    "h264": "H.264",
    "h264-nvenc": "H.264 (NVIDIA NVENC)",
    "h265": "H.265",
    "h265-nvenc": "H.265 (NVIDIA NVENC)",
    "av1": "AV1",
    "av1-nvenc": "AV1 (NVIDIA NVENC)",
    "prores": "ProRes Proxy",
}

REQUIRED_RUNTIME = (
    Path("bin") / "runtime" / "dlssg" / "nvngx_dlssg.dll",
    Path("bin") / "runtime" / "dlssg" / "dlssg-worker.exe",
    Path("bin") / "runtime" / "host" / "nvngx.dll",
    Path("bin") / "runtime" / "host" / "dxgi.dll",
    Path("bin") / "runtime" / "host" / "renodx-dlss5.addon64",
    Path("bin") / "runtime" / "host" / "nvngx_dlssnr.dll",
    Path("bin") / "runtime" / "dlss" / "nvngx_dlss.dll",
)


def _dll_report(pack: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    stubs: list[str] = []
    for rel in REQUIRED_RUNTIME:
        path = pack / rel
        ok = path.is_file()
        size = path.stat().st_size if ok else 0
        # Git LFS pointer files are tiny text; real DLLs are megabytes.
        stub = ok and size < 1024
        files.append(
            {
                "path": str(path),
                "ok": ok and not stub,
                "bytes": size,
                "stub": stub,
            }
        )
        if not ok:
            missing.append(str(rel).replace("\\", "/"))
        elif stub:
            stubs.append(str(rel).replace("\\", "/"))
    return {
        "ok": not missing and not stubs,
        "missing": missing,
        "stubs": stubs,
        "files": files,
    }


def _require_runtime(pack: Path) -> None:
    report = _dll_report(pack)
    if report["ok"]:
        return
    parts = []
    if report["missing"]:
        parts.append("missing: " + ", ".join(report["missing"]))
    if report["stubs"]:
        parts.append("Git LFS stubs: " + ", ".join(report["stubs"]))
    raise SystemExit(
        "NVIDIA DLSS Frame Generation runtimes are incomplete under "
        f"{pack}. {'; '.join(parts)}. "
        "Clone/pull with Git LFS (`git lfs pull`) into "
        "runtimes/minimax-h3/ComfyUI/custom_nodes/"
        "ComfyUI-NVIDIA-DLSS-Frame-Interpolation. "
        "NVIDIA SDK DLLs are not committed."
    )


def _forward_ffmpeg(env: dict[str, str]) -> None:
    ffmpeg = env.get("GEMMY_FFMPEG") or env.get("DLSS_FFMPEG_PATH")
    if ffmpeg:
        env["DLSS_FFMPEG_PATH"] = ffmpeg
        probe = Path(ffmpeg).with_name("ffprobe.exe")
        if not probe.is_file():
            probe = Path(ffmpeg).with_name("ffprobe")
        if probe.is_file():
            env["DLSS_FFPROBE_PATH"] = str(probe)


def main() -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    h3_root = Path(req.get("h3_root") or H3_ROOT)
    comfy_root = resolve_comfy_root(req, h3_root)
    python = Path(req.get("python") or sys.executable)
    pack = Path(req.get("node_pack") or comfy_root / "custom_nodes" / PACK_DIR)
    if not pack.is_dir() or not (pack / "__init__.py").is_file():
        raise SystemExit(
            f"ComfyUI-NVIDIA-DLSS-Frame-Interpolation is not pinned at {pack}. "
            "Expected runtimes/minimax-h3/ComfyUI/custom_nodes/"
            "ComfyUI-NVIDIA-DLSS-Frame-Interpolation."
        )
    if req.get("check_only"):
        report = _dll_report(pack)
        print(json.dumps({"ok": report["ok"], "pack": str(pack), **report}, indent=2))
        return 0 if report["ok"] else 3

    _require_runtime(pack)

    inp = Path(req["input"])
    out = Path(req["output"])
    if not inp.is_file():
        raise SystemExit(f"input missing: {inp}")
    work = Path(req.get("work_dir") or out.parent / ".h3_interpolate")
    work.mkdir(parents=True, exist_ok=True)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    fps = str(req.get("fps") or "48")
    if fps not in FPS_CHOICES:
        raise SystemExit(f"fps must be one of {', '.join(FPS_CHOICES)}; got {fps!r}")
    engine_key = str(req.get("engine") or "auto").strip().lower()
    if engine_key not in ENGINE_MAP:
        raise SystemExit(f"engine must be auto|native|cascade; got {engine_key!r}")
    quality_key = str(req.get("quality") or "max").strip().lower()
    if quality_key not in QUALITY_MAP:
        raise SystemExit(f"quality must be auto|max|best|good; got {quality_key!r}")
    codec_key = str(req.get("codec") or "h264").strip().lower()
    if codec_key not in CODEC_MAP:
        raise SystemExit(
            "codec must be h264|h264-nvenc|h265|h265-nvenc|av1|av1-nvenc|prores; "
            f"got {codec_key!r}"
        )

    stem = inp.stem
    basename = stage_file(inp, comfy_root / "input", f"gemmy_dlss_{inp.name}")
    prefix = f"gemmy/dlss/{stem}"
    graph = materialize_workflow(
        PACKAGE,
        {
            "video": basename,
            "output_fps": fps,
            "dlss_engine": ENGINE_MAP[engine_key],
            "encoding_quality": QUALITY_MAP[quality_key],
            "video_codec": CODEC_MAP[codec_key],
            "container": "MP4",
            "filename_prefix": prefix,
        },
    )
    present = {n.get("class_type") for n in graph.values() if isinstance(n, dict)}
    if "NvidiaDLSSFrameInterpolation" not in present:
        raise SystemExit("interpolate graph missing NvidiaDLSSFrameInterpolation")
    if "SaveVideo" not in present:
        raise SystemExit("interpolate graph missing SaveVideo")

    env = os.environ.copy()
    _forward_ffmpeg(env)
    os.environ.update(
        {
            k: v
            for k, v in env.items()
            if k in ("DLSS_FFMPEG_PATH", "DLSS_FFPROBE_PATH", "GEMMY_FFMPEG")
        }
    )

    t0 = time.time()
    outputs_dir, note, elapsed = run_comfy_graph(
        comfy_root=comfy_root,
        python=python,
        graph=graph,
        graph_path=work / "comfy_workflow.json",
        timing_path=work / "comfy_timing.json",
        attn="sage",
        require_accel=False,
    )
    hit = find_latest(outputs_dir, prefix, t0, ".mp4")
    if hit is None:
        hit = find_latest(outputs_dir, prefix, t0, ".mkv")
    if hit is None:
        raise SystemExit(
            "DLSS interpolation produced no video under Comfy output "
            f"(prefix={prefix}, elapsed={elapsed:.1f}s)"
        )
    shutil.copy2(hit, out)
    report_src = find_latest(outputs_dir, prefix, t0, ".json")
    report_dest = out.with_suffix(".dlss.json")
    if report_src is not None:
        shutil.copy2(report_src, report_dest)
    write_json(
        work / "result.json",
        {
            "ok": True,
            "input": str(inp),
            "output": str(out),
            "source": str(hit),
            "fps": fps,
            "engine": ENGINE_MAP[engine_key],
            "elapsed": elapsed,
            "note": note,
        },
    )
    print(f"[h3 interpolate] wrote {out} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
