"""H3 sprites via pinned ComfyUI-PixelForge-H3 (replaces gary149 cut/atlas)."""
from __future__ import annotations

import argparse
import json
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

PACKAGE = "comfy-workflow-h3-pixelforge"
PACK_DIR = "ComfyUI-PixelForge-H3"


def _copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest / src.name)
        return
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def _collect(outputs_dir: Path, prefix: str, t0: float, suffixes: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for suffix in suffixes:
        hit = find_latest(outputs_dir, prefix, t0, suffix)
        if hit is not None:
            found.append(hit)
    return found


def _frames_dir(outputs_dir: Path, prefix: str, t0: float) -> Path | None:
    hits = _collect(outputs_dir, prefix, t0, (".png", ".json"))
    for hit in hits:
        parent = hit.parent
        if (parent / "frames.json").is_file() or any(parent.glob("frame_*.png")):
            return parent
        if hit.suffix.lower() == ".json" and hit.name == "frames.json":
            return hit.parent
    # Aseprite export writes `<prefix>_NNNNN_frames/frame_000.png`
    if not outputs_dir.is_dir():
        return None
    candidates: list[Path] = []
    for p in outputs_dir.rglob("frames.json"):
        try:
            if p.stat().st_mtime + 0.5 < t0:
                continue
        except OSError:
            continue
        candidates.append(p.parent)
    if not candidates:
        for p in outputs_dir.rglob("frame_000.png"):
            try:
                if p.stat().st_mtime + 0.5 >= t0:
                    candidates.append(p.parent)
            except OSError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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
            f"ComfyUI-PixelForge-H3 is not pinned at {pack}. "
            "Expected runtimes/minimax-h3/ComfyUI/custom_nodes/ComfyUI-PixelForge-H3."
        )

    inp = Path(req["input"])
    out = Path(req["output"])
    if not inp.is_file():
        raise SystemExit(f"input missing: {inp}")
    work = Path(req.get("work_dir") or out / ".work")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    stem = inp.stem
    basename = stage_file(inp, comfy_root / "input", f"gemmy_sprite_{inp.name}")
    every_nth = int(req.get("every_nth") or 2)
    start_offset = int(req.get("start_offset") or 0)
    max_frames = int(req.get("max_frames") or req.get("frames") or 0)
    loop_mode = str(req.get("loop_mode") or ("auto" if req.get("loop") else "off"))
    if loop_mode not in ("auto", "pingpong", "off"):
        raise SystemExit(f"loop_mode must be auto|pingpong|off, got {loop_mode}")
    key_color = str(req.get("key_color") or "auto")
    fps = float(req.get("fps") or 12.0)
    atlas = bool(req.get("atlas"))
    prefix = f"gemmy/pixelforge/{stem}"

    graph = materialize_workflow(
        PACKAGE,
        {
            "video": basename,
            "every_nth": every_nth,
            "start_offset": start_offset,
            "max_frames": max_frames,
            "key_color": key_color,
            "loop_mode": loop_mode,
            "filename_prefix": f"{prefix}/sprite",
            "sheet_prefix": f"{prefix}/sheet",
            "gif_prefix": f"{prefix}/gif",
            "webp_prefix": f"{prefix}/preview",
            "tag_name": stem,
            "fps": fps,
        },
    )
    needed = (
        "PixelForgeVideoToFrames",
        "PixelForgeChromaKey",
        "PixelForgeAutoCrop",
        "PixelForgeLoopTrim",
        "PixelForgeFrameDedup",
        "PixelForgeSheetPack",
        "PixelForgeAsepriteExport",
        "PixelForgeSaveGIF",
    )
    present = {n.get("class_type") for n in graph.values() if isinstance(n, dict)}
    missing = [name for name in needed if name not in present]
    if missing:
        raise SystemExit(f"pixelforge graph missing classes: {missing}")

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

    keyed = out / "keyed"
    frames_src = _frames_dir(outputs_dir, f"{prefix}/sprite", t0)
    if frames_src is None:
        raise SystemExit("PixelForge produced no PNG sequence (frames.json missing)")
    _copy_tree(frames_src, keyed)

    artifacts: dict[str, str] = {"frames": str(keyed)}
    sheet = find_latest(outputs_dir, f"{prefix}/sheet", t0, ".png")
    if sheet is not None:
        dest = out / "sheet.png"
        shutil.copy2(sheet, dest)
        artifacts["sheet"] = str(dest)
    gif = find_latest(outputs_dir, f"{prefix}/gif", t0, ".gif")
    if gif is None:
        gif = find_latest(outputs_dir, f"{prefix}/gif", t0, ".webp")
    if gif is not None:
        dest = out / gif.name
        shutil.copy2(gif, dest)
        artifacts["gif" if gif.suffix.lower() == ".gif" else "preview"] = str(dest)
    webp = find_latest(outputs_dir, f"{prefix}/preview", t0, ".webp")
    if webp is not None:
        dest = out / "preview.webp"
        shutil.copy2(webp, dest)
        artifacts["preview"] = str(dest)

    atlas_path = None
    if atlas:
        atlas_path = out / "atlas.json"
        payload: dict[str, Any] = {
            "source": str(inp),
            "engine": "pixelforge",
            "package": PACKAGE,
            "loop_mode": loop_mode,
            "fps": fps,
            "frames_dir": "keyed",
            "sheet": "sheet.png" if "sheet" in artifacts else None,
        }
        manifest = keyed / "frames.json"
        if manifest.is_file():
            payload["frames"] = json.loads(manifest.read_text(encoding="utf-8"))
        atlas_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        artifacts["atlas"] = str(atlas_path)

    result = {
        "ok": True,
        "output": str(out),
        "package": PACKAGE,
        "elapsed_s": round(elapsed, 3),
        "artifacts": artifacts,
        "comfy_note": {k: note.get(k) for k in ("outputs_dir", "elapsed") if k in note},
    }
    write_json(work / "result.json", result)
    write_json(out / "report.json", result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
