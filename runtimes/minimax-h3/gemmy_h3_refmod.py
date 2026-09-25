"""Create / list / inspect MiniMax-H3 RefMods via pinned ComfyUI-MiniMaxH3Mod."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

H3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(H3_ROOT))

from h3_comfy_common import (  # noqa: E402
    bind_video_vae,
    comfy_python,
    force_utf8_stdio,
    materialize_workflow,
    resolve_comfy_root,
    run_comfy_graph,
    stage_file,
    write_json,
)

PACK_DIR = "ComfyUI-MiniMaxH3Mod"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _pack(comfy_root: Path, req: dict[str, Any]) -> Path:
    pack = Path(req.get("node_pack") or comfy_root / "custom_nodes" / PACK_DIR)
    if not (pack / "__init__.py").is_file():
        raise SystemExit(
            f"ComfyUI-MiniMaxH3Mod is not pinned at {pack}. "
            "Expected runtimes/minimax-h3/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3Mod."
        )
    return pack


def _refmods_dir(req: dict[str, Any], comfy_root: Path) -> Path:
    raw = req.get("refmods_dir") or os.environ.get("GEMMY_H3_REFMODS")
    if raw:
        path = Path(str(raw))
    else:
        ckpt = Path(str(req.get("checkpoints_root") or ""))
        path = ckpt / "refmods" if ckpt.as_posix() not in ("", ".") else comfy_root / "models" / "refmods"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _list_mods(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*.safetensors")):
        rel = p.relative_to(root).as_posix()[:-len(".safetensors")]
        rows.append(
            {
                "name": rel,
                "path": str(p),
                "bytes": p.stat().st_size,
            }
        )
    return rows


def _inspect(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    meta: dict[str, Any] = {}
    keys: list[str] = []
    shapes: dict[str, list[int]] = {}
    with safe_open(str(path), framework="numpy") as st:
        keys = list(st.keys())
        raw_meta = st.metadata() or {}
        for k, v in raw_meta.items():
            meta[k] = v
        for k in keys:
            shapes[k] = list(st.get_tensor(k).shape)
    blob = None
    for key in ("refmod_meta", "audio_refmod_meta"):
        if key in meta:
            try:
                blob = json.loads(meta[key])
            except json.JSONDecodeError:
                blob = meta[key]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensors": keys,
        "shapes": shapes,
        "metadata": meta,
        "refmod": blob,
    }


def _stage_sources(req: dict[str, Any], work: Path) -> tuple[Path | None, str | None]:
    dest = work / "refs"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    folder = req.get("folder")
    if folder:
        src = Path(str(folder))
        if not src.is_dir():
            raise SystemExit(f"folder not found: {src}")
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS and p.is_file():
                shutil.copy2(p, dest / p.name)
                n += 1
    for i, raw in enumerate(req.get("images") or []):
        p = Path(str(raw))
        if not p.is_file():
            raise SystemExit(f"image not found: {p}")
        shutil.copy2(p, dest / f"{i:02d}_{p.name}")
        n += 1
    for i, raw in enumerate(req.get("videos") or []):
        p = Path(str(raw))
        if not p.is_file():
            raise SystemExit(f"video not found: {p}")
        shutil.copy2(p, dest / f"v{i:02d}_{p.name}")
        n += 1
    audio_base = None
    if req.get("audio"):
        p = Path(str(req["audio"]))
        if not p.is_file():
            raise SystemExit(f"audio not found: {p}")
        if p.suffix.lower() not in AUDIO_EXTS:
            raise SystemExit(f"unsupported audio type: {p.suffix}")
        audio_base = stage_file(p, Path(req["comfy_input"]), f"gemmy_refmod_{p.name}")
    visual = dest if n else None
    if visual is None and audio_base is None:
        raise SystemExit("refmod create needs --folder, --image, --video, and/or --audio")
    return visual, audio_base


def _create(req: dict[str, Any]) -> int:
    h3_root = Path(req["h3_root"]).resolve()
    python = Path(req["python"]).resolve()
    comfy_root = resolve_comfy_root(req, h3_root)
    pack = _pack(comfy_root, req)
    work = Path(req["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    req["comfy_input"] = str(comfy_root / "input")
    visual, audio_base = _stage_sources(req, work)
    name = str(req.get("name") or "character").strip() or "character"
    mode = str(req.get("mode") or "encode").strip().lower()
    ui_mode = "Full Reference" if mode in ("encode", "full", "identity_encode") else "Compressed Reference"
    has_visual = visual is not None
    has_audio = audio_base is not None
    if has_visual and has_audio:
        package = "comfy-workflow-h3-refmod-create-master"
        params = {
            "folder": str(visual),
            "audio": audio_base,
            "name": name,
            "mode": ui_mode,
            "concept_type": str(req.get("concept_type") or "identity"),
            "ref_resolution": int(req.get("ref_resolution") or 1024),
            "max_tokens": int(req.get("max_tokens") or 8192),
            "max_items": int(req.get("max_items") or 32),
            "max_frames": int(req.get("max_frames") or 240),
            "max_seconds": float(req.get("max_seconds") or 15),
            "subfolder": str(req.get("subfolder") or ""),
            "description": str(req.get("description") or ""),
            "save_layout": str(req.get("save_layout") or "bundle"),
            "extraction_preset": "manual",
        }
        need = {"MiniMaxH3RefModMasterExtract"}
    elif has_audio:
        package = "comfy-workflow-h3-refmod-create-audio"
        params = {
            "audio": audio_base,
            "name": name,
            "max_seconds": float(req.get("max_seconds") or 15),
            "max_tokens": int(req.get("max_tokens") or 5120),
            "concept_type": str(req.get("concept_type") or "voice"),
            "subfolder": str(req.get("subfolder") or ""),
            "description": str(req.get("description") or ""),
        }
        need = {"MiniMaxH3RefModAudioExtract", "MiniMaxH3RefModSave"}
    else:
        package = "comfy-workflow-h3-refmod-create"
        params = {
            "folder": str(visual),
            "name": name,
            "mode": ui_mode,
            "concept_type": str(req.get("concept_type") or "identity"),
            "ref_resolution": int(req.get("ref_resolution") or 1024),
            "max_tokens": int(req.get("max_tokens") or 8192),
            "max_items": int(req.get("max_items") or 32),
            "max_frames": int(req.get("max_frames") or 240),
            "subfolder": str(req.get("subfolder") or ""),
            "description": str(req.get("description") or ""),
            "extraction_preset": "manual",
        }
        need = {"MiniMaxH3RefModFolderLoader", "MiniMaxH3RefModExtract", "MiniMaxH3RefModSave"}

    bind_video_vae(params, req)
    graph = materialize_workflow(package, params)
    present = {n.get("class_type") for n in graph.values() if isinstance(n, dict)}
    missing = need - present
    if missing:
        raise SystemExit(f"{package} missing nodes: {sorted(missing)}")

    python = comfy_python(python, comfy_root)
    import time

    t0 = time.time()
    _outputs_dir, note, elapsed = run_comfy_graph(
        comfy_root=comfy_root,
        python=python,
        graph=graph,
        graph_path=work / "comfy_workflow.json",
        timing_path=work / "comfy_timing.json",
        attn="sage",
        require_accel=False,
    )
    root = _refmods_dir(req, comfy_root)
    written = [
        p
        for p in root.rglob("*.safetensors")
        if p.stat().st_mtime + 0.5 >= t0
    ]
    written.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not written:
        # Master/Save may write into the first registered refmods folder.
        fallback = comfy_root / "models" / "refmods"
        if fallback.is_dir():
            extra = [
                p
                for p in fallback.rglob("*.safetensors")
                if p.stat().st_mtime + 0.5 >= t0
            ]
            extra.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            written.extend(extra)
    if not written:
        raise SystemExit(
            f"RefMod create produced no .safetensors under {root} "
            f"(elapsed={elapsed:.1f}s). Check video VAE path and source folder."
        )
    dest = Path(str(req["output"])) if req.get("output") else written[0]
    if dest != written[0]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(written[0], dest)
    result = {
        "ok": True,
        "package": package,
        "name": name,
        "output": str(dest),
        "written": [str(p) for p in written],
        "bytes": dest.stat().st_size,
        "elapsed": elapsed,
        "note": note,
        "pack": str(pack),
    }
    write_json(work / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[h3 refmod] wrote {dest} ({dest.stat().st_size} bytes, {elapsed:.1f}s)", flush=True)
    return 0


def main() -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    action = str(req.get("action") or "create").lower()
    h3_root = Path(req.get("h3_root") or H3_ROOT)
    comfy_root = resolve_comfy_root(req, h3_root) if action == "create" else h3_root / "ComfyUI"
    if action == "list":
        root = _refmods_dir(req, comfy_root if comfy_root.is_dir() else h3_root / "ComfyUI")
        rows = _list_mods(root)
        print(json.dumps({"ok": True, "refmods_dir": str(root), "count": len(rows), "mods": rows}, indent=2))
        return 0
    if action == "inspect":
        raw = req.get("path") or req.get("name")
        if not raw:
            raise SystemExit("inspect needs path or name")
        path = Path(str(raw))
        if not path.is_file():
            root = _refmods_dir(req, comfy_root if comfy_root.is_dir() else h3_root / "ComfyUI")
            cand = root / f"{raw}.safetensors"
            if cand.is_file():
                path = cand
            else:
                raise SystemExit(f"RefMod not found: {raw}")
        print(json.dumps({"ok": True, **_inspect(path)}, indent=2))
        return 0
    if action == "create":
        return _create(req)
    raise SystemExit(f"unknown action={action}")


if __name__ == "__main__":
    raise SystemExit(main())
