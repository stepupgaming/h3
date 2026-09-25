"""Optional H3 FaceRefine post pass. Discovers node class ids at runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

H3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(H3_ROOT))

from h3_comfy_common import (  # noqa: E402
    find_latest,
    force_utf8_stdio,
    resolve_comfy_root,
    run_comfy_graph,
    stage_file,
    write_json,
)

FL2VA_DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def _load_mappings(pack: Path) -> dict[str, object]:
    nodes = pack / "nodes.py"
    init = pack / "__init__.py"
    # Load nodes.py, not __init__.py — the package init is `from .nodes import`
    # and fails under importlib's non-package module name.
    target = nodes if nodes.is_file() else init
    if not target.is_file():
        raise SystemExit(f"FaceRefine pack missing {target}")
    spec = importlib.util.spec_from_file_location("gemmy_h3_facerefine_pack", target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import FaceRefine pack {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mapping = getattr(mod, "NODE_CLASS_MAPPINGS", None)
    if not mapping:
        raise SystemExit(f"{target} has no NODE_CLASS_MAPPINGS")
    return dict(mapping)


def _pick(mapping: dict[str, object], *needles: str) -> str:
    names = list(mapping)
    for needle in needles:
        for name in names:
            if name == needle:
                return name
    lower = [n.lower() for n in needles]
    for name in names:
        hay = name.lower()
        if all(part in hay for part in lower[0].split()):
            return name
    raise SystemExit(f"FaceRefine node missing {needles}. Discovered: {sorted(names)}")


def _find_detector(comfy_root: Path, ckpt: Path) -> tuple[str, Path]:
    name = "face_yolov8m.pt"
    candidates = [
        ckpt / "ultralytics" / "bbox" / name,
        comfy_root / "models" / "ultralytics" / "bbox" / name,
        ckpt / "ultralytics" / name,
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise SystemExit(
            f"{name} is not installed. Place it under models/ultralytics/bbox/ "
            "(HF Bingsu/adetailer). See docs/MODEL_LOCATIONS.md."
        )
    dest_dir = comfy_root / "models" / "ultralytics" / "bbox"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if not dest.is_file():
        try:
            dest.hardlink_to(found)
        except OSError:
            shutil.copy2(found, dest)
    return name, dest


def _probe_frames(video: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "csv=p=0",
        str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        for token in out.replace("\n", ",").split(","):
            token = token.strip()
            if token.isdigit() and int(token) > 0:
                return int(token)
    except Exception:
        pass
    return 124


def _snap(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def main() -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    h3_root = Path(req.get("h3_root") or H3_ROOT)
    comfy_root = resolve_comfy_root(req, h3_root)
    python = Path(req.get("python") or sys.executable)
    pack = Path(req.get("node_pack") or comfy_root / "custom_nodes" / "ComfyUI-H3-FaceRefine")
    if not pack.is_dir():
        raise SystemExit(
            f"ComfyUI-H3-FaceRefine is not installed at {pack}. "
            "Pin the MIT pack after license/deps check; do not dual-install "
            "onnxruntime + onnxruntime-gpu."
        )
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    mapping = _load_mappings(pack)
    track = _pick(mapping, "H3FaceTrackCrop")
    stitch = _pick(mapping, "H3FaceStitch")
    inject = _pick(mapping, "H3InjectVideoLatent")
    ckpt = Path(req.get("checkpoints_root") or "")
    detector, _det_path = _find_detector(comfy_root, ckpt)

    inp = Path(req["input"])
    out = Path(req["output"])
    if not inp.is_file():
        raise SystemExit(f"input missing: {inp}")
    work = Path(req.get("work_dir") or out.parent / ".h3_face_refine")
    work.mkdir(parents=True, exist_ok=True)
    comfy_input = comfy_root / "input"
    basename = stage_file(inp, comfy_input, f"gemmy_face_{inp.name}")
    frames = _snap(int(req.get("frames") or _probe_frames(inp)))
    prompt = str(req.get("prompt") or "the same person, clear face, natural skin, sharp eyes")
    seed = int(req.get("seed") or 42)
    steps = int(req.get("steps") or 8)

    from h3_comfy_common import bind_video_vae, materialize_workflow

    params = {
        "video": basename,
        "detector": detector,
        "prompt": prompt,
        "length": frames,
        "seed": seed,
        "steps": steps,
        "output_prefix": "gemmy/h3_face",
    }
    bind_video_vae(params, req)
    graph = materialize_workflow(
        "comfy-workflow-h3-face-refine",
        params,
    )
    # The pack may register the same nodes under slightly different class ids.
    expected = {"H3FaceTrackCrop": track, "H3InjectVideoLatent": inject, "H3FaceStitch": stitch}
    for want, got in expected.items():
        if got != want:
            raise SystemExit(
                f"FaceRefine class {want} is registered as {got}; "
                "do not rewrite class_type in Python. Pin the expected pack."
            )
    t0 = time.time()
    outputs_dir, _note, _elapsed = run_comfy_graph(
        comfy_root=comfy_root,
        python=python,
        graph=graph,
        graph_path=work / "comfy_workflow.json",
        timing_path=work / "comfy_timing.json",
        attn="sage",
        require_accel=False,
    )
    produced = find_latest(outputs_dir, "gemmy/h3_face", t0, ".mp4")
    if produced is None:
        raise SystemExit("FaceRefine produced no MP4")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out)
    write_json(
        work / "result.json",
        {"ok": True, "output": str(out), "track": track, "stitch": stitch, "inject": inject},
    )
    print(json.dumps({"ok": True, "output": str(out), "class_ids": [track, stitch, inject]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
