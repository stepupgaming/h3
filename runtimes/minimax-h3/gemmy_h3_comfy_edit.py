"""Gemmy MiniMax-H3 masked video edit worker.

Stages (request.stage):
  track  — SAM3.1 (or --mask) → cleaned mask.mp4 + overlay.mp4; no Eros
  sample — crop + Eros + uncrop + mux (needs mask + ref-image)
  all    — track then sample in one process (no verify gate)

16 GB: SAM3 is one Comfy process; Eros is a second process. Overlay is
the look-at artifact for `gemmy analyze` / a human.

See video PJWfUAO1Oco / ganloss 2026-08-31 mask-edit: the mask answers
*where* to change; Picture 1 answers *who*.
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

import numpy as np
from PIL import Image

from h3_comfy_common import (
    copy_sidecar_next_to,
    existing_sidecar,
    find_latest,
    force_utf8_stdio,
    resolve_comfy_root,
    stage_file,
    write_json,
)
from h3_mask_edit import (
    apply_crop,
    cleanup_masks,
    compose_overlay,
    mask_coverage,
    plan_combined_crop,
    plan_tracked_crop,
    size_for_megapixels,
    uncrop,
)


def _ffmpeg() -> str:
    raw = os.environ.get("GEMMY_FFMPEG") or os.environ.get("FFMPEG")
    if raw and Path(raw).is_file():
        return raw
    here = Path(__file__).resolve().parent.parent / "ffmpeg" / "bin" / "ffmpeg.exe"
    if here.is_file():
        return str(here)
    return "ffmpeg"


def _run(cmd: list[str], *, label: str) -> None:
    print(f"[h3-edit] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit {proc.returncode}")


def _extract_frames(video: Path, dest: Path, frames: int, ffmpeg: str) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "%05d.png"
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=24",
            "-frames:v",
            str(int(frames)),
            str(pattern),
        ],
        label="extract-frames",
    )
    files = sorted(dest.glob("*.png"))
    if not files:
        raise SystemExit(f"no frames extracted from {video}")
    return files[: int(frames)]


def _extract_wav(video: Path, dest: Path, ffmpeg: str) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "32000",
            "-ac",
            "2",
            str(dest),
        ]
    )
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size < 64:
        print("[h3-edit] source has no usable audio; output will be silent until mux skip", flush=True)
        if dest.is_file():
            dest.unlink()
        return None
    return dest


def _load_rgb(paths: list[Path]) -> np.ndarray:
    frames = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            frames.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)


def _load_mask_frames(paths: list[Path]) -> np.ndarray:
    masks = []
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            masks.append(arr)
    return np.stack(masks, axis=0)


def _write_png_seq(frames: np.ndarray, dest: Path, *, mask: bool = False) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, fr in enumerate(frames, start=1):
        p = dest / f"{i:05d}.png"
        if mask:
            img = Image.fromarray(np.clip(fr * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        else:
            img = Image.fromarray(np.clip(fr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        img.save(p)
        out.append(p)
    return out


def _png_seq_to_mp4(frames_dir: Path, dest: Path, ffmpeg: str, n: int) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "24",
            "-i",
            str(frames_dir / "%05d.png"),
            "-frames:v",
            str(int(n)),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ],
        label=f"write-{dest.name}",
    )


def _mux(video: Path, audio: Path | None, dest: Path, ffmpeg: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if audio is None:
        shutil.copy2(video, dest)
        return
    _run(
        [
            ffmpeg,
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
            "-shortest",
            str(dest),
        ],
        label="mux-source-audio",
    )


def _comfy_python(comfy_root: Path, fallback: Path) -> Path:
    for cand in (
        comfy_root.parent / ".venv" / "Scripts" / "python.exe",
        comfy_root.parent / ".venv" / "bin" / "python",
        fallback,
    ):
        if cand.is_file():
            return cand
    return fallback


def _execute_graph(
    *,
    graph: dict[str, Any],
    graph_path: Path,
    comfy_root: Path,
    python: Path,
    attn: str,
    prefix: str,
    work: Path,
    persist_latent: bool = False,
) -> Path:
    write_json(graph_path, graph)
    runner = comfy_root / "run_h3_workflow.py"
    timing_path = graph_path.with_suffix(".timing.json")
    cmd = [
        str(python),
        "-u",
        str(runner),
        str(graph_path),
        "--out-note",
        str(timing_path),
        "--attn",
        attn,
    ]
    print(f"[h3-edit] execute: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(comfy_root), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"comfy runner failed with exit {proc.returncode}")
    outputs_dir = comfy_root / "output"
    if timing_path.is_file():
        try:
            note = json.loads(timing_path.read_text(encoding="utf-8"))
            if note.get("outputs_dir"):
                outputs_dir = Path(note["outputs_dir"])
        except Exception:
            pass
    mp4 = find_latest(outputs_dir, prefix, t0 - 1.0, ".mp4")
    if mp4 is None:
        raise SystemExit(f"no mp4 found under {outputs_dir} for prefix={prefix}")
    dest = work / f"{graph_path.stem}.mp4"
    shutil.copy2(mp4, dest)
    print(f"[h3-edit] wrote {dest}", flush=True)
    if persist_latent:
        st = find_latest(outputs_dir, prefix, t0 - 1.0, ".safetensors")
        if st is None:
            raise SystemExit(
                f"ganloss stage-1 did not persist AV latent under {outputs_dir} prefix={prefix}"
            )
        side = copy_sidecar_next_to(dest, st)
        print(f"[h3-edit] latent sidecar {side}", flush=True)
    return dest


def _pngs_or_extract(video: Path, dest: Path, frames: int, ffmpeg: str) -> list[Path]:
    existing = sorted(dest.glob("*.png"))
    if len(existing) >= int(frames):
        return existing[: int(frames)]
    return _extract_frames(video, dest, frames, ffmpeg)


def _run_track(req: dict[str, Any], *, work: Path, video: Path, ffmpeg: str) -> dict[str, Any]:
    h3_root = Path(req["h3_root"]).resolve()
    python = Path(req["python"]).resolve()
    frames = int(req["frames"])
    mask_grow = int(req.get("mask_grow") or 0)
    object_indices = str(req.get("object_indices") or "0")
    max_objects = int(req.get("max_objects") or 1)
    threshold = float(req.get("detection_threshold") or 0.5)
    mask_prompt = str(req.get("mask_prompt") or "").strip()
    sam3_ckpt = str(req.get("sam3_ckpt") or "sam3.1_multiplex_fp16.safetensors")

    comfy_root = resolve_comfy_root(req, h3_root)
    comfy_input = comfy_root / "input"
    job_tag = work.name.replace(" ", "_")[:40]
    py = _comfy_python(comfy_root, python)

    frame_files = _pngs_or_extract(video, work / "frames", frames, ffmpeg)
    originals = _load_rgb(frame_files)
    n = originals.shape[0]
    print(f"[h3-edit] source frames={n} {originals.shape[2]}x{originals.shape[1]}", flush=True)

    mask_path = req.get("mask")
    if mask_path:
        msrc = Path(str(mask_path)).resolve()
        if not msrc.is_file():
            raise SystemExit(f"mask missing: {msrc}")
        print(f"[h3-edit] using provided mask {msrc}", flush=True)
        mask_files = _extract_frames(msrc, work / "mask_frames_raw", n, ffmpeg)
        masks = _load_mask_frames(mask_files)
    else:
        if not mask_prompt:
            raise SystemExit("mask_prompt is required unless --mask is set")
        sys.path.insert(0, str(comfy_root))
        sys.path.insert(0, str(h3_root))
        from h3_comfy_common import bind_video_vae, materialize_workflow

        staged = stage_file(video, comfy_input, f"gemmy_{job_tag}_src.mp4")
        prefix = f"gemmy/{job_tag}_sam3"
        graph = materialize_workflow(
            "comfy-workflow-h3-sam3-track",
            {
                "video": staged,
                "prompt": mask_prompt,
                "checkpoint": sam3_ckpt,
                "output_prefix": prefix,
            },
        )
        _ = (object_indices, threshold, max_objects, n)
        print(f"[h3-edit] SAM3 track prompt={mask_prompt!r} object={object_indices}", flush=True)
        mask_mp4 = _execute_graph(
            graph=graph,
            graph_path=work / "sam3_workflow.json",
            comfy_root=comfy_root,
            python=py,
            attn="none",
            prefix=prefix,
            work=work,
        )
        mask_files = _extract_frames(mask_mp4, work / "mask_frames_raw", n, ffmpeg)
        masks = _load_mask_frames(mask_files)

    if masks.shape[0] != n:
        take = min(masks.shape[0], n)
        originals = originals[:take]
        masks = masks[:take]
        n = take
        print(f"[h3-edit] aligned to {n} frames after mask extract", flush=True)

    masks = cleanup_masks(masks, grow=mask_grow)
    cov = mask_coverage(masks)
    print(
        f"[h3-edit] mask coverage mean={cov['mean']:.3f} min={cov['min']:.3f} max={cov['max']:.3f}",
        flush=True,
    )
    if float(cov["mean"]) < 0.002:
        raise SystemExit(
            f"SAM3 mask is empty or near-empty (mean coverage {cov['mean']:.4f}). "
            "Retry --mask-prompt / --object-id, or pass --mask."
        )

    _write_png_seq(masks, work / "mask_frames", mask=True)
    mask_mp4_out = work / "mask.mp4"
    _png_seq_to_mp4(work / "mask_frames", mask_mp4_out, ffmpeg, n)

    overlay = compose_overlay(originals, masks)
    _write_png_seq(overlay, work / "overlay_png")
    overlay_mp4 = work / "overlay.mp4"
    _png_seq_to_mp4(work / "overlay_png", overlay_mp4, ffmpeg, n)
    mid = work / "overlay_mid.png"
    Image.fromarray(np.clip(overlay[n // 2] * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(mid)

    note = {
        "ok": True,
        "stage": "track",
        "frames": n,
        "width": int(originals.shape[2]),
        "height": int(originals.shape[1]),
        "mask": str(mask_mp4_out),
        "overlay": str(overlay_mp4),
        "overlay_mid": str(mid),
        "coverage": cov,
        "mask_prompt": mask_prompt,
        "mask_grow": mask_grow,
    }
    write_json(work / "track.json", note)
    print(f"[h3-edit] overlay {overlay_mp4}", flush=True)
    return note


def _run_sample(req: dict[str, Any], *, work: Path, video: Path, ffmpeg: str) -> None:
    h3_root = Path(req["h3_root"]).resolve()
    python = Path(req["python"]).resolve()
    output = Path(req["output"]).resolve()
    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit("prompt is required for the Eros sample stage")
    frames = int(req["frames"])
    steps = int(req.get("steps") or 8)
    seed = int(req.get("seed") or 42)
    crop_mode = str(req.get("crop_mode") or "combined").lower()
    crop_scale = float(req.get("crop_scale") or 1.5)
    crop_mp = float(req.get("crop_mp") or 1.0)
    feather = int(req.get("feather") or 8)
    mask_grow = int(req.get("mask_grow") or 0)

    comfy_root = resolve_comfy_root(req, h3_root)
    comfy_input = comfy_root / "input"
    job_tag = work.name.replace(" ", "_")[:40]
    py = _comfy_python(comfy_root, python)
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(h3_root))
    from h3_comfy_common import bind_video_vae, materialize_workflow

    src_wav = work / "source.wav"
    if not src_wav.is_file():
        src_wav = _extract_wav(video, src_wav, ffmpeg) or src_wav
        if not src_wav.is_file():
            src_wav_path = None
        else:
            src_wav_path = src_wav
    else:
        src_wav_path = src_wav

    frame_files = _pngs_or_extract(video, work / "frames", frames, ffmpeg)
    originals = _load_rgb(frame_files)
    n = originals.shape[0]

    mask_src = req.get("mask")
    if mask_src:
        msrc = Path(str(mask_src)).resolve()
        if not msrc.is_file():
            raise SystemExit(f"mask missing: {msrc}")
        mask_files = _pngs_or_extract(msrc, work / "mask_frames", n, ffmpeg)
        masks = _load_mask_frames(mask_files)
    else:
        raise SystemExit("sample stage needs request.mask (track first, or pass --mask)")

    if masks.shape[0] != n:
        take = min(masks.shape[0], n)
        originals = originals[:take]
        masks = masks[:take]
        n = take

    # Track already grew; sample grows only when the caller asked again on a raw mask.
    if mask_grow > 0 and not (work / "track.json").is_file():
        masks = cleanup_masks(masks, grow=mask_grow)
    elif mask_grow > 0:
        print("[h3-edit] mask already cleaned in track; not growing twice", flush=True)

    if crop_mode == "tracked":
        plan = plan_tracked_crop(
            masks, crop_scale=crop_scale, divisible_by=32, upscale_megapixels=crop_mp
        )
    else:
        plan = plan_combined_crop(
            masks, crop_scale=crop_scale, divisible_by=32, upscale_megapixels=crop_mp
        )
    write_json(work / "crop_plan.json", plan)
    cropped_f, cropped_m = apply_crop(originals, masks, plan)
    print(
        f"[h3-edit] crop {plan['mode']} {plan['out_width']}x{plan['out_height']} "
        f"from {plan['source_width']}x{plan['source_height']}",
        flush=True,
    )
    _write_png_seq(cropped_f, work / "crop_png")
    _write_png_seq(cropped_m, work / "crop_mask_png", mask=True)
    crop_mp4 = work / "crop.mp4"
    crop_mask_mp4 = work / "crop_mask.mp4"
    _png_seq_to_mp4(work / "crop_png", crop_mp4, ffmpeg, n)
    _png_seq_to_mp4(work / "crop_mask_png", crop_mask_mp4, ffmpeg, n)

    ref_bases: list[str] = []
    for i, raw in enumerate(req.get("ref_images") or []):
        src = Path(str(raw)).resolve()
        if not src.is_file():
            raise SystemExit(f"ref-image missing: {src}")
        ext = src.suffix.lower() or ".png"
        ref_bases.append(stage_file(src, comfy_input, f"gemmy_{job_tag}_ref{i}{ext}"))
    if not ref_bases:
        raise SystemExit("mask edit needs at least one --ref-image")

    crop_base = stage_file(crop_mp4, comfy_input, f"gemmy_{job_tag}_crop.mp4")
    crop_w = int(plan["out_width"])
    crop_h = int(plan["out_height"])
    stage1_w, stage1_h = size_for_megapixels(crop_w, crop_h, 0.2, 32)
    weights_name = Path(str(req.get("weights") or "")).name
    unet_name = weights_name if weights_name.endswith(".safetensors") else None
    attn = str(req.get("attn") or "sage")
    shift_v = float(req.get("shift_video") or 12.0)
    shift_a = float(req.get("shift_audio") or 3.0)
    paired_audio: list[str | None] | None = None
    if src_wav_path is not None:
        paired_audio = [
            stage_file(src_wav_path, comfy_input, f"gemmy_{job_tag}_src.wav")
        ]

    prefix1 = f"gemmy/{job_tag}_s1"
    stage1_params = {
        "prompt": prompt,
        "width": stage1_w,
        "height": stage1_h,
        "length": n,
        "seed": seed,
        "steps": steps,
        "ref_image": ref_bases[0],
        "crop_video": crop_base,
        "unet": unet_name,
        "shift_video": shift_v,
        "shift_audio": shift_a,
        "output_prefix": prefix1,
        "latent_prefix": f"{prefix1}_av",
    }
    bind_video_vae(stage1_params, req)
    graph1 = materialize_workflow(
        "comfy-workflow-h3-ganloss-stage1",
        stage1_params,
    )
    print(
        f"[h3-edit] ganloss stage-1 {stage1_w}x{stage1_h} SplitSigmas@4 "
        f"ref_video={crop_w}x{crop_h} frames={n} steps={steps}",
        flush=True,
    )
    stage1_mp4 = _execute_graph(
        graph=graph1,
        graph_path=work / "edit_stage1.json",
        comfy_root=comfy_root,
        python=py,
        attn=attn,
        prefix=prefix1,
        work=work,
        persist_latent=True,
    )
    stage1_latent = existing_sidecar(stage1_mp4)
    if stage1_latent is None:
        raise SystemExit(f"stage-1 AV latent missing next to {stage1_mp4}")

    prefix2 = f"gemmy/{job_tag}_s2"
    stage2_params = {
        "prompt": prompt,
        "width": crop_w,
        "height": crop_h,
        "length": n,
        "seed": seed,
        "ref_image": ref_bases[0],
        "crop_video": crop_base,
        "unet": unet_name,
        "shift_video": shift_v,
        "shift_audio": shift_a,
        "context_latent": str(stage1_latent),
        "output_prefix": prefix2,
    }
    bind_video_vae(stage2_params, req)
    graph2 = materialize_workflow(
        "comfy-workflow-h3-ganloss-stage2",
        stage2_params,
    )
    print(
        f"[h3-edit] ganloss stage-2 3D SR + 3-step ManualSigmas → {crop_w}x{crop_h}",
        flush=True,
    )
    crop_out = _execute_graph(
        graph=graph2,
        graph_path=work / "edit_stage2.json",
        comfy_root=comfy_root,
        python=py,
        attn=attn,
        prefix=prefix2,
        work=work,
    )
    processed_files = _extract_frames(crop_out, work / "processed_png", n, ffmpeg)
    processed = _load_rgb(processed_files)
    if processed.shape[0] != n:
        take = min(processed.shape[0], n)
        processed = processed[:take]
        originals = originals[:take]
        cropped_m = cropped_m[:take]
        n = take
    if processed.shape[1] != crop_h or processed.shape[2] != crop_w:
        resized = []
        for fr in processed:
            img = Image.fromarray(np.clip(fr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
            img = img.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
            resized.append(np.asarray(img, dtype=np.float32) / 255.0)
        processed = np.stack(resized, axis=0)
    rebuilt = uncrop(processed, originals, plan, cropped_m, feather=feather)
    _write_png_seq(rebuilt, work / "uncrop_png")
    silent = work / "uncrop_silent.mp4"
    _png_seq_to_mp4(work / "uncrop_png", silent, ffmpeg, n)
    output.parent.mkdir(parents=True, exist_ok=True)
    _mux(silent, src_wav_path, output, ffmpeg)
    write_json(
        work / "result.json",
        {
            "ok": True,
            "output": str(output),
            "frames": n,
            "crop": {
                "width": crop_w,
                "height": crop_h,
                "mode": plan["mode"],
            },
            "ganloss": {
                "stage1": [stage1_w, stage1_h],
                "stage2": [crop_w, crop_h],
                "ref_video": True,
            },
        },
    )
    print(f"[h3-edit] wrote {output}", flush=True)


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser(description="Gemmy MiniMax-H3 masked video edit")
    ap.add_argument("--request", type=Path, required=True)
    args = ap.parse_args(argv)
    req = json.loads(args.request.read_text(encoding="utf-8"))

    work = Path(req["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    video = Path(req["video"]).resolve()
    if not video.is_file():
        raise SystemExit(f"source video missing: {video}")
    ffmpeg = _ffmpeg()
    stage = str(req.get("stage") or "all").strip().lower()
    if stage not in {"track", "sample", "all"}:
        raise SystemExit(f"unknown stage {stage!r} (track|sample|all)")

    if stage in {"track", "all"}:
        _extract_wav(video, work / "source.wav", ffmpeg)
        _run_track(req, work=work, video=video, ffmpeg=ffmpeg)
        if stage == "track":
            return 0
        req = dict(req)
        req["mask"] = str(work / "mask.mp4")
        req["mask_grow"] = 0
    _run_sample(req, work=work, video=video, ffmpeg=ffmpeg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
