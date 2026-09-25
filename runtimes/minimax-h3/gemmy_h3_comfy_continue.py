"""Gemmy MiniMax-H3 Comfy same-shot continue worker.

Default: native_guide (previous AV tail as keyframes, no denoise mask).
`--legacy-masked-av` keeps the freeze-prefix copy. Imported MP4 without a
sidecar uses VAE-tail once, then native_guide from the new sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

H3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(H3_ROOT))

from h3_comfy_common import (  # noqa: E402
    COMFY_SERVE_ENV,
    assemble_mp4s,
    copy_sidecar_next_to,
    existing_sidecar,
    fingerprint_payload,
    fingerprints_match,
    force_utf8_stdio,
    load_fingerprint,
    start_comfy_serve,
    stop_comfy_serve,
    video_vae_filename,
    video_vae_fingerprint_id,
    write_json,
)


def _snap_context(requested: int, available: int) -> int:
    sys.path.insert(0, str(H3_ROOT / "ComfyUI" / "custom_nodes" / "gemmy-h3-context"))
    from av_math import snap_context_frames  # type: ignore

    return snap_context_frames(requested, available)


def _assemble_native_guide(
    source: Path | None, segs: list[Path], output: Path, overlap_frames: int
) -> None:
    """SatoDive stitch: cut the source at the re-rendered tail, then append."""
    sys.path.insert(0, str(H3_ROOT / "ComfyUI" / "custom_nodes" / "gemmy-h3-context"))
    from stitch_continuation import stitch_mp4s  # type: ignore

    if not segs:
        raise SystemExit("no segments to assemble")
    acc = source
    tmp_dir = output.parent / f".{output.stem}_stitch"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(segs):
        if acc is None:
            acc = seg
            continue
        nxt = tmp_dir / f"join_{i:02d}.mp4"
        stitch_mp4s(acc, seg, nxt, overlap_frames=overlap_frames)
        acc = nxt
    output.parent.mkdir(parents=True, exist_ok=True)
    if acc.resolve() != output.resolve():
        shutil.copy2(acc, output)


def main() -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))

    python = Path(req.get("python") or sys.executable)
    h3_root = Path(req.get("h3_root") or H3_ROOT)
    work = Path(req["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    output = Path(req["output"])
    generate_worker = h3_root / "gemmy_h3_comfy_generate.py"
    if not generate_worker.is_file():
        raise SystemExit(f"generate worker missing: {generate_worker}")

    width = int(req["width"])
    height = int(req["height"])
    window_frames = int(req["window_frames"])
    num_windows = int(req.get("num_windows") or 2)
    context_req = int(req.get("context_frames") or 39)
    context_frames = _snap_context(context_req, window_frames)
    seed = int(req.get("seed") or 42)
    seed_inc = bool(req.get("seed_inc"))
    prompt = str(req.get("prompt") or "")
    input_mp4 = Path(req["input"]) if req.get("input") else None
    start_image = Path(req["start_image"]) if req.get("start_image") else None
    weights = req.get("weights")
    graph_mode = str(req.get("mode") or "t2va").lower()
    refmods = req.get("refmods") or []
    refs = req.get("refs") or []
    quality = str(req.get("quality") or "fast")
    attn = str(req.get("attn") or "sage")
    compile_ir = bool(req.get("compile_ir", True))
    keep_segs = bool(req.get("keep_segs"))

    vae_name = video_vae_filename(req)
    fp_template = fingerprint_payload(
        width=width,
        height=height,
        frames=window_frames,
        dit=str(weights or ""),
        mode="continue",
        context_frames=context_frames,
        vae=video_vae_fingerprint_id(vae_name),
    )

    kwargs = dict(
        req=req,
        python=python,
        h3_root=h3_root,
        work=work,
        output=output,
        generate_worker=generate_worker,
        width=width,
        height=height,
        window_frames=window_frames,
        num_windows=num_windows,
        context_frames=context_frames,
        seed=seed,
        seed_inc=seed_inc,
        prompt=prompt,
        input_mp4=input_mp4,
        start_image=start_image,
        weights=weights,
        graph_mode=graph_mode,
        refmods=refmods,
        refs=refs,
        quality=quality,
        attn=attn,
        compile_ir=compile_ir,
        keep_segs=keep_segs,
        vae_name=vae_name,
        fp_template=fp_template,
    )
    if num_windows < 2:
        return _continue_windows(**kwargs)

    comfy_root = Path(str(req.get("comfy_root") or (h3_root / "ComfyUI")))
    print(f"[h3-continue] keeping Comfy warm for {num_windows} windows (cached UNET/TE/VAE)", flush=True)
    serve_proc, serve_addr = start_comfy_serve(
        comfy_root=comfy_root,
        python=python,
        attn=attn,
    )
    os.environ[COMFY_SERVE_ENV] = serve_addr
    try:
        return _continue_windows(**kwargs)
    finally:
        stop_comfy_serve(serve_proc, serve_addr)


def _continue_windows(**kw) -> int:
    req = kw["req"]
    python = kw["python"]
    h3_root = kw["h3_root"]
    work = kw["work"]
    output = kw["output"]
    generate_worker = kw["generate_worker"]
    width = kw["width"]
    height = kw["height"]
    window_frames = kw["window_frames"]
    num_windows = kw["num_windows"]
    context_frames = kw["context_frames"]
    seed = kw["seed"]
    seed_inc = kw["seed_inc"]
    prompt = kw["prompt"]
    input_mp4 = kw["input_mp4"]
    start_image = kw["start_image"]
    weights = kw["weights"]
    graph_mode = kw["graph_mode"]
    refmods = kw["refmods"]
    refs = kw["refs"]
    quality = kw["quality"]
    attn = kw["attn"]
    compile_ir = kw["compile_ir"]
    keep_segs = kw["keep_segs"]
    vae_name = kw["vae_name"]
    fp_template = kw["fp_template"]
    segs: list[Path] = []
    prev_latent: Path | None = None
    prev_video: Path | None = None
    first_mode = "none"
    same_shot = "masked_av" if bool(req.get("legacy_masked_av")) else "native_guide"
    if input_mp4 is not None:
        prev_video = input_mp4
        side = existing_sidecar(input_mp4)
        if side is not None:
            prev_latent = side
            first_mode = same_shot
        else:
            first_mode = "vae_tail"

    for i in range(num_windows):
        w = i + 1
        wdir = work / f"window_{w:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        seg = wdir / "segment.mp4"
        wjson = wdir / "window.json"
        side = existing_sidecar(seg)
        if seg.is_file() and side is not None and wjson.is_file():
            try:
                prev_fp = json.loads(wjson.read_text(encoding="utf-8")).get("fingerprint")
            except json.JSONDecodeError:
                prev_fp = None
            if fingerprints_match(prev_fp, fp_template):
                print(f"[h3-continue] resume window {w} (fingerprint match)", flush=True)
                segs.append(seg)
                prev_latent = side
                prev_video = seg
                continue
            print(f"[h3-continue] window {w} fingerprint changed; regenerating", flush=True)

        if i == 0:
            if first_mode in ("native_guide", "masked_av") and prev_latent is not None:
                continuation = first_mode
                context_latent = str(prev_latent)
                context_video = None
                trim_prefix = continuation == "masked_av"
                mode = graph_mode
                first_image = None
            elif first_mode == "vae_tail" and prev_video is not None:
                continuation = "vae_tail"
                context_latent = None
                context_video = str(prev_video)
                trim_prefix = True
                mode = graph_mode
                first_image = None
            else:
                continuation = "none"
                context_latent = None
                context_video = None
                trim_prefix = False
                mode = "i2v" if start_image is not None else graph_mode
                first_image = str(start_image) if start_image is not None else None
        else:
            if prev_latent is None:
                raise SystemExit(f"window {w} needs previous AV sidecar; none found")
            continuation = same_shot
            context_latent = str(prev_latent)
            context_video = None
            trim_prefix = continuation == "masked_av"
            mode = graph_mode
            first_image = None

        win_req: dict[str, Any] = {
            "prompt": prompt,
            "output": str(seg),
            "work_dir": str(wdir / "work"),
            "h3_root": str(h3_root),
            "checkpoints_root": req.get("checkpoints_root"),
            "comfy_root": req.get("comfy_root"),
            "python": str(python),
            "mode": mode,
            "quality": quality,
            "attn": attn,
            "engine": "comfy",
            "turbo": "off",
            "realism": False,
            "sol": False,
            "cache": "off",
            "dit_quant": "int8",
            "width": width,
            "height": height,
            "frames": window_frames,
            "steps": int(req.get("steps") or 20),
            "seed": seed + i if seed_inc else seed,
            "compile_ir": compile_ir,
            "first_image": first_image,
            "weights": weights,
            "persist_av_latent": True,
            "continuation_mode": continuation,
            "context_latent": context_latent,
            "context_video": context_video,
            "context_frames": context_frames,
            "trim_prefix": trim_prefix,
            "audio_mode": "generated_audio",
            "video_vae": vae_name,
            "refmods": refmods,
            "refs": refs,
        }
        req_path = wdir / "request.json"
        write_json(req_path, win_req)
        write_json(
            wjson,
            {
                "window": w,
                "continuation_mode": continuation,
                "fingerprint": fp_template,
                "segment": str(seg),
            },
        )
        print(
            f"[h3-continue] window {w}/{num_windows} mode={continuation} "
            f"frames={window_frames} ctx={context_frames}",
            flush=True,
        )
        proc = subprocess.run(
            [str(python), str(generate_worker), "--request", str(req_path)],
            cwd=str(h3_root),
        )
        if proc.returncode != 0:
            raise SystemExit(f"window {w} generate failed with exit {proc.returncode}")
        if not seg.is_file():
            raise SystemExit(f"window {w} missing {seg}")
        side = existing_sidecar(seg)
        if side is None:
            raise SystemExit(f"window {w} missing AV sidecar next to {seg}")
        segs.append(seg)
        prev_latent = side
        prev_video = seg

    if same_shot == "native_guide":
        _assemble_native_guide(input_mp4, segs, output, context_frames)
    else:
        assemble_mp4s(segs, output)
    if segs:
        copy_sidecar_next_to(output, existing_sidecar(segs[-1]))
    result = {
        "ok": True,
        "path": same_shot,
        "windows": len(segs),
        "context_frames": context_frames,
        "first_window": first_mode,
        "output": str(output),
        "av_latent": str(existing_sidecar(output) or ""),
    }
    write_json(work / "result.json", result)
    print(json.dumps(result), flush=True)
    if not keep_segs:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
