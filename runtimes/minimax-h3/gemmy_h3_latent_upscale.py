"""H3 latent upscale.

``refine`` (default product path): LBH recommended — enlarge the leftover
working picture, then run a second H3 sample at the new size, then decode.

``preview``: enlarge then decode only (their quick-preview graph).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

H3_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(H3_ROOT))

from h3_comfy_common import (  # noqa: E402
    bind_video_vae,
    compile_ir_text,
    copy_sidecar_next_to,
    existing_sidecar,
    find_latest,
    fingerprint_payload,
    force_utf8_stdio,
    load_fingerprint,
    materialize_workflow,
    resolve_comfy_root,
    video_vae_filename,
    video_vae_fingerprint_id,
    run_comfy_graph,
    stage_file,
    write_json,
)

# Official LBH I2V example refine schedule (3-step euler).
LBH_REFINE_SIGMAS = "0.9035, 0.6316, 0.3158, 0.0000"


def _discover_upscaler(comfy_root: Path) -> str:
    nodes = comfy_root / "custom_nodes"
    candidates = [
        nodes / "Comfyui_Minimax_h3_latent_Upscaler",
        nodes / "ComfyUI_Minimax_h3_latent_Upscaler",
        nodes / "Comfyui-Minimax-h3-latent-Upscaler",
    ]
    for pack in candidates:
        if not pack.is_dir():
            continue
        mapping: dict = {}
        init = pack / "__init__.py"
        if init.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("gemmy_h3_latent_up_pack", init)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    mapping = dict(getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {})
                except Exception:
                    mapping = {}
        if "MinimaxH3LatentUpscaler3D" in mapping:
            return "MinimaxH3LatentUpscaler3D"
        for name in mapping:
            if "3D" in name or name.endswith("3d"):
                return name
        return "MinimaxH3LatentUpscaler3D"
    raise SystemExit(
        "Comfyui_Minimax_h3_latent_Upscaler is not installed under "
        f"{nodes}. Pin the Apache-2.0 weights and the 3D node pack."
    )


def _resolve_model_name(req: dict, h3_root: Path) -> str:
    model_name = str(req.get("model_name") or "")
    ckpt = Path(req.get("checkpoints_root") or "")
    models_dir = ckpt / "latent_upscale_models" if ckpt else h3_root / "latent_upscale_models"
    if not model_name:
        if models_dir.is_dir():
            found = sorted(models_dir.glob("*.safetensors"))
            if found:
                model_name = found[0].name
    if not model_name:
        raise SystemExit(
            f"no 3D latent upscaler weights under {models_dir} "
            "(see docs/MODEL_LOCATIONS.md latent_upscale_models/)"
        )
    return model_name


def _refine_graph(
    *,
    req: dict,
    h3_root: Path,
    comfy_root: Path,
    python: Path,
    work: Path,
    side: Path,
    model_name: str,
    src_w: int,
    src_h: int,
    prev_fp: dict,
) -> dict:
    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit(
            "latent refine needs --prompt (same scene as the source clip). "
            "Use --backend latent-preview to enlarge and decode without a second H3 pass."
        )
    frames = int(prev_fp.get("frames") or req.get("frames") or 0)
    if frames <= 0:
        raise SystemExit("latent refine needs frame count in the sidecar fingerprint")

    mode = str(req.get("mode") or "t2va").lower()
    graph_mode = "fl2va" if mode == "i2v" else mode
    turbo = str(req.get("turbo") or "off").lower()
    realism = bool(req.get("realism") or False)
    compile_ir = bool(req.get("compile_ir", True))
    seed = int(req.get("seed") or 42)
    steps = int(req.get("steps") or 8)
    denoise = req.get("denoise")
    refine_sigmas = req.get("refine_sigmas")
    if denoise is not None:
        denoise = float(denoise)
        refine_sigmas = None
    elif refine_sigmas:
        refine_sigmas = str(refine_sigmas)
    else:
        refine_sigmas = LBH_REFINE_SIGMAS
        denoise = 1.0

    text = prompt
    if compile_ir:
        ir_mode = "ref2va" if graph_mode == "ref2va" else "base"
        text, _ir_s = compile_ir_text(
            python=python,
            h3_root=h3_root,
            prompt=prompt,
            ir_mode=ir_mode,
            frames=frames,
            work=work,
            realism=realism,
        )

    comfy_input = comfy_root / "input"
    first_base = None
    last_base = None
    if req.get("first_image"):
        src = Path(req["first_image"])
        if not src.is_file():
            raise SystemExit(f"first-frame missing: {src}")
        first_base = stage_file(src, comfy_input, f"gemmy_latref_first{src.suffix.lower() or '.png'}")
    if req.get("last_image"):
        src = Path(req["last_image"])
        if not src.is_file():
            raise SystemExit(f"last-frame missing: {src}")
        last_base = stage_file(src, comfy_input, f"gemmy_latref_last{src.suffix.lower() or '.png'}")

    ref_bases = []
    ref_audio_bases = []
    for i, item in enumerate(req.get("refs") or []):
        src = Path(item["path"] if isinstance(item, dict) else item)
        if not src.is_file():
            raise SystemExit(f"refine ref missing: {src}")
        kind = str(item.get("kind") or "image").lower() if isinstance(item, dict) else "image"
        ext = src.suffix.lower()
        if kind == "audio" or ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            if not ext:
                ext = ".wav"
            ref_audio_bases.append(stage_file(src, comfy_input, f"gemmy_latref_refa{i}{ext}"))
        else:
            if not ext:
                ext = ".png"
            ref_bases.append(stage_file(src, comfy_input, f"gemmy_latref_ref{i}{ext}"))

    sys.path.insert(0, str(comfy_root))

    vae_name = video_vae_filename(req)
    fp = fingerprint_payload(
        width=src_w,
        height=src_h,
        frames=frames,
        dit=str(req.get("weights") or prev_fp.get("dit") or ""),
        lora=(
            (["h3-realism-people-t2v-i2v-r2v"] if realism else [])
            + ([f"turbo-{turbo}"] if turbo != "off" else [])
        ),
        # Keep the predecessor mode so loop/continue fingerprint matches at the new canvas.
        mode=str(prev_fp.get("mode") or mode),
        vae=video_vae_fingerprint_id(vae_name),
        extra={
            "path": "latent_refine",
            "source_digest": prev_fp.get("digest"),
            "source_mode": prev_fp.get("mode"),
        },
    )
    weights_name = Path(str(req.get("weights") or "")).name
    unet_name = weights_name if weights_name.endswith(".safetensors") else None
    params: dict = {
        "prompt": text,
        "width": src_w,
        "height": src_h,
        "length": frames,
        "seed": seed,
        "context_latent": str(side),
        "upscale_model": model_name,
        "output_prefix": "gemmy/h3_latent_up",
        "latent_prefix": "gemmy/h3_latent_up",
        "fingerprint": json.dumps(fp),
    }
    bind_video_vae(params, req)
    if unet_name:
        params["unet"] = unet_name
    if first_base:
        params["first_image"] = first_base
    if last_base:
        params["last_image"] = last_base
    vsa = bool(req.get("vsa") or False)
    refmods_raw = req.get("refmods") or []
    none = "(none)"
    for i in range(1, 5):
        params[f"refmod_{i}"] = none
        params[f"refmod_strength_{i}"] = 1.0
        params[f"refmod_copies_{i}"] = 1
    for i, row in enumerate(refmods_raw[:4], start=1):
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        params[f"refmod_{i}"] = name
        params[f"refmod_strength_{i}"] = float(row.get("strength") if row.get("strength") is not None else 1.0)
        params[f"refmod_copies_{i}"] = int(row.get("copies") or 1)
    if refmods_raw:
        params["refmod_retention"] = float(req.get("refmod_retention") if req.get("refmod_retention") is not None else 1.0)
        if vsa:
            raise SystemExit("no compiled workflow package for --vsa plus --ref-mod")
    n_aud = len(ref_audio_bases)
    if n_aud:
        params["reference_audio"] = ref_audio_bases[0]
        if n_aud > 1:
            params["reference_audio_2"] = ref_audio_bases[1]
        if n_aud > 2:
            raise SystemExit("Eros stage-2 supports 1 or 2 --ref-audio")
        if vsa:
            raise SystemExit("no compiled workflow package for --vsa plus --ref-audio on Eros stage-2")
        if n_aud == 2 and refmods_raw:
            raise SystemExit("no compiled workflow package for --ref-mod plus two --ref-audio on Eros stage-2")
    if ref_bases:
        params["ref_image"] = ref_bases[0]
        pkg = "comfy-workflow-h3-ganloss-stage2-still"
        if refmods_raw:
            pkg += "-refmod"
        elif vsa:
            pkg += "-vsa"
            params["gate_file"] = str(req.get("vsa_gate") or "fasth3_vsa_gate.safetensors")
            params["sparsity"] = float(req.get("vsa_sparsity") if req.get("vsa_sparsity") is not None else 0.75)
        if n_aud == 1:
            pkg += "-audio"
        elif n_aud == 2:
            pkg += "-2audio"
    else:
        pkg = "comfy-workflow-h3-latent-refine"
        if vsa:
            raise SystemExit("--vsa on latent refine needs a Ref2VA still (ganloss stage-2)")
        if refmods_raw:
            raise SystemExit("--ref-mod on latent refine needs a Ref2VA still (ganloss stage-2-still-refmod)")
        if n_aud:
            raise SystemExit("--ref-audio on latent refine needs a Ref2VA still (ganloss stage-2-still-audio)")
    return materialize_workflow(pkg, params)


def main() -> int:
    force_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    args = ap.parse_args()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    h3_root = Path(req.get("h3_root") or H3_ROOT)
    comfy_root = resolve_comfy_root(req, h3_root)
    python = Path(req.get("python") or sys.executable)
    inp = Path(req["input"])
    out = Path(req["output"])
    work = Path(req.get("work_dir") or out.parent / ".h3_latent_up")
    work.mkdir(parents=True, exist_ok=True)
    scale = int(req.get("scale") or 2)
    width = req.get("width")
    height = req.get("height")
    refine = bool(req.get("refine", True))

    cls = _discover_upscaler(comfy_root)
    side = existing_sidecar(inp)
    if side is None:
        raise SystemExit(
            f"latent upscale needs {inp.with_suffix('.h3av.safetensors')} next to the MP4"
        )

    prev_fp = load_fingerprint(side) or {}
    src_w = int(width or prev_fp.get("width") or 0)
    src_h = int(height or prev_fp.get("height") or 0)
    if src_w <= 0 or src_h <= 0:
        raise SystemExit("latent upscale needs predecessor width/height in the sidecar fingerprint")
    if not width and not height:
        src_w *= scale
        src_h *= scale

    model_name = _resolve_model_name(req, h3_root)
    if refine:
        print(
            f"[h3 latent] refine {prev_fp.get('width')}x{prev_fp.get('height')} "
            f"→ {src_w}x{src_h} (LBH enlarge + second H3 sample)",
            flush=True,
        )
        graph = _refine_graph(
            req=req,
            h3_root=h3_root,
            comfy_root=comfy_root,
            python=python,
            work=work,
            side=side,
            model_name=model_name,
            src_w=src_w,
            src_h=src_h,
            prev_fp=prev_fp,
        )
    else:
        print(
            f"[h3 latent] preview enlarge {prev_fp.get('width')}x{prev_fp.get('height')} "
            f"→ {src_w}x{src_h} (decode only, no second H3 sample)",
            flush=True,
        )
        preview_params: dict = {
            "context_latent": str(side),
            "upscale_model": model_name,
            "scale": float(scale),
            "output_prefix": "gemmy/h3_latent_up",
            "latent_prefix": "gemmy/h3_latent_up",
            "fingerprint": json.dumps(
                fingerprint_payload(
                    width=src_w,
                    height=src_h,
                    frames=int(prev_fp.get("frames") or 0),
                    dit=str(prev_fp.get("dit") or "latent-upscale"),
                    mode="latent_preview",
                    vae=video_vae_fingerprint_id(video_vae_filename(req)),
                    extra={"source_digest": prev_fp.get("digest")},
                )
            ),
        }
        bind_video_vae(preview_params, req)
        if width and height:
            preview_params["mode"] = "target dimensions"
            preview_params["target_width"] = int(width)
            preview_params["target_height"] = int(height)
        graph = materialize_workflow("comfy-workflow-h3-latent-preview", preview_params)
        if cls != "MinimaxH3LatentUpscaler3D":
            raise SystemExit(
                f"latent-preview requires MinimaxH3LatentUpscaler3D; discovered {cls}. "
                "Do not rewrite class_type in Python."
            )

    t0 = time.time()
    outputs_dir, _note, _elapsed = run_comfy_graph(
        comfy_root=comfy_root,
        python=python,
        graph=graph,
        graph_path=work / "comfy_workflow.json",
        timing_path=work / "comfy_timing.json",
        attn=str(req.get("attn") or "sage"),
        require_accel=refine and (
            str(req.get("turbo") or "off") != "off"
            or bool(req.get("sol"))
            or str(req.get("cache") or "off") != "off"
            or bool(req.get("realism"))
        ),
    )
    produced = find_latest(outputs_dir, "gemmy/h3_latent_up", t0, ".mp4")
    if produced is None:
        raise SystemExit("latent upscale produced no MP4")
    shutil.copy2(produced, out)
    av = find_latest(outputs_dir, "gemmy/h3_latent_up", t0, ".safetensors")
    copy_sidecar_next_to(out, av)
    write_json(
        work / "result.json",
        {
            "ok": True,
            "output": str(out),
            "class": cls,
            "refine": refine,
            "canvas": [src_w, src_h],
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "class": cls,
                "refine": refine,
                "width": src_w,
                "height": src_h,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
