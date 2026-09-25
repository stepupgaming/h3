"""Gemmy MiniMax-H3 Comfy generate worker (in-process).

Same request.json contract as gemmy_h3_generate.py, plus engine-specific fields:

  engine: "comfy"
  turbo: "off"|"v4"|"v1"|"ema_wan"
  turbo_strength: float (Larryvrh default 1.0)
  realism: bool                     # fal MiniMax-H3-Realism-People-LoRA
  realism_strength: float           # default 1.0 (card: 0.6–0.8 lighter)
  sol: bool
  vsa: bool
  vsa_sparsity: float
  vsa_gate: str
  jev: bool                      # opt-in 009jev native SLA packages (not default)
  jev_sdk_python: str            # dedicated typesafe-sdk interpreter (adaptive only)
  jev_initial_policy: str        # jev_first (adaptive), const1|3|5|10, or table (per-step/layer; no Jev)
  sla_fixed: int|null            # 1|3|5|10 true fixed keep; bypasses Jev worker
  sla_table: str|null            # JSON path with keep_table N×50; initial_policy=table; no Jev
  keep_table: list|null          # inline N×50 keeps (same as sla_table file)
  jev_log_dataset: str|null      # append teacher JSONL after a successful generate
  cache: "off"|"spectrum"|"easy"|"fbc"
  dit_quant: "int8"|"w4a8"
  comfy_root: str|null   # default: <h3_root>/ComfyUI (internalized); GEMMY_H3_COMFY override
  attn: "sage"|"fa2"     # optional override of quality mapping

Sequences optional IR compile, then one Comfy graph (TE+DiT+VAE inside Comfy).
`--dit singularity` is the exception: four fresh processes (encode, turbo sample,
final 10 steps, decode). Each process exits before the next loads. Writes
work_dir/result.json and copies mp4 to output.
Comfy engine code is fully under runtimes/minimax-h3/ComfyUI — no research-tree dep.
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
) -> None:
    print(f"[h3-comfy] {label}: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit {proc.returncode}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _native_sla_policy():
    node = Path(__file__).resolve().parent / "ComfyUI" / "custom_nodes" / "ComfyUI-MiniMax-H3-009jev"
    if str(node) not in sys.path:
        sys.path.insert(0, str(node))
    import native_sla_policy as nsp  # noqa: E402

    return nsp


def load_keep_table(req: dict[str, Any]) -> list[list[float]] | None:
    """Load an N×50 keep table from request.keep_table or request.sla_table path."""
    raw = req.get("keep_table")
    path = str(req.get("sla_table") or "").strip()
    if path:
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"sla_table missing: {p}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        raw = obj.get("keep_table", obj) if isinstance(obj, dict) else obj
    if raw is None or raw == "":
        return None
    return _native_sla_policy().normalize_keep_table(raw)


def inject_keep_table(graph: dict[str, Any], keep_table: list[list[float]]) -> None:
    """Bind initial_policy=table and keep_table onto the live H3JevNativeSLAPatch node."""
    context = json.dumps({"keep_table": keep_table}, allow_nan=False)
    hits = 0
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") != "H3JevNativeSLAPatch":
            continue
        inputs = node.setdefault("inputs", {})
        inputs["initial_policy"] = "table"
        inputs["initial_context"] = context
        inputs["sdk_python"] = inputs.get("sdk_python") or ""
        hits += 1
    if hits != 1:
        raise SystemExit(f"expected one H3JevNativeSLAPatch to inject keep_table, found {hits}")


def _resolve_comfy_root(req: dict[str, Any], h3_root: Path) -> Path:
    """Production default: internalized ``<h3_root>/ComfyUI``.

    External research trees are never consulted. Optional override only via
    request ``comfy_root`` or env ``GEMMY_H3_COMFY`` (dev A/B).
    """
    candidates: list[Path] = []
    raw = req.get("comfy_root") or os.environ.get("GEMMY_H3_COMFY")
    if raw:
        candidates.append(Path(str(raw)))
    candidates.append(h3_root / "ComfyUI")
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "run_h3_workflow.py").is_file() and (c / "comfy").is_dir():
            return c
    raise SystemExit(
        "ComfyUI root not found under the internalized H3 runtime.\n"
        f"Expected: {h3_root / 'ComfyUI' / 'run_h3_workflow.py'}\n"
        "Dev override only: set GEMMY_H3_COMFY or request.comfy_root."
    )


def _stage_image(src: Path, comfy_input: Path, dest_name: str) -> str:
    """Copy/link still into Comfy input/; return basename for LoadImage."""
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


def _packaged_generate_name(
    *,
    graph_mode: str,
    turbo: str,
    continuation_mode: str,
    first_base: str | None,
    last_base: str | None,
    ref_bases: list[str],
    ref_video_bases: list[str],
    ref_audio_bases: list[str],
    guide_images: list[dict[str, Any]],
    init_audio_base: str | None,
    persist_latent: bool,
    sol: bool,
    vsa: bool,
    cache: str,
    realism: bool,
    refmods: list[dict[str, Any]],
    jev: bool = False,
    no_sla: bool = False,
    steps: int = 4,
) -> str:
    """Map a generate request onto a compiled workflow package.

    Topology lives in comfy-workflows packages. Unknown combos fail closed
    instead of falling back to the retired Python composer.
    """
    mode = graph_mode if graph_mode != "t2va" else "t2va"
    if continuation_mode == "singularity_dual":
        if vsa or sol or cache != "off" or turbo != "off" or jev or no_sla or refmods:
            raise SystemExit(
                "singularity dual-sample graph does not stack --vsa, --sol, --cache, "
                "--turbo, --jev, --sla-fixed, --sla-table, --no-sla, or --ref-mod"
            )
        if last_base or ref_video_bases or ref_audio_bases or len(ref_bases) != 1:
            raise SystemExit(
                "singularity dual-sample graph needs exactly one --ref-image"
            )
        return "singularity-split"
    if no_sla and jev:
        raise SystemExit("no compiled workflow package for --no-sla plus --jev")
    if no_sla:
        if vsa:
            raise SystemExit("no compiled workflow package for --no-sla plus --vsa")
        if sol or cache != "off" or turbo != "off" or realism:
            raise SystemExit(
                "no compiled workflow package for --no-sla plus turbo/sol/realism/cache"
            )
        if refmods:
            raise SystemExit("no compiled workflow package for --no-sla plus --ref-mod")
        if last_base:
            raise SystemExit("no compiled workflow package for --no-sla plus --last-frame")
        if ref_audio_bases or ref_video_bases:
            raise SystemExit(
                "no compiled workflow package for --no-sla plus --ref-audio/--ref-video"
            )
        if continuation_mode not in ("none", ""):
            raise SystemExit(
                f"no compiled workflow package for --no-sla plus continuation_mode={continuation_mode}"
            )
        if init_audio_base or guide_images:
            raise SystemExit("no compiled workflow package for --no-sla plus init-audio/guides")
        if mode == "ref2va":
            if len(ref_bases) >= 2:
                return "comfy-workflow-h3-ref2va-multiref-nosla"
            if len(ref_bases) == 1:
                return "comfy-workflow-h3-ref2va-nosla"
            raise SystemExit("--no-sla Ref2VA needs at least one --ref-image")
        if first_base:
            return "comfy-workflow-h3-fl2va-nosla"
        return "comfy-workflow-h3-t2va-nosla"
    if jev:
        hq = int(steps) != 4
        suffix = "-jev-hq" if hq else "-jev"
        if vsa:
            raise SystemExit("no compiled workflow package for --jev plus --vsa")
        if sol or cache != "off" or turbo != "off" or realism:
            raise SystemExit(
                "no compiled workflow package for --jev plus turbo/sol/realism/cache"
            )
        if refmods:
            raise SystemExit("no compiled workflow package for --jev plus --ref-mod")
        if last_base:
            if not hq:
                raise SystemExit(
                    "no compiled workflow package for 4-step --jev plus --last-frame "
                    "(pass --steps 20 for HQ euler first+last)"
                )
            if ref_audio_bases or ref_video_bases:
                raise SystemExit(
                    "no compiled workflow package for --jev plus --last-frame plus --ref-audio/--ref-video"
                )
            if continuation_mode not in ("none", ""):
                raise SystemExit(
                    f"no compiled workflow package for --jev plus continuation_mode={continuation_mode}"
                )
            if init_audio_base or guide_images:
                raise SystemExit("no compiled workflow package for --jev plus init-audio/guides")
            return "comfy-workflow-h3-fl2va-last-jev"
        if ref_audio_bases or ref_video_bases:
            raise SystemExit(
                "no compiled workflow package for --jev plus --ref-audio/--ref-video"
            )
        if continuation_mode not in ("none", ""):
            raise SystemExit(
                f"no compiled workflow package for --jev plus continuation_mode={continuation_mode}"
            )
        if init_audio_base or guide_images:
            raise SystemExit("no compiled workflow package for --jev plus init-audio/guides")
        if mode == "ref2va":
            if len(ref_bases) >= 2:
                return f"comfy-workflow-h3-ref2va-multiref{suffix}"
            if len(ref_bases) == 1:
                return f"comfy-workflow-h3-ref2va{suffix}"
            raise SystemExit("009jev Ref2VA needs at least one --ref-image")
        if first_base:
            return f"comfy-workflow-h3-fl2va{suffix}"
        return f"comfy-workflow-h3-t2va{suffix}"
    if continuation_mode == "native_guide":
        if first_base:
            raise SystemExit(
                "native_guide continue has no first-frame package yet "
                "(omit the start still, or pass --legacy-masked-av)"
            )
        if ref_video_bases or ref_audio_bases:
            raise SystemExit(
                "native_guide continue has no --ref-video/--ref-audio package yet"
            )
        if len(ref_bases) > 1:
            raise SystemExit("native_guide continue supports one --ref-image (got {})".format(len(ref_bases)))
        if refmods and ref_bases:
            return "comfy-workflow-h3-continue-native-ref2va-refmod"
        if refmods:
            return "comfy-workflow-h3-continue-native-t2va-refmod"
        if ref_bases:
            return "comfy-workflow-h3-continue-native-ref2va"
        return "comfy-workflow-h3-continue-native-t2va"
    if continuation_mode == "masked_av":
        return "comfy-workflow-h3-continue" if first_base else "comfy-workflow-h3-continue-t2va"
    if continuation_mode == "vae_tail":
        return "comfy-workflow-h3-continue-vae-tail"
    if continuation_mode == "guide" or guide_images:
        return "comfy-workflow-h3-guide"
    if continuation_mode == "ganloss_two_stage":
        n_aud = len(ref_audio_bases)
        if n_aud and (vsa or ref_video_bases or len(ref_bases) != 1):
            raise SystemExit(
                "no compiled workflow package for Eros two-stage --ref-audio with "
                f"vsa={vsa} videos={len(ref_video_bases)} images={len(ref_bases)}. "
                "Still-only: one --ref-image plus 1 or 2 --ref-audio (optional --ref-mod)."
            )
        if n_aud > 2:
            raise SystemExit(
                f"no compiled workflow package for {n_aud} --ref-audio on Eros two-stage (supported: 1 or 2)"
            )
        if n_aud == 2 and refmods:
            raise SystemExit(
                "no compiled workflow package for --ref-mod plus two --ref-audio on Eros two-stage"
            )
        name = "comfy-workflow-h3-ganloss-stage1" + ("" if ref_video_bases else "-still")
        if refmods:
            if vsa:
                raise SystemExit("no compiled workflow package for --vsa plus --ref-mod")
            name += "-refmod"
        elif vsa:
            name += "-vsa"
        if n_aud == 1:
            name += "-audio"
        elif n_aud == 2:
            name += "-2audio"
        return name
    if continuation_mode == "latent_refine":
        if ref_bases or ref_video_bases:
            name = "comfy-workflow-h3-ganloss-stage2" + ("" if ref_video_bases else "-still")
            if refmods:
                if vsa:
                    raise SystemExit("no compiled workflow package for --vsa plus --ref-mod")
                return name + "-refmod"
            return name + ("-vsa" if vsa else "")
        return "comfy-workflow-h3-latent-refine"
    if continuation_mode == "mask_edit":
        return "comfy-workflow-h3-mask-edit"
    if init_audio_base:
        if len(ref_audio_bases) >= 2:
            return "comfy-workflow-h3-convert-2voice"
        return "comfy-workflow-h3-convert"
    if last_base:
        return "comfy-workflow-h3-fl2va-last"
    if turbo != "off" or sol or realism or cache != "off":
        if turbo != "off" and first_base and mode in ("t2va", "fl2va") and not sol and not realism and cache == "off":
            return "comfy-workflow-h3-fl2va-turbo"
        if turbo != "off" and mode == "t2va" and not first_base and not sol and not realism and cache == "off":
            return "comfy-workflow-h3-t2va-turbo"
        if sol and mode == "t2va" and not first_base and turbo == "off" and not realism and cache == "off":
            return "comfy-workflow-h3-t2va-sol"
        if realism and mode == "t2va" and not first_base and turbo == "off" and not sol and cache == "off":
            return "comfy-workflow-h3-t2va-realism"
        if cache == "spectrum" and mode == "t2va" and not first_base and turbo == "off" and not sol and not realism:
            return "comfy-workflow-h3-t2va-spectrum"
        if cache == "easy" and mode == "t2va" and not first_base and turbo == "off" and not sol and not realism:
            return "comfy-workflow-h3-t2va-easycache"
        if cache == "fbc" and mode == "t2va" and not first_base and turbo == "off" and not sol and not realism:
            return "comfy-workflow-h3-t2va-fbc"
        raise SystemExit(
            f"no compiled workflow package for combo mode={mode} turbo={turbo} "
            f"sol={sol} realism={realism} cache={cache} first={bool(first_base)}. "
            "Author a dedicated package instead of composing nodes in Python."
        )
    if mode == "ref2va":
        if ref_audio_bases:
            if vsa or refmods or ref_video_bases or len(ref_bases) != 1:
                raise SystemExit(
                    "no compiled workflow package for --ref-audio with "
                    f"vsa={vsa} refmods={bool(refmods)} videos={len(ref_video_bases)} "
                    f"images={len(ref_bases)}. Stock path is one still plus 1 or 2 --ref-audio."
                )
            n_aud = len(ref_audio_bases)
            if n_aud == 1:
                return "comfy-workflow-h3-ref2va-audio"
            if n_aud == 2:
                return "comfy-workflow-h3-ref2va-2audio"
            raise SystemExit(
                f"no compiled workflow package for {n_aud} --ref-audio (supported: 1 or 2)"
            )
        if refmods and not ref_bases and not ref_video_bases:
            if vsa:
                raise SystemExit("no compiled workflow package for --vsa plus --ref-mod without a still")
            return "comfy-workflow-h3-t2va-refmod"
        if ref_video_bases and not ref_bases:
            name = "comfy-workflow-h3-ref2va-video"
        elif len(ref_bases) >= 2:
            name = "comfy-workflow-h3-ref2va-multiref"
        else:
            name = "comfy-workflow-h3-ref2va"
        if refmods:
            if vsa or name != "comfy-workflow-h3-ref2va":
                raise SystemExit(
                    f"no compiled workflow package for combo {name} plus --ref-mod"
                )
            return "comfy-workflow-h3-ref2va-refmod"
        return name + ("-vsa" if vsa else "")
    if refmods:
        if first_base or last_base or vsa or turbo != "off" or sol or realism or cache != "off":
            raise SystemExit(
                "no compiled workflow package for --ref-mod with turbo/sol/realism/cache/I2V stills"
            )
        return "comfy-workflow-h3-t2va-refmod"
    if vsa:
        raise SystemExit(
            f"no compiled workflow package for combo mode={mode} vsa=True "
            f"(VSA is Ref2VA only; continuation_mode={continuation_mode})."
        )
    if first_base:
        return "comfy-workflow-h3-fl2va"
    return "comfy-workflow-h3-t2va"


def _newest_named(output_dir: Path, token: str, t_after: float) -> Path:
    hits: list[Path] = []
    if output_dir.is_dir():
        for path in output_dir.rglob("*"):
            # JSON sidecars share the stem and are written after the tensor file.
            if path.suffix.lower() != ".safetensors":
                continue
            if not path.is_file() or token not in path.name:
                continue
            try:
                if path.stat().st_mtime + 0.5 < t_after:
                    continue
            except OSError:
                continue
            hits.append(path)
    if not hits:
        raise SystemExit(f"singularity stage did not write {token} under {output_dir}")
    hits.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return hits[0]


def _comfy_python(comfy_root: Path, python: Path) -> Path:
    for cand in (
        comfy_root.parent / ".venv" / "Scripts" / "python.exe",
        comfy_root.parent / ".venv" / "bin" / "python",
        python,
    ):
        if cand.is_file():
            return cand
    return python


def _run_one_graph(
    *,
    package: str,
    params: dict[str, Any],
    comfy_root: Path,
    python: Path,
    graph_path: Path,
    timing_path: Path,
    attn: str,
    require_accel: bool,
) -> tuple[Path, float]:
    """One Comfy process. Serve mode is ignored so weights die with the process."""
    from h3_comfy_common import materialize_workflow, run_comfy_graph

    graph = materialize_workflow(package, params)
    saved = os.environ.pop("GEMMY_H3_COMFY_SERVE", None)
    try:
        outputs_dir, _note, elapsed = run_comfy_graph(
            comfy_root=comfy_root,
            python=python,
            graph=graph,
            graph_path=graph_path,
            timing_path=timing_path,
            attn=attn if attn != "none" else "sage",
            require_accel=require_accel,
        )
    finally:
        if saved is not None:
            os.environ["GEMMY_H3_COMFY_SERVE"] = saved
    return outputs_dir, elapsed


def _run_singularity_split(
    *,
    req: dict[str, Any],
    params: dict[str, Any],
    comfy_root: Path,
    python: Path,
    work: Path,
    output: Path,
    prefix: str,
    persist_latent: bool,
    attn: str,
    stages: dict[str, float],
    t0: float,
    mode: str,
    quality: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    turbo: str,
    turbo_strength: float,
    realism: bool,
    realism_strength: float,
    sol: bool,
    cache: str,
    dit_quant: str,
) -> int:
    """Encode, turbo sample, final 10 steps, decode. Each stage is its own process."""
    from h3_comfy_common import copy_sidecar_next_to

    comfy_python = _comfy_python(comfy_root, python)
    shared = {
        "prompt": params["prompt"],
        "width": params["width"],
        "height": params["height"],
        "length": params["length"],
        "seed": params["seed"],
        "unet": params.get("unet") or "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors",
        "video_vae": params.get("video_vae") or "minimax_h3_video_vae_fp16.safetensors",
        "ref_image": params["ref_image"],
        "fingerprint": params.get("fingerprint") or "{}",
    }
    print(
        "[h3-comfy] singularity split: encode exits, then UNET turbo, then a new UNET for the last 10 steps, then VAE decode",
        flush=True,
    )

    print("[h3-comfy] singularity encode: text encoder and VAEs, no UNET", flush=True)
    t_stage = time.perf_counter()
    encode_dir, elapsed = _run_one_graph(
        package="comfy-workflow-h3-singularity-encode",
        params={
            **shared,
            "cond_prefix": f"{prefix}_cond",
            "latent_prefix": f"{prefix}_empty",
        },
        comfy_root=comfy_root,
        python=comfy_python,
        graph_path=work / "singularity_encode.json",
        timing_path=work / "singularity_encode_timing.json",
        attn=attn,
        require_accel=False,
    )
    stages["singularity_encode_s"] = elapsed
    cond_path = _newest_named(encode_dir, "_h3cond.safetensors", t_stage)
    empty_path = _newest_named(encode_dir, "_empty_", t_stage)
    print(f"[h3-comfy] singularity encode wrote {cond_path.name} and {empty_path.name}", flush=True)

    print("[h3-comfy] singularity turbo: UNET only, 2 denoise steps, then exit", flush=True)
    t_stage = time.perf_counter()
    turbo_dir, elapsed = _run_one_graph(
        package="comfy-workflow-h3-singularity-turbo",
        params={
            **shared,
            "conditioning_path": str(cond_path),
            "latent_path": str(empty_path),
            "latent_prefix": f"{prefix}_turbo",
            "sigmas_prefix": f"{prefix}_sig",
        },
        comfy_root=comfy_root,
        python=comfy_python,
        graph_path=work / "singularity_turbo.json",
        timing_path=work / "singularity_turbo_timing.json",
        attn=attn,
        require_accel=True,
    )
    stages["singularity_turbo_s"] = elapsed
    turbo_latent = _newest_named(turbo_dir, "_turbo_", t_stage)
    sigmas_path = _newest_named(turbo_dir, "_h3sig.safetensors", t_stage)
    print(f"[h3-comfy] singularity turbo wrote {turbo_latent.name} and {sigmas_path.name}", flush=True)

    print("[h3-comfy] singularity final: new UNET process, last 10 denoise steps, then exit", flush=True)
    t_stage = time.perf_counter()
    final_dir, elapsed = _run_one_graph(
        package="comfy-workflow-h3-singularity-final",
        params={
            **shared,
            "conditioning_path": str(cond_path),
            "latent_path": str(turbo_latent),
            "sigmas_path": str(sigmas_path),
            "latent_prefix": f"{prefix}_final",
        },
        comfy_root=comfy_root,
        python=comfy_python,
        graph_path=work / "singularity_final.json",
        timing_path=work / "singularity_final_timing.json",
        attn=attn,
        require_accel=True,
    )
    stages["singularity_final_s"] = elapsed
    final_latent = _newest_named(final_dir, "_final_", t_stage)
    print(f"[h3-comfy] singularity final wrote {final_latent.name}", flush=True)

    print("[h3-comfy] singularity decode: VAEs only", flush=True)
    t_stage = time.perf_counter()
    decode_out, elapsed = _run_one_graph(
        package="comfy-workflow-h3-singularity-decode",
        params={
            **shared,
            "latent_path": str(final_latent),
            "output_prefix": prefix,
        },
        comfy_root=comfy_root,
        python=comfy_python,
        graph_path=work / "singularity_decode.json",
        timing_path=work / "singularity_decode_timing.json",
        attn=attn,
        require_accel=False,
    )
    stages["singularity_decode_s"] = elapsed
    mp4 = _find_latest_mp4(decode_out, prefix, t_stage - 1.0)
    if mp4 is None:
        raise SystemExit(f"no mp4 found under {decode_out} for prefix={prefix}")

    decode_dir = work / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    dest_mp4 = decode_dir / "output.mp4"
    shutil.copy2(mp4, dest_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest_mp4, output)

    sidecar = None
    if persist_latent:
        sidecar = copy_sidecar_next_to(output, final_latent)
        if sidecar is None:
            print("[h3-comfy] warn: AV latent sidecar was not produced", flush=True)
        else:
            print(f"[h3-comfy] latent sidecar: {sidecar}", flush=True)

    wall = time.perf_counter() - t0
    result = {
        "ok": True,
        "engine": "comfy",
        "output": str(output),
        "av_latent": str(sidecar) if sidecar else None,
        "work_dir": str(work),
        "mode": mode,
        "quality": quality,
        "attn": attn,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
        "turbo": turbo,
        "turbo_strength": turbo_strength,
        "realism": realism,
        "realism_strength": realism_strength,
        "sol": sol,
        "cache": cache,
        "dit_quant": dit_quant,
        "comfy_root": str(comfy_root),
        "comfy_mp4": str(mp4),
        "continuation_mode": "singularity_dual",
        "singularity_processes": ["encode", "turbo", "final", "decode"],
        "stages_s": stages,
        "wall_s": wall,
    }
    _write_json(work / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[h3-comfy] wrote {output} in {wall:.1f}s", flush=True)
    return 0


def _find_latest_mp4(output_dir: Path, prefix: str, t_after: float) -> Path | None:
    """Find SaveVideo output matching filename_prefix written after t_after."""
    # SaveVideo writes prefix_00001_.mp4 under output_dir (subdirs allowed).
    prefix = prefix.replace("\\", "/").strip("/")
    candidates: list[Path] = []
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
        # fallback: newest mp4 after t_after
        all_new = []
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


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    ap = argparse.ArgumentParser(description="Gemmy MiniMax-H3 Comfy generate worker")
    ap.add_argument("--request", type=Path, required=True)
    args = ap.parse_args(argv)

    req = json.loads(args.request.read_text(encoding="utf-8"))
    h3_root = Path(req["h3_root"]).resolve()
    python = Path(req["python"]).resolve()
    work = Path(req["work_dir"]).resolve()
    output = Path(req["output"]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    if not h3_root.is_dir():
        raise SystemExit(f"h3_root missing: {h3_root}")
    if not python.is_file():
        raise SystemExit(f"python missing: {python}")

    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit("prompt is required")

    mode = str(req.get("mode") or "t2va").lower()
    quality = str(req.get("quality") or "fast").lower()
    attn = str(req.get("attn") or "").lower()
    if not attn:
        attn = "fa2" if quality in ("hq", "fa2", "high") else "sage"
    width = int(req.get("width") or 864)
    height = int(req.get("height") or 480)
    frames = int(req.get("frames") or 124)
    steps = int(req.get("steps") or 20)
    seed = int(req.get("seed") or 42)
    shift_video = float(req["shift_video"] if req.get("shift_video") is not None else 12.0)
    shift_audio = float(req["shift_audio"] if req.get("shift_audio") is not None else 3.0)
    print(
        f"[h3-comfy] mode={mode} {width}x{height} frames={frames} steps={steps} "
        f"shift={shift_video}/{shift_audio} cache={str(req.get('cache') or 'off')} "
        f"denoise={req.get('denoise', 1.0)} init_audio={bool(req.get('init_audio'))}",
        flush=True,
    )
    compile_ir = bool(req.get("compile_ir", True))
    first_image = req.get("first_image")
    last_image = req.get("last_image")
    refs_raw = req.get("refs") or []
    turbo = str(req.get("turbo") or "off").lower()
    turbo_strength = float(req.get("turbo_strength") if req.get("turbo_strength") is not None else 1.0)
    realism = bool(req.get("realism", False))
    realism_strength = float(
        req.get("realism_strength") if req.get("realism_strength") is not None else 1.0
    )
    sol = bool(req.get("sol", False))
    vsa = bool(req.get("vsa", False))
    vsa_sparsity = float(req.get("vsa_sparsity") if req.get("vsa_sparsity") is not None else 0.75)
    vsa_gate = str(req.get("vsa_gate") or "fasth3_vsa_gate.safetensors")
    jev = bool(req.get("jev", False))
    jev_sdk_python = str(req.get("jev_sdk_python") or "").strip()
    jev_initial_policy = str(req.get("jev_initial_policy") or "").strip()
    sla_fixed_raw = req.get("sla_fixed")
    sla_fixed = None if sla_fixed_raw in (None, "", False) else int(sla_fixed_raw)
    keep_table = load_keep_table(req)
    if keep_table is not None and len(keep_table) != steps:
        raise SystemExit(
            f"keep_table has {len(keep_table)} steps but request steps={steps}"
        )
    if keep_table is not None:
        if sla_fixed is not None:
            raise SystemExit("cannot combine sla_table / keep_table with --sla-fixed")
        if jev_initial_policy == "jev_first":
            raise SystemExit("cannot combine adaptive --jev (jev_first) with sla_table")
        jev = True
        jev_initial_policy = "table"
    elif sla_fixed is not None:
        if sla_fixed not in (1, 3, 5, 10):
            raise SystemExit(f"sla_fixed must be 1, 3, 5, or 10 (got {sla_fixed})")
        if jev_initial_policy == "jev_first":
            raise SystemExit("cannot combine adaptive --jev (jev_first) with --sla-fixed")
        jev = True
        jev_initial_policy = f"const{sla_fixed}"
    elif not jev_initial_policy:
        jev_initial_policy = "jev_first"
    jev_log_dataset = str(req.get("jev_log_dataset") or "").strip()
    no_sla = bool(req.get("no_sla", False))
    if no_sla and (jev or sla_fixed is not None or keep_table is not None):
        raise SystemExit("cannot combine --no-sla with --jev, --sla-fixed, or --sla-table")
    cache = str(req.get("cache") or "off").lower()
    dit_quant = str(req.get("dit_quant") or "int8").lower()

    if cache not in ("off", "spectrum", "easy", "fbc"):
        raise SystemExit(f"invalid cache={cache}")
    if turbo not in ("off", "v4", "v1", "ema_wan"):
        raise SystemExit(f"invalid turbo={turbo}")
    if realism and not (0.0 <= realism_strength <= 2.0):
        raise SystemExit(f"realism_strength out of range: {realism_strength}")

    # Turbo implies step count unless caller already set a turbo-range steps.
    if turbo != "off" and int(req.get("steps") or 0) <= 0:
        steps = 4

    comfy_root = _resolve_comfy_root(req, h3_root)
    print(f"[h3-comfy] comfy_root={comfy_root}", flush=True)

    t0 = time.perf_counter()
    stages: dict[str, float] = {}

    # --- optional IR compile (same quality arm as python worker) ---
    ir_mode = "t2va"
    if mode in ("i2v", "fl2va", "i2va"):
        ir_mode = "fl2va"
    elif mode == "ref2va":
        ir_mode = "ref2va"

    text_for_graph = prompt
    # fal card: start the prompt with trigger `r34l1sm` when Realism People LoRA is on.
    if realism:
        trig = "r34l1sm"
        low = text_for_graph.lstrip().lower()
        if not low.startswith(trig):
            text_for_graph = f"{trig} {text_for_graph.lstrip()}"
            print(f"[h3-comfy] realism: prepended trigger {trig!r}", flush=True)
    if compile_ir:
        t = time.perf_counter()
        ir_out = work / "prompt_ir.txt"
        ir_script = h3_root / "scripts" / "h3_prompt_ir.py"
        if ir_script.is_file():
            cmd = [
                str(python),
                str(ir_script),
                text_for_graph,
                "--mode",
                ir_mode,
                "--duration",
                f"{frames / 24.0:.4f}",
                "-o",
                str(ir_out),
            ]
            _run(cmd, cwd=h3_root, label="prompt_ir")
            text_for_graph = ir_out.read_text(encoding="utf-8")
            # Keep trigger at the front after IR compile if the compiler dropped it.
            if realism:
                trig = "r34l1sm"
                low = text_for_graph.lstrip().lower()
                if not low.startswith(trig):
                    text_for_graph = f"{trig}\n{text_for_graph.lstrip()}"
        else:
            print("[h3-comfy] warn: h3_prompt_ir.py missing; using raw prompt", flush=True)
        stages["prompt_ir_s"] = time.perf_counter() - t
    (work / "prompt.txt").write_text(text_for_graph, encoding="utf-8")

    # --- stage media into Comfy input/ ---
    comfy_input = comfy_root / "input"
    job_tag = work.name.replace(" ", "_")[:48]
    first_base = None
    last_base = None
    ref_bases: list[str] = []

    if first_image:
        src = Path(first_image)
        if not src.is_file():
            raise SystemExit(f"first_image missing: {src}")
        ext = src.suffix.lower() or ".png"
        first_base = _stage_image(src, comfy_input, f"gemmy_{job_tag}_first{ext}")
    if last_image:
        src = Path(last_image)
        if not src.is_file():
            raise SystemExit(f"last_image missing: {src}")
        ext = src.suffix.lower() or ".png"
        last_base = _stage_image(src, comfy_input, f"gemmy_{job_tag}_last{ext}")

    ref_video_bases: list[str] = []
    ref_video_audio_bases: list[str | None] = []
    ref_audio_bases: list[str] = []
    for i, ref in enumerate(refs_raw):
        kind = str(ref.get("kind") or "image").lower()
        if kind == "image":
            src = Path(ref["path"])
            if not src.is_file():
                raise SystemExit(f"ref image missing: {src}")
            ext = src.suffix.lower() or ".png"
            ref_bases.append(_stage_image(src, comfy_input, f"gemmy_{job_tag}_ref{i}{ext}"))
        elif kind == "video":
            src = Path(ref["path"])
            if not src.is_file():
                raise SystemExit(f"ref video missing: {src}")
            ext = src.suffix.lower() or ".mp4"
            ref_video_bases.append(_stage_image(src, comfy_input, f"gemmy_{job_tag}_refv{i}{ext}"))
            ref_video_audio_bases.append(None)
        elif kind == "audio":
            src = Path(ref["path"])
            if not src.is_file():
                raise SystemExit(f"ref audio missing: {src}")
            ext = src.suffix.lower() or ".wav"
            ref_audio_bases.append(_stage_image(src, comfy_input, f"gemmy_{job_tag}_refa{i}{ext}"))
        elif kind == "av":
            vsrc = Path(ref["video"])
            asrc = Path(ref["audio"])
            if not vsrc.is_file():
                raise SystemExit(f"ref av video missing: {vsrc}")
            if not asrc.is_file():
                raise SystemExit(f"ref av audio missing: {asrc}")
            vext = vsrc.suffix.lower() or ".mp4"
            aext = asrc.suffix.lower() or ".wav"
            ref_video_bases.append(_stage_image(vsrc, comfy_input, f"gemmy_{job_tag}_refav{i}{vext}"))
            ref_video_audio_bases.append(
                _stage_image(asrc, comfy_input, f"gemmy_{job_tag}_refava{i}{aext}")
            )
        else:
            raise SystemExit(f"unknown ref kind={kind}")

    graph_mode = mode
    if mode == "i2v":
        graph_mode = "fl2va"
    refmods_raw = req.get("refmods") or []
    if mode == "ref2va" and not ref_bases and not ref_video_bases and not refmods_raw:
        raise SystemExit("ref2va requires image or video refs, or saved --ref-mod")

    persist_latent = bool(req.get("persist_av_latent", True))
    continuation_mode = str(req.get("continuation_mode") or "none").lower()
    context_latent = req.get("context_latent")
    context_video_base = None
    if req.get("context_video"):
        src = Path(req["context_video"])
        if not src.is_file():
            raise SystemExit(f"context_video missing: {src}")
        ext = src.suffix.lower() or ".mp4"
        context_video_base = _stage_image(src, comfy_input, f"gemmy_{job_tag}_ctx{ext}")
    context_frames = int(req.get("context_frames") or 39)
    audio_mode = str(req.get("audio_mode") or "generated_audio")
    denoise = float(req["denoise"] if req.get("denoise") is not None else 1.0)
    if not 0.0 < denoise <= 1.0:
        raise SystemExit(f"denoise must be in (0, 1], got {denoise}")
    init_audio_base = None
    if req.get("init_audio"):
        src = Path(req["init_audio"])
        if not src.is_file():
            raise SystemExit(f"init_audio missing: {src}")
        ext = src.suffix.lower() or ".wav"
        init_audio_base = _stage_image(src, comfy_input, f"gemmy_{job_tag}_inita{ext}")
    trim_prefix = bool(req.get("trim_prefix", False))
    guide_images = []
    for i, gimg in enumerate(req.get("guide_images") or []):
        src = Path(gimg["path"] if isinstance(gimg, dict) else gimg)
        if not src.is_file():
            raise SystemExit(f"guide image missing: {src}")
        ext = src.suffix.lower() or ".png"
        guide_images.append(
            {
                "image": _stage_image(src, comfy_input, f"gemmy_{job_tag}_guide{i}{ext}"),
                "frame_idx": int(gimg.get("frame_idx") or 0) if isinstance(gimg, dict) else 0,
            }
        )

    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(h3_root))
    from h3_comfy_common import (  # type: ignore
        bind_video_vae,
        copy_sidecar_next_to,
        find_latest,
        fingerprint_payload,
        materialize_workflow,
        run_comfy_graph,
        video_vae_filename,
        video_vae_fingerprint_id,
    )

    prefix = f"gemmy/{job_tag}"
    vae_name = video_vae_filename(req)
    fp = fingerprint_payload(
        width=width,
        height=height,
        frames=frames,
        dit=str(req.get("weights") or ""),
        lora=(
            (["h3-realism-people-t2v-i2v-r2v"] if realism else [])
            + ([f"turbo-{turbo}"] if turbo != "off" else [])
        ),
        mode=mode,
        context_frames=context_frames,
        vae=video_vae_fingerprint_id(vae_name),
    )
    weights_name = Path(str(req.get("weights") or "")).name
    unet_name = weights_name if weights_name.endswith(".safetensors") else None
    packaged = _packaged_generate_name(
        graph_mode=graph_mode if graph_mode != "t2va" else "t2va",
        turbo=turbo,
        continuation_mode=continuation_mode,
        first_base=first_base,
        last_base=last_base,
        ref_bases=ref_bases,
        ref_video_bases=ref_video_bases,
        ref_audio_bases=ref_audio_bases,
        guide_images=guide_images,
        init_audio_base=init_audio_base,
        persist_latent=persist_latent,
        sol=sol,
        vsa=vsa,
        cache=cache,
        realism=realism,
        refmods=list(refmods_raw),
        jev=jev,
        no_sla=no_sla,
        steps=steps,
    )
    params: dict[str, Any] = {
        "prompt": text_for_graph,
        "width": width,
        "height": height,
        "length": frames,
        "seed": seed,
        "steps": steps,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "denoise": denoise,
        "output_prefix": prefix,
        "latent_prefix": f"{prefix}_av",
        "fingerprint": json.dumps(fp),
    }
    bind_video_vae(params, req)
    if unet_name:
        params["unet"] = unet_name
    if vsa:
        params["gate_file"] = vsa_gate
        params["sparsity"] = vsa_sparsity
    if jev:
        const_policies = {"const1", "const3", "const5", "const10"}
        if jev_initial_policy in const_policies:
            params["sdk_python"] = jev_sdk_python
            params["initial_policy"] = jev_initial_policy
        elif jev_initial_policy == "table":
            if keep_table is None:
                raise SystemExit("initial_policy=table requires sla_table or keep_table")
            params["sdk_python"] = jev_sdk_python or ""
            params["initial_policy"] = "table"
        elif jev_initial_policy == "jev_first":
            if not os.environ.get("TYPESAFE_API_KEY", "").strip():
                raise SystemExit(
                    "009jev requires TYPESAFE_API_KEY in the Comfy process environment "
                    "(env or config env_overrides). Default generate does not. "
                    "--sla-fixed and --sla-table do not need this key."
                )
            if not jev_sdk_python or not Path(jev_sdk_python).is_file():
                raise SystemExit(
                    "009jev needs the dedicated SDK Python (typesafe-sdk==0.7.0): "
                    f"{jev_sdk_python or 'runtimes/minimax-h3/jev-sdk/.venv/Scripts/python.exe'}"
                )
            params["sdk_python"] = jev_sdk_python
            params["initial_policy"] = jev_initial_policy
        else:
            raise SystemExit(
                "009jev product path is initial_policy=jev_first, const1|3|5|10, or table "
                f"(got {jev_initial_policy!r})"
            )
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
    if first_base:
        params["first_image"] = first_base
    if last_base:
        params["last_image"] = last_base
    if ref_bases:
        params["ref_image"] = ref_bases[0]
        if len(ref_bases) > 1:
            params["ref_image_1"] = ref_bases[1]
    if ref_video_bases:
        params["ref_video"] = ref_video_bases[0]
        params["crop_video"] = ref_video_bases[0]
    if ref_audio_bases:
        params["reference_audio"] = ref_audio_bases[0]
        if len(ref_audio_bases) > 1:
            params["reference_audio_2"] = ref_audio_bases[1]
    if init_audio_base:
        params["audio_1"] = init_audio_base
    if context_latent:
        params["context_latent"] = str(context_latent)
        params["context_frames"] = context_frames
    if context_video_base:
        params["context_video"] = context_video_base
        params["context_frames"] = context_frames
    if guide_images:
        params["guide_image"] = guide_images[0]["image"]
    if packaged == "singularity-split":
        return _run_singularity_split(
            req=req,
            params=params,
            comfy_root=comfy_root,
            python=python,
            work=work,
            output=output,
            prefix=prefix,
            persist_latent=persist_latent,
            attn=attn,
            stages=stages,
            t0=t0,
            mode=mode,
            quality=quality,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            turbo=turbo,
            turbo_strength=turbo_strength,
            realism=realism,
            realism_strength=realism_strength,
            sol=sol,
            cache=cache,
            dit_quant=dit_quant,
        )
    print(f"[h3-comfy] packaged workflow={packaged}", flush=True)
    graph = materialize_workflow(packaged, params)
    if keep_table is not None:
        inject_keep_table(graph, keep_table)
        print("[h3-comfy] sla-table: per-(step,layer) keep (no Jev worker)", flush=True)
    graph_path = work / "comfy_workflow.json"
    _write_json(graph_path, graph)

    # Prefer research venv python next to Comfy (torch/sage) when request python
    # is the gemmy runtime — fall back to request python.
    comfy_py_candidates = [
        comfy_root.parent / ".venv" / "Scripts" / "python.exe",
        comfy_root.parent / ".venv" / "bin" / "python",
        python,
    ]
    comfy_python = python
    for cand in comfy_py_candidates:
        if cand.is_file():
            comfy_python = cand
            break

    timing_path = work / "comfy_timing.json"
    if jev:
        events_path = work / "jev_events.jsonl"
        os.environ["GEMMY_JEV_EVENTS_PATH"] = str(events_path)
        if jev_initial_policy == "jev_first":
            os.environ["GEMMY_JEV_DUMP_DIR"] = str(work)
    t_exec0 = time.perf_counter()
    outputs_dir, _note, elapsed = run_comfy_graph(
        comfy_root=comfy_root,
        python=comfy_python,
        graph=graph,
        graph_path=graph_path,
        timing_path=timing_path,
        attn=attn if attn != "none" else "sage",
        require_accel=turbo != "off" or sol or cache != "off" or realism,
    )
    stages["comfy_execute_s"] = elapsed

    mp4 = _find_latest_mp4(outputs_dir, prefix, t_exec0 - 1.0)
    if mp4 is None:
        raise SystemExit(f"no mp4 found under {outputs_dir} for prefix={prefix}")

    decode_dir = work / "decode"
    decode_dir.mkdir(parents=True, exist_ok=True)
    dest_mp4 = decode_dir / "output.mp4"
    shutil.copy2(mp4, dest_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest_mp4, output)

    sidecar = None
    if persist_latent:
        produced_av = find_latest(outputs_dir, prefix, t_exec0 - 1.0, ".safetensors")
        if produced_av is None:
            produced_av = find_latest(outputs_dir, prefix + "_av", t_exec0 - 1.0, ".safetensors")
        sidecar = copy_sidecar_next_to(output, produced_av)
        if sidecar is None:
            print("[h3-comfy] warn: AV latent sidecar was not produced", flush=True)
        else:
            print(f"[h3-comfy] latent sidecar: {sidecar}", flush=True)

    wall = time.perf_counter() - t0
    result = {
        "ok": True,
        "engine": "comfy",
        "output": str(output),
        "av_latent": str(sidecar) if sidecar else None,
        "work_dir": str(work),
        "mode": mode,
        "quality": quality,
        "attn": attn,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
        "turbo": turbo,
        "turbo_strength": turbo_strength,
        "realism": realism,
        "realism_strength": realism_strength,
        "sol": sol,
        "cache": cache,
        "dit_quant": dit_quant,
        "comfy_root": str(comfy_root),
        "comfy_mp4": str(mp4),
        "stages_s": stages,
        "wall_s": wall,
    }
    if timing_path.is_file():
        try:
            result["comfy_timing"] = json.loads(timing_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    _write_json(work / "result.json", result)
    if jev and jev_log_dataset:
        try:
            from jev_teacher import append_from_run, parse_009jev_log, load_events

            events_file = work / "jev_events.jsonl"
            events = []
            if events_file.is_file():
                events = load_events(events_file)
            if not events:
                log_guess = Path(str(req.get("generate_log") or ""))
                if log_guess.is_file():
                    events = parse_009jev_log(log_guess.read_text(encoding="utf-8", errors="replace"))
            if events:
                summary = append_from_run(
                    dataset_dir=Path(jev_log_dataset),
                    events=events,
                    generation={
                        "generation_id": Path(output).stem,
                        "prompt": text_for_graph,
                        "seed": seed,
                        "first_frame": first_image,
                        "width": width,
                        "height": height,
                        "duration": frames / 24.0,
                        "steps": steps,
                        "model": str(req.get("weights") or ""),
                        "sampler": "res_multistep" if int(steps) == 4 else "euler",
                        "model_family": str(req.get("model_family") or mode),
                        "num_steps": int(steps),
                        "compile_ir": compile_ir,
                        "controller": "jev" if jev_initial_policy == "jev_first" else jev_initial_policy,
                        "controller_model": "jev-1.13.0" if jev_initial_policy == "jev_first" else None,
                        "total_wall_seconds": wall,
                        "sampler_seconds": None,
                        "controller_seconds": None,
                        "peak_vram_mb": None,
                        "output_video": str(output),
                        "quality_tag": str(req.get("quality_tag") or "clean"),
                    },
                )
                result["jev_dataset"] = summary
                print(f"[h3-comfy] jev teacher dataset += {summary['layer_rows']} rows", flush=True)
            else:
                print("[h3-comfy] warn: --jev-log-dataset set but no [009jev] events found", flush=True)
        except Exception as exc:
            print(f"[h3-comfy] warn: jev teacher dataset append failed: {exc}", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"[h3-comfy] wrote {output} in {wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
