"""Headless MiniMax-H3 workflow runner (Gemmy internalized Comfy engine).

Lives under ``runtimes/minimax-h3/ComfyUI`` — production code root is Gemmy, not
any external research checkout.

Weights are **external** multi-GB checkpoints. Resolution order for the
checkpoints base path written into a runtime extra_model_paths config:

  1. ``GEMMY_H3_CHECKPOINTS``
  2. ``extra_model_paths.yaml`` base_path when it exists on disk
  3. fail loudly (never silently fall back to a hard-coded research tree)

Comfy only parses sys.argv when comfy.options.enable_args_parsing() is called
*before* the first import of comfy.cli_args (main.py does this; we must too).

Loads builtin extras **and** pinned custom_nodes (Turbo / Sol / Spectrum / KJ / FBC).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import tempfile
import time
import uuid
from pathlib import Path

COMFY_ROOT = Path(__file__).resolve().parent
_BOOTED = False
_WEIGHT_CACHE: dict = {}
_WEIGHT_CACHE_INSTALLED = False



def _resolve_checkpoints_base() -> Path:
    """External weights root only — never vendor multi-GB shards into the runtime."""
    env = (os.environ.get("GEMMY_H3_CHECKPOINTS") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise SystemExit(f"GEMMY_H3_CHECKPOINTS is not a directory: {p}")

    yaml_path = COMFY_ROOT / "extra_model_paths.yaml"
    if yaml_path.is_file():
        try:
            ytext = yaml_path.read_text(encoding="utf-8")
            for line in ytext.splitlines():
                s = line.strip()
                if s.startswith("base_path:"):
                    raw = s.split(":", 1)[1].strip().strip("'\"")
                    if raw and not raw.startswith("${"):
                        cand = Path(raw).expanduser().resolve()
                        if cand.is_dir():
                            return cand
        except OSError:
            pass

    raise SystemExit(
        "H3 checkpoints root not found. "
        "Set GEMMY_H3_CHECKPOINTS to the external weights tree "
        "(diffusion_models / text_encoders / vae / loras). "
        "Code is internalized under runtimes/minimax-h3/ComfyUI; weights stay external."
    )


def _dit_search_root(raw: str, default: str | None = None) -> Path | None:
    """Resolve a DiT extra-root from env or a default; files map to their tree root."""
    value = (raw or "").strip() or (default or "")
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        path = path.parent
        if path.name == "diffusion_models":
            path = path.parent
    if path.is_dir():
        return path.resolve()
    return None


def _append_dit_search_root(lines: list[str], name: str, root: Path) -> None:
    base = str(root)
    if not base.endswith(os.sep):
        base = base + os.sep
    lines.extend(
        [
            f"{name}:",
            f"    base_path: {base}",
            "    diffusion_models: diffusion_models/",
            "",
        ]
    )


def _write_runtime_extra_model_paths(checkpoints: Path) -> Path:
    """Emit a temp extra_model_paths.yaml pointing at the resolved checkpoints root."""
    base = str(checkpoints)
    if not base.endswith(chr(92)) and not base.endswith("/"):
        base = base + os.sep
    lines = [
        "minimax_h3_local:",
        f"    base_path: {base}",
        "    is_default: true",
        "    diffusion_models: diffusion_models/",
        "    text_encoders: text_encoders/",
        "    vae: vae/",
        "    loras: loras/",
        "    refmods: refmods/",
        "    latent_upscale_models: latent_upscale_models/",
        "    ultralytics: ultralytics/",
        "    ultralytics_bbox: ultralytics/bbox/",
        "    checkpoints: checkpoints/",
        "",
    ]
    refmods = os.environ.get("GEMMY_H3_REFMODS", "").strip()
    if refmods:
        rbase = refmods if refmods.endswith(("/", "\\")) else refmods + os.sep
        lines = [
            "minimax_h3_refmods:",
            f"    base_path: {rbase}",
            "    refmods: ./",
            "",
        ] + lines
    eros = _dit_search_root(
        os.environ.get("GEMMY_H3_EROS_CHECKPOINTS", ""),
        r"F:\Models\minimax-h3-eros",
    )
    if eros is not None:
        _append_dit_search_root(lines, "minimax_h3_eros", eros)
    stock = _dit_search_root(
        os.environ.get("GEMMY_H3_REF2VA_STOCK", ""),
        r"G:\Models\minimax-h3-backup",
    )
    if stock is not None:
        _append_dit_search_root(lines, "minimax_h3_ref2va_stock", stock)
    singularity = _dit_search_root(
        os.environ.get("GEMMY_H3_SINGULARITY_CHECKPOINTS", ""),
        r"F:\Models\minimax-h3-singularity",
    )
    if singularity is not None:
        _append_dit_search_root(lines, "minimax_h3_singularity", singularity)
    hybrid = _dit_search_root(
        os.environ.get("GEMMY_H3_HYBRID_CHECKPOINTS", ""),
        r"C:\Models\minimax-h3-hybrid",
    )
    if hybrid is not None:
        base_h = str(hybrid)
        if not base_h.endswith(os.sep):
            base_h = base_h + os.sep
        lines.extend(
            [
                "minimax_h3_hybrid:",
                f"    base_path: {base_h}",
                "    diffusion_models: diffusion_models/",
                "    loras: loras/",
                "    latent_upscale_models: latent_upscale_models/",
                "",
            ]
        )
    sam3 = _dit_search_root(
        os.environ.get("GEMMY_SAM3_CHECKPOINTS", ""),
        r"F:\Models\sam3",
    )
    if sam3 is not None:
        base_s = str(sam3)
        if not base_s.endswith(os.sep):
            base_s = base_s + os.sep
        lines.extend(
            [
                "sam3:",
                f"    base_path: {base_s}",
                "    checkpoints: checkpoints/",
                "",
            ]
        )
    fd, name = tempfile.mkstemp(prefix="gemmy_h3_extra_paths_", suffix=".yaml")
    os.close(fd)
    path = Path(name)
    path.write_text(chr(10).join(lines), encoding="utf-8")
    return path


# Always needed for core H3 graphs.
CORE_NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "KSampler",
    "VAEDecode",
    "ConditioningZeroOut",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3AddGuide",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
]

# Acceleration packs (present after Phase 0 pin install).
ACCEL_NODES = [
    "MiniMaxH3TurboLoRA",
    "MiniMaxH3TurboSampler",
    "MiniMaxH3ScheduledSolAttentionPatch",
    "MiniMaxH3FusedModulation",
    "MiniMaxH3ChunkFeedForward",
    "SpectrumApplyMiniMaxH3",
    "PathchSageAttentionKJ",
    "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "ApplyMiniMaxH3FirstBlockCache",
    "EasyCache",
    "RandomNoise",
    "BasicGuider",
    "BasicScheduler",
    "SamplerCustomAdvanced",
]


def _live_preview_path() -> Path | None:
    raw = (os.environ.get("GEMMY_LIVE_PREVIEW") or "").strip()
    return Path(raw) if raw else None


def _preview_to_pil(preview: object):
    if preview is None:
        return None
    img = preview[1] if isinstance(preview, (tuple, list)) and len(preview) >= 2 else preview
    try:
        from PIL import Image

        if isinstance(img, Image.Image):
            return img
    except Exception:
        return None
    return None


def _install_live_preview_hook(comfy_utils, dest: Path) -> None:
    """Write Latent2RGB JPEGs for Nightshift. Not a second VAE. Not TAESD."""
    dest = dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")

    def hook(current, total, preview=None, node_id=None):
        image = _preview_to_pil(preview)
        if image is None:
            return
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(tmp, format="JPEG", quality=80)
            os.replace(tmp, dest)
            step_n = int(current) if current is not None else 0
            step_d = int(total) if total else 0
            node = node_id or ""
            print(
                f"[h3-preview] step={step_n}/{step_d} node={node} file={dest}",
                flush=True,
            )
        except OSError as exc:
            print(f"warn: live preview write failed: {exc}", flush=True)

    comfy_utils.set_progress_bar_global_hook(hook)


def _ensure_fp16_accumulation(argv: list[str]) -> None:
    """Enable Comfy ``--fast fp16_accumulation`` for the H3 video VAE.

    That flag is what turns on kitchen ``fp16_conv3d`` in the encoder. A bare
    ``--fast`` enables every performance feature (fp8 matmul, cublas_ops,
    autotune). Do not pass that. Argparse keeps only the last ``--fast``, so
    merge this name into an existing valued list instead of appending a second
    flag. MiniMax H3's sampler allow-list is bf16/fp32, so the companion
    PRIORITIZE_FP16 switch does not retarget the DiT.
    """
    last: tuple[int, int] | None = None
    i = 0
    while i < len(argv):
        if argv[i] != "--fast":
            i += 1
            continue
        j = i + 1
        while j < len(argv) and not argv[j].startswith("-"):
            j += 1
        last = (i, j)
        i = j
    if last is None:
        argv.extend(["--fast", "fp16_accumulation"])
        return
    start, end = last
    values = argv[start + 1 : end]
    if not values or "fp16_accumulation" in values:
        return
    argv.insert(end, "fp16_accumulation")


def _collect_comfy_extra(unknown: list[str], remainder: list[str] | None) -> list[str]:
    comfy_extra: list[str] = []
    for chunk in (unknown, list(remainder or [])):
        for a in chunk:
            if a == "--":
                continue
            if a not in comfy_extra:
                comfy_extra.append(a)
    # Always disable pinned host weight runway on this 64 GB box — Comfy's default
    # "Enabled pinned memory ~25 GB" starves the desktop when TE+DiT are resident.
    if "--disable-pinned-memory" not in comfy_extra:
        comfy_extra.append("--disable-pinned-memory")
    # HARD RULE (owner 16 GB / 64 GB): never page model weights through the
    # checkpoint NVMe. Comfy AIMDO DynamicVRAM mmaps safetensors and can
    # bounce dirty pages back to the weight files (disk thrash / SSD wear).
    # Force legacy RAM↔VRAM only; keep weights in system RAM, not disk swap.
    if "--disable-dynamic-vram" not in comfy_extra:
        comfy_extra.append("--disable-dynamic-vram")
    if "--disable-mmap" not in comfy_extra:
        comfy_extra.append("--disable-mmap")
    # Async CUDA weight streams are fine (GPU↔RAM); keep them. Disk is the
    # forbidden path (handled by disable-dynamic-vram / disable-mmap above).
    _ensure_fp16_accumulation(comfy_extra)
    return comfy_extra


def _normalize_save_video_inputs(prompt: dict) -> None:
    """Comfy 0.36 SaveVideo DynamicCombo: live `format` is the option key.

    Nested codec is the dotted child `format.codec`. Putting a nested dict
    in `format` does not match option keys, so the V3 expander drops it and
    execute() never receives `format`. Compiled 0.33.4 packages still emit
    sibling `format`/`codec` strings. Rewrite in place; do not recapture.
    """
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "SaveVideo":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        fmt = inputs.get("format")
        codec = inputs.get("codec")
        dotted = inputs.get("format.codec")
        format_name = "auto"
        codec_name = "auto"
        if isinstance(fmt, str):
            format_name = fmt
        elif isinstance(fmt, dict):
            inner = fmt.get("format")
            if isinstance(inner, str):
                format_name = inner
            inner_codec = fmt.get("codec")
            if isinstance(inner_codec, dict) and isinstance(inner_codec.get("codec"), str):
                codec_name = inner_codec["codec"]
            elif isinstance(inner_codec, str):
                codec_name = inner_codec
        if isinstance(codec, str):
            codec_name = codec
        elif isinstance(dotted, str):
            codec_name = dotted
        elif isinstance(dotted, dict) and isinstance(dotted.get("codec"), str):
            codec_name = dotted["codec"]
        inputs["format"] = format_name
        inputs["format.codec"] = codec_name
        if "codec" in inputs:
            del inputs["codec"]



def _boot_comfy(*, extra: list[str], require_accel: bool, live_path):
    os.chdir(COMFY_ROOT)

    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))

    # RAM↔VRAM only. Never AIMDO/mmap weight paging onto the checkpoint NVMe.
    # Do NOT force --lowvram (that parks the 32B TE on CPU and spikes system RAM).
    sys.argv = [str(Path(__file__).resolve())] + extra

    import comfy.options

    comfy.options.enable_args_parsing(True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from utils.extra_config import load_extra_path_config

    ckpt_base = _resolve_checkpoints_base()
    extra_yaml = _write_runtime_extra_model_paths(ckpt_base)
    print(f"h3_checkpoints_base: {ckpt_base}")
    print(f"h3_extra_model_paths: {extra_yaml}")
    load_extra_path_config(str(extra_yaml))

    # KJNodes (and a few extras) touch PromptServer.instance at import time.
    # Headless runs never start the HTTP server — stub enough surface for
    # MiniMax H3 Mem-Efficient Sage and routes to register without aiohttp.
    try:
        import server as _comfy_server

        if getattr(_comfy_server.PromptServer, "instance", None) is None:

            class _Router:
                frozen = False

                def add_get(self, *a, **k):
                    return None

                def add_post(self, *a, **k):
                    return None

                def add_routes(self, *a, **k):
                    return None

            class _App:
                def __init__(self) -> None:
                    self.router = _Router()

                def add_routes(self, *a, **k):
                    return None

            class _NodeReplaceManager:
                def register(self, *a, **k):
                    return None

                def add_routes(self, *a, **k):
                    return None

                def apply_replacements(self, prompt):
                    return prompt

            class _HeadlessPromptServer:
                def __init__(self) -> None:
                    self.client_id = None
                    self.app = _App()
                    self.node_replace_manager = _NodeReplaceManager()

                def send_sync(self, *a, **k):
                    return None

                def send(self, *a, **k):
                    return None

                def queue_updated(self):
                    return None

                def add_on_prompt_handler(self, *a, **k):
                    return None

            _comfy_server.PromptServer.instance = _HeadlessPromptServer()  # type: ignore[attr-defined]
    except Exception as exc:
        print("warn: could not stub PromptServer.instance:", exc)

    import folder_paths
    import nodes
    import execution
    from comfy.cli_args import args as comfy_args
    import comfy.model_management as mm
    import comfy.memory_management as mem_mm
    import comfy.utils as comfy_utils

    if live_path is not None:
        try:
            _install_live_preview_hook(comfy_utils, live_path)
            print(f"h3_live_preview: {live_path}", flush=True)
        except Exception as exc:
            print("warn: could not install live preview hook:", exc, flush=True)

    # Belt-and-suspenders: even if a future Comfy path flips these, refuse mmap
    # weight loads that can dirty-write checkpoint files on the NVMe.
    if not getattr(comfy_args, "disable_dynamic_vram", False):
        raise RuntimeError(
            "H3 runner requires --disable-dynamic-vram (AIMDO mmap can page weights to disk)"
        )
    if not getattr(comfy_args, "disable_mmap", False):
        raise RuntimeError(
            "H3 runner requires --disable-mmap (safetensors mmap can dirty-write checkpoints)"
        )
    fast_names = {
        item.value if hasattr(item, "value") else str(item)
        for item in (getattr(comfy_args, "fast", None) or ())
    }
    if "fp16_accumulation" not in fast_names:
        raise RuntimeError(
            "H3 runner requires --fast fp16_accumulation (H3 video VAE encoder conv)"
        )
    mem_mm.aimdo_enabled = False
    comfy_utils.DISABLE_MMAP = True
    # Refuse AIMDO even if a later import tries to flip it (main.py path).
    try:
        import comfy.model_patcher as _mp

        # Keep legacy patcher — Dynamic patcher is the AIMDO/mmap path.
        if hasattr(_mp, "ModelPatcher") and hasattr(_mp, "CoreModelPatcher"):
            _mp.CoreModelPatcher = _mp.ModelPatcher
    except Exception as exc:
        print("warn: could not pin CoreModelPatcher to legacy:", exc, flush=True)

    print(
        "comfy_vram:",
        {
            "lowvram": comfy_args.lowvram,
            "highvram": comfy_args.highvram,
            "disable_dynamic_vram": comfy_args.disable_dynamic_vram,
            "disable_mmap": getattr(comfy_args, "disable_mmap", None),
            "disable_pinned_memory": getattr(comfy_args, "disable_pinned_memory", None),
            "aimdo_enabled": getattr(mem_mm, "aimdo_enabled", None),
            "use_sage_attention": getattr(comfy_args, "use_sage_attention", None),
            "use_flash_attention": getattr(comfy_args, "use_flash_attention", None),
            "fast": sorted(fast_names),
            "preview_method": str(getattr(comfy_args, "preview_method", None)),
            "dynamic_vram": mm.enables_dynamic_vram()
            if hasattr(mm, "enables_dynamic_vram")
            else "n/a",
            "vram_state": str(getattr(mm, "vram_state", "?")),
            "argv": sys.argv,
        },
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()

    # Builtin extras (nodes_minimax_h3, EasyCache, custom sampler, …)
    # + pinned custom_nodes (Turbo / Sol / Spectrum / KJ / FBC).
    failed = asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    if failed:
        print("node import failures:", failed)

    missing_core = [n for n in CORE_NODES if n not in nodes.NODE_CLASS_MAPPINGS]
    if missing_core:
        print("MISSING CORE NODES:", missing_core)
        print(
            "have MiniMax*",
            [k for k in nodes.NODE_CLASS_MAPPINGS if "MiniMax" in k or "minimax" in k],
        )
        raise SystemExit(3)

    missing_accel = [n for n in ACCEL_NODES if n not in nodes.NODE_CLASS_MAPPINGS]
    if missing_accel:
        print("ACCEL NODES missing (optional unless required):", missing_accel)
        if require_accel:
            raise SystemExit(4)

    dmodels = folder_paths.get_filename_list("diffusion_models")
    tes = folder_paths.get_filename_list("text_encoders")
    vaes = folder_paths.get_filename_list("vae")
    try:
        loras = folder_paths.get_filename_list("loras")
    except Exception:
        loras = []
    print("diffusion_models:", dmodels)
    print("text_encoders:", tes)
    print("vae:", vaes)
    print("loras:", loras)
    for must in (
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ):
        pool = dmodels + tes + vaes
        if must not in pool:
            print("WARNING missing from folder_paths:", must)

    _install_weight_cache(nodes)


def _share_loader_out(kind: str, out: tuple) -> tuple:
    """Clone UNET/CLIP patchers so a later graph cannot mutate the cache."""
    obj = out[0]
    if kind in ("unet", "clip") and hasattr(obj, "clone"):
        return (obj.clone(),)
    return out


def _install_weight_cache(nodes_mod) -> None:
    """Intern UNET / CLIP / VAE by filename so a second graph cannot reload them.

    Window-2 of keep-warm died at ``Requested to load MiniMaxH3`` because
    UNETLoader always ``load_torch_file``s a new copy. A second 15 GB TE
    plus a second UNET in one process hard-locks this 16 GB box.
    """
    global _WEIGHT_CACHE_INSTALLED
    if _WEIGHT_CACHE_INSTALLED:
        return

    def wrap(class_name: str, method_name: str, kind: str) -> None:
        cls = nodes_mod.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            print(f"[h3-comfy] weight cache: missing {class_name}", flush=True)
            return
        orig = getattr(cls, method_name)

        def wrapped(self, *args, **kwargs):
            key = (kind, args, tuple(sorted(kwargs.items())))
            hit = _WEIGHT_CACHE.get(key)
            if hit is not None:
                print(f"[h3-comfy] reuse cached {kind} {args[:1]}", flush=True)
                return _share_loader_out(kind, hit)
            print(f"[h3-comfy] load and cache {kind} {args[:1]}", flush=True)
            out = orig(self, *args, **kwargs)
            _WEIGHT_CACHE[key] = out
            return out

        setattr(cls, method_name, wrapped)

    wrap("UNETLoader", "load_unet", "unet")
    wrap("CLIPLoader", "load_clip", "clip")
    wrap("VAELoader", "load_vae", "vae")
    _WEIGHT_CACHE_INSTALLED = True
    print("[h3-comfy] weight cache installed (UNET/CLIP/VAE interned)", flush=True)


def run_workflow(
    workflow_path: Path | None = None,
    *,
    prompt: dict | None = None,
    out_note: Path | None = None,
    comfy_extra: list[str] | None = None,
    require_accel: bool = False,
    attn: str = "sage",
) -> dict:
    """Execute one API workflow. Returns timing note dict.

    Provide either ``workflow_path`` or an in-memory ``prompt`` graph.
    """
    if workflow_path is None and prompt is None:
        raise ValueError("workflow_path or prompt required")

    extra = list(comfy_extra or [])
    live_path = _live_preview_path()
    if live_path is not None and "--preview-method" not in extra:
        extra.extend(["--preview-method", "auto"])
    attn = (attn or "sage").lower()
    if attn in ("sage", "fast", "auto"):
        if "--use-sage-attention" not in extra and "--use-flash-attention" not in extra:
            extra.append("--use-sage-attention")
    elif attn in ("fa2", "flash", "hq"):
        # Strip sage if present; fa2 exclusive.
        extra = [a for a in extra if a != "--use-sage-attention"]
        if "--use-flash-attention" not in extra:
            extra.append("--use-flash-attention")
    if "--disable-pinned-memory" not in extra:
        extra.append("--disable-pinned-memory")
    if "--disable-dynamic-vram" not in extra:
        extra.append("--disable-dynamic-vram")
    if "--disable-mmap" not in extra:
        extra.append("--disable-mmap")
    _ensure_fp16_accumulation(extra)

    if workflow_path is not None:
        workflow_path = workflow_path.expanduser().resolve()
        if not workflow_path.is_file():
            raise FileNotFoundError(f"workflow not found: {workflow_path}")

    out_note_path = out_note.expanduser().resolve() if out_note is not None else None

    global _BOOTED
    if not _BOOTED:
        _boot_comfy(extra=extra, require_accel=require_accel, live_path=live_path)
        _BOOTED = True

    import folder_paths
    import nodes
    import execution
    from comfy.cli_args import args as comfy_args
    import comfy.model_management as mm

    vaes = folder_paths.get_filename_list("vae")
    missing_accel = [n for n in ACCEL_NODES if n not in nodes.NODE_CLASS_MAPPINGS]

    if prompt is None:
        assert workflow_path is not None
        prompt = json.loads(workflow_path.read_text(encoding="utf-8"))

    _normalize_save_video_inputs(prompt)

    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "VAELoader":
            continue
        name = node.get("inputs", {}).get("vae_name")
        if not isinstance(name, str) or not name:
            continue
        base = Path(name).name
        if "audio" in base.lower():
            continue
        if base not in vaes:
            print("WARNING missing from folder_paths:", base)

    for nid, node in prompt.items():
        for key, val in node.get("inputs", {}).items():
            if key.endswith("_name") and isinstance(val, str) and (
                ("/" in val) or ("\\" in val) or (":" in val)
            ):
                raise SystemExit(f"refusing path-like model ref {nid}.{key}={val!r}")

    # Graph-required class_types must exist.
    graph_types = {n.get("class_type") for n in prompt.values() if isinstance(n, dict)}
    missing_graph = sorted(t for t in graph_types if t not in nodes.NODE_CLASS_MAPPINGS)
    if missing_graph:
        print("MISSING NODES REQUIRED BY GRAPH:", missing_graph)
        raise SystemExit(5)

    prompt_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    valid = asyncio.run(execution.validate_prompt(prompt_id, prompt, None))
    print("validate_ok:", valid[0] if isinstance(valid, (tuple, list)) else valid)
    if isinstance(valid, (tuple, list)) and valid[0] is not True:
        print(json.dumps(valid, indent=2, default=str)[:6000])
        raise SystemExit(2)
    execute_outputs = (
        list(valid[2]) if isinstance(valid, (tuple, list)) and len(valid) > 2 else []
    )
    if not execute_outputs:
        print("no output nodes resolved from validate_prompt")
        raise SystemExit(2)
    print("execute_outputs:", execute_outputs)

    class _Server:
        client_id = None
        last_prompt_id = None

        def send_sync(self, *a, **k):
            return None

        def queue_updated(self):
            return None

        def send(self, *a, **k):
            return None

    executor = execution.PromptExecutor(
        _Server(),
        cache_type=execution.CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0.0, "ram_inactive": 0.0},
    )
    t1 = time.perf_counter()
    executor.execute(prompt, prompt_id, {"client_id": None}, execute_outputs)
    t2 = time.perf_counter()

    success = bool(getattr(executor, "success", False))
    err = None
    if not success:
        err = getattr(executor, "history_result", None)
        print(
            "execute_failed history:",
            json.dumps(err, indent=2, default=str)[:6000] if err else None,
        )

    outputs_dir = Path(folder_paths.get_output_directory())
    note = {
        "workflow": str(workflow_path) if workflow_path else None,
        "validate_s": t1 - t0,
        "execute_wall_s": t2 - t1,
        "total_wall_s": t2 - t0,
        "success": success,
        "prompt_id": prompt_id,
        "outputs_dir": str(outputs_dir),
        "comfy_argv": sys.argv,
        "vram_state": str(getattr(mm, "vram_state", "?")),
        "accel_missing": missing_accel,
        "graph_class_types": sorted(graph_types),
    }
    if out_note_path is None and workflow_path is not None:
        out_note_path = COMFY_ROOT / "output" / f"timing_{workflow_path.stem}.json"
    if out_note_path is not None:
        out_note_path = Path(out_note_path)
        out_note_path.parent.mkdir(parents=True, exist_ok=True)
        out_note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
        print("wrote", out_note_path)
    print(json.dumps(note, indent=2))
    if not success:
        raise RuntimeError("Comfy execute failed")
    # Drop dead graph tensors only. Do NOT unload UNET/TE/VAE — that is what
    # made window 2 reload 15 GB + DiT in-process and hard-lock the box.
    try:
        import gc

        if hasattr(mm, "cleanup_models_gc"):
            mm.cleanup_models_gc()
        gc.collect()
    except Exception as exc:
        print("warn: post-graph gc:", exc, flush=True)
    return note


def _serve_one(req: dict) -> dict:
    workflow = Path(str(req.get("workflow") or ""))
    out_note = req.get("out_note")
    try:
        note = run_workflow(
            workflow,
            out_note=Path(out_note) if out_note else None,
        )
        return {"ok": True, "note": note}
    except SystemExit as exc:
        return {"ok": False, "error": f"exit {exc.code}"}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {"ok": False, "error": str(exc)}


def serve_loop(host: str = "127.0.0.1", port: int = 0) -> int:
    """Boot Comfy once, then execute graphs sent as JSON lines over TCP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(port)))
    sock.listen(8)
    bound_host, bound_port = sock.getsockname()
    print(f"h3-comfy-serve tcp={bound_host}:{bound_port}", flush=True)
    try:
        while True:
            conn, _addr = sock.accept()
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf.strip():
                    continue
                line = buf.split(b"\n", 1)[0]
                req = json.loads(line.decode("utf-8"))
                if req.get("cmd") == "quit":
                    conn.sendall(b'{"ok":true,"quit":true}\n')
                    return 0
                reply = _serve_one(req)
                conn.sendall((json.dumps(reply, default=str) + "\n").encode("utf-8"))
    finally:
        sock.close()
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workflow", type=Path, nargs="?", default=None, help="API-format workflow JSON")
    ap.add_argument("--out-note", type=Path, default=None, help="Write timing JSON here")
    ap.add_argument(
        "--attn",
        default="sage",
        choices=["sage", "fa2", "flash", "hq", "fast", "auto", "none"],
        help="Attention backend via Comfy CLI flags (default sage)",
    )
    ap.add_argument(
        "--require-accel",
        action="store_true",
        help="Fail if turbo/sol/spectrum/KJ nodes failed to import",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="Keep Comfy loaded and execute graphs over loopback TCP (JSON lines)",
    )
    ap.add_argument(
        "--serve-bind",
        default="127.0.0.1:0",
        help="host:port for --serve (port 0 = ephemeral)",
    )
    ap.add_argument(
        "--build",
        action="store_true",
        help="Build a graph via h3_workflow_build instead of loading a JSON file",
    )
    ap.add_argument("--mode", default="fl2va")
    ap.add_argument("--prompt", default="a cinematic test shot")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=736)
    ap.add_argument("--length", type=int, default=39)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--first-image", default=None, help="Basename under ComfyUI/input/")
    ap.add_argument("--turbo", default="off", choices=["off", "v4", "v1", "ema_wan"])
    ap.add_argument("--sol", action="store_true")
    ap.add_argument("--cache", default="off", choices=["off", "spectrum", "easy", "fbc"])
    ap.add_argument("--prefix", default="h3/smoke")
    ap.add_argument("--video-vae", default=None, help="Video VAE filename under vae/")
    ap.add_argument(
        "--comfy-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args after -- forwarded to Comfy",
    )
    ours, unknown = ap.parse_known_args()
    comfy_extra = _collect_comfy_extra(unknown, list(ours.comfy_args or []))

    prompt = None
    workflow_path = ours.workflow
    if ours.build:
        from h3_workflow_build import build_h3_workflow

        prompt = build_h3_workflow(
            mode=ours.mode,
            prompt=ours.prompt,
            width=ours.width,
            height=ours.height,
            length=ours.length,
            seed=ours.seed,
            steps=ours.steps,
            first_image=ours.first_image,
            turbo=ours.turbo,
            sol=ours.sol,
            cache=ours.cache,
            filename_prefix=ours.prefix,
            video_vae=ours.video_vae,
        )
        # Persist built graph next to timing for debug.
        built = COMFY_ROOT / "output" / f"built_{ours.prefix.replace('/', '_')}.json"
        built.parent.mkdir(parents=True, exist_ok=True)
        built.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
        print("wrote built graph", built)
    elif workflow_path is None and not ours.serve:
        raise SystemExit("workflow path required unless --build or --serve")

    if ours.serve:
        host, _, port_s = str(ours.serve_bind).rpartition(":")
        host = host or "127.0.0.1"
        try:
            return serve_loop(host, int(port_s or "0"))
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 1
        except Exception as e:
            print("ERROR:", e)
            return 1

    try:
        run_workflow(
            workflow_path,
            prompt=prompt,
            out_note=ours.out_note,
            comfy_extra=comfy_extra,
            require_accel=ours.require_accel,
            attn=ours.attn,
        )
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
