"""WanGP-style FL2VA multi-window continue (pixel last-frame keyframe).

Protocol (FL2VA only — not Ref2VA latent-tail):

  for each window:
    last RGB of previous full window → encode_condition → first-frame keyframe
    full-window joint A/V sample (pure noise target)
    drop first OVERLAP frames (+ matching audio) before commit  (default OVERLAP=1)
    next prev_tail = last OVERLAP frames of the *full* window (pre-drop)

Defaults match WanGP H3 handler: window=362 (~15s), overlap=1, steps=20,
shift=12, fps=24, audio 32 kHz. No latent overlap noise, no color correction,
no audio cond carry.

VRAM sequencing: VAE encode/decode and DiT sample run as separate subprocesses
so the streamed DiT and official VAE never co-reside on 16 GB.
Gemmy invokes this with the runtime venv python + GEMMY_H3_CHECKPOINTS.

Multishot memory (opt-in):
  ComfyUI-H3-Multishot separates KEYFRAME (seam lock = last RGB, same as
  stock continue) from MEMORY (identity bank shown to the *text encoder*
  via Qwen3-VL vision: persistent start anchor + last N shot ends).
  This driver:
    - tracks an on-disk anchor + recent-end bank under ``memory/`` (PNG + latent)
    - **default memory path:** re-encodes each window's text embeds through
      ``h3_text_encode.py --image …`` (MiniMax Picture labels + vision tower
      already inside the TE safetensors — no separate mmproj)
    - optional legacy DiT stand-in: ``--dit-memory-refs`` packs bank frames as
      DiT ``--ref-image`` (needs Ref2VA weights + ``--allow-keyframe-refs``)

Requires ``--prompt`` / ``--prompt-file`` / ``--prompts`` when TE memory is on
(so each window can re-encode). Plain ``--text`` safetensors still works for
stock continue without memory images.

Decision policy (one-shot vs continue vs hard cut) lives in root AGENTS.md /
README — this script is the continue path only.
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
from typing import List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from h3_vae_decode import decode_audio, decode_video, build_audio_vae, build_video_vae, mux_av, save_mp4  # noqa: E402
from h3_vae_encode import encode_pixels_m11, fit_image_rgb, load_image_rgb, save_keyframe_latent  # noqa: E402

# Package canvas + temporal grid (shared with sample pipeline)
sys.path.insert(0, str(ROOT))
from minimax_h3.canvas import resolve_canvas  # noqa: E402
from minimax_h3.frame_grid import FPS as GRID_FPS  # noqa: E402
from minimax_h3.frame_grid import floor_frames, snap_frames  # noqa: E402

FPS = GRID_FPS
SR = 32000
OVERLAP_DEFAULT = 1
WINDOW_DEFAULT = 362  # ~15.08s on 5+17k grid


def _child_env() -> dict:
    """Propagate checkpoints root + package path for TE/VAE/sample children."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    pp = env.get("PYTHONPATH", "")
    root = str(ROOT)
    env["PYTHONPATH"] = root + (os.pathsep + pp if pp else "")
    return env


def run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(list(cmd), cwd=str(cwd or ROOT), env=_child_env())
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


def runtime_python() -> List[str]:
    """Use the active interpreter (Gemmy H3 venv), not a nested `uv run`."""
    return [sys.executable]


def sample_window(
    *,
    weights: Path,
    text: Path,
    out_dir: Path,
    steps: int,
    width: int,
    height: int,
    length: int,
    seed: int,
    first_frame: Optional[Path],
    device: str,
    attn: str,
    sage_backend: str,
    extra_sample_args: Sequence[str],
    ref_images: Optional[Sequence[Path]] = None,
    allow_keyframe_refs: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *runtime_python(),
        "-m",
        "minimax_h3",
        "--device",
        device,
        "--attn",
        attn,
        "--sage-backend",
        sage_backend,
        "sample",
        str(weights),
        str(text),
        "--steps",
        str(steps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--length",
        str(length),
        "--seed",
        str(seed),
        "--out",
        str(out_dir),
        *list(extra_sample_args),
    ]
    if first_frame is not None:
        cmd.extend(["--first-frame", str(first_frame)])
    if ref_images:
        if first_frame is not None and not allow_keyframe_refs:
            raise RuntimeError(
                "ref_images with first_frame requires allow_keyframe_refs "
                "(experimental Multishot DiT memory)"
            )
        if allow_keyframe_refs:
            cmd.append("--allow-keyframe-refs")
        for p in ref_images:
            cmd.extend(["--ref-image", str(p)])
    run_cmd(cmd)
    vpath = out_dir / "video_latent.safetensors"
    apath = out_dir / "audio_latent.safetensors"
    if not vpath.is_file() or not apath.is_file():
        raise FileNotFoundError(f"sample missing latents under {out_dir}")
    return out_dir


def decode_window(
    latent_dir: Path,
    *,
    device: str,
    frames: int,
    fps: int = FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return video uint8 [T,H,W,3] and audio float32 [L,2] in [-1,1]."""
    import safetensors.torch
    from scipy.io import wavfile

    video_z = safetensors.torch.load_file(str(latent_dir / "video_latent.safetensors"))["video_latent"]
    audio_z = safetensors.torch.load_file(str(latent_dir / "audio_latent.safetensors"))["audio_latent"]

    torch.set_grad_enabled(False)
    print(f"decode video latent {tuple(video_z.shape)}…", flush=True)
    vvae = build_video_vae(device)
    try:
        pixels = decode_video(vvae, video_z, frames).float().cpu()  # [1,3,T,H,W] [-1,1]
    finally:
        del vvae
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    frames_np = pixels.squeeze(0).permute(1, 2, 3, 0).numpy()  # [T,H,W,3]
    actual_t = frames_np.shape[0]
    if actual_t < frames:
        print(f"warn: decoded T={actual_t} < requested {frames}", flush=True)

    print(f"decode audio latent {tuple(audio_z.shape)}…", flush=True)
    avae = build_audio_vae(device)
    try:
        stereo = decode_audio(avae, audio_z.to(next(avae.parameters()).device)).float().cpu()
    finally:
        del avae
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wav = stereo[0].numpy().T  # [L, 2]
    # Fit audio length to frame count * sr / fps (WanGP-style)
    target_samples = int(round(actual_t * SR / float(fps)))
    if wav.shape[0] > target_samples:
        wav = wav[:target_samples]
    elif wav.shape[0] < target_samples:
        pad = np.zeros((target_samples - wav.shape[0], wav.shape[1]), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)

    u8 = np.clip((frames_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return u8, wav.astype(np.float32)


def write_seg_media(
    out: Path,
    u8: np.ndarray,
    wav_f32: np.ndarray,
    *,
    fps: int = FPS,
    mux: bool = True,
) -> None:
    from scipy.io import wavfile

    out.mkdir(parents=True, exist_ok=True)
    vpath = out / "video.mp4"
    apath = out / "audio.wav"
    save_mp4(u8, vpath, fps=fps)
    wav_i16 = np.clip(wav_f32 * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(apath), SR, wav_i16)
    if mux:
        try:
            mux_av(vpath, apath, out / "output.mp4")
        except Exception as e:
            print(f"mux skip: {e}", flush=True)


def last_frame_m11(u8_frame: np.ndarray) -> np.ndarray:
    return (u8_frame.astype(np.float32) / 127.5) - 1.0


def parse_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        # Multishot script style: '---' between shots; else blank-line blocks.
        if "\n---\n" in text or text.strip().startswith("---"):
            blocks = [b.strip() for b in text.split("---") if b.strip()]
        else:
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if len(blocks) > 1:
            prompts.extend(blocks)
        else:
            prompts.append(text.strip())
    if args.prompts:
        for p in args.prompts:
            prompts.append(p)
    # text embeds are precomputed paths — prompts list is only for logging /
    # multi-text path. Primary conditioning is --text / --texts safetensors.
    return prompts


def resolve_text_paths(args) -> List[Path]:
    """Pre-encoded text paths (optional when TE memory re-encodes from prompts)."""
    paths: List[Path] = []
    if args.text is not None:
        paths.append(Path(args.text))
    if args.texts:
        paths.extend(Path(p) for p in args.texts)
    return paths


def save_u8_png(u8_frame: np.ndarray, dest: Path) -> Path:
    """Write one RGB uint8 frame as PNG for TE vision ``--image``."""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8_frame, mode="RGB").save(dest)
    return dest


def encode_text_with_memory(
    *,
    prompt: str,
    images: Sequence[Path],
    dest: Path,
    device: str,
) -> Path:
    """Run h3_text_encode.py with Multishot Picture images → text.safetensors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        *runtime_python(),
        str(ROOT / "scripts" / "h3_text_encode.py"),
        prompt,
        str(dest),
        "--device",
        device,
    ]
    for p in images:
        cmd.extend(["--image", str(p)])
    run_cmd(cmd)
    if not dest.is_file():
        raise FileNotFoundError(f"text encode missing output {dest}")
    return dest


def encode_u8_frame_latent(
    u8_frame: np.ndarray,
    *,
    dest: Path,
    device: str,
    encode_seed: int,
    exp_h: int,
    exp_w: int,
) -> Path:
    """Encode one RGB uint8 frame → DiT-normalized keyframe/ref latent on disk."""
    z = encode_pixels_m11(
        last_frame_m11(u8_frame),
        width=None,
        height=None,
        device=device,
        encode_seed=encode_seed,
    )
    if z.shape[3] != exp_h or z.shape[4] != exp_w:
        raise RuntimeError(
            f"cond spatial {z.shape[3]}x{z.shape[4]} != {exp_h}x{exp_w}; "
            "start image / decode canvas mismatch"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_keyframe_latent(z, dest)
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="WanGP-style FL2VA multi-window continue (last RGB → first-frame + drop f0)"
    )
    ap.add_argument(
        "weights",
        type=Path,
        help="FL2VA pruned int8 convrot checkpoint",
    )
    ap.add_argument(
        "--text",
        type=Path,
        default=None,
        help="pre-encoded text safetensors (stock continue; optional when TE memory re-encodes)",
    )
    ap.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="optional per-window text embeds (falls back to last)",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="prompt string (required for TE vision memory re-encode; else log-only)",
    )
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--start-image", type=Path, default=None, help="optional still for window 1")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "fl2va_continue")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="sage")
    ap.add_argument("--sage-backend", default="auto")
    ap.add_argument("--width", type=int, default=None, help="with --height; default owner preview 864×480")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--aspect", type=str, default=None, help="e.g. 16:9 / 9:16 / 1:1 with --mp or --short-edge")
    ap.add_argument("--megapixels", "--mp", type=float, default=None, dest="megapixels")
    ap.add_argument("--short-edge", type=int, default=None)
    ap.add_argument(
        "--canvas-mode",
        choices=["preview", "official", "raw"],
        default="preview",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=WINDOW_DEFAULT,
        help=f"frames per window before snap (default {WINDOW_DEFAULT} ≈15s)",
    )
    ap.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="target committed frames (default: 2 windows worth)",
    )
    ap.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help="alternative to --total-frames: exact window count",
    )
    ap.add_argument("--overlap", type=int, default=OVERLAP_DEFAULT, help="must be 1 for WanGP FL2VA")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seed-mode",
        choices=["same", "inc"],
        default="same",
        help="same=seed every window (WanGP-like); inc=seed+w",
    )
    ap.add_argument("--encode-seed", type=int, default=42)
    ap.add_argument(
        "--extra-sample-arg",
        action="append",
        default=[],
        help="extra arg passed through to sample (repeatable), e.g. --extra-sample-arg=--profile",
    )
    ap.add_argument("--keep-segs", action="store_true", help="keep per-window mp4/latents")
    # Multishot-style memory bank (opt-in). KEYFRAME stays last-RGB continue;
    # MEMORY bank is tracked on disk. DiT-side packing needs Ref2VA weights.
    ap.add_argument(
        "--memory-frames",
        type=int,
        default=0,
        help=(
            "Multishot memory: keep last N shot-end frames in the bank "
            "(0 = stock continue, no bank). Bank written under <out>/memory/. "
            "With a prompt, each window re-encodes TE with anchor+ends as --image."
        ),
    )
    ap.add_argument(
        "--anchor-frames",
        type=int,
        default=0,
        help=(
            "Multishot memory: keep a persistent start anchor (0/1). "
            "Set from --start-image or window-1 frame 0."
        ),
    )
    ap.add_argument(
        "--te-memory",
        action="store_true",
        default=None,
        help=(
            "Force TE vision memory re-encode (MiniMax Picture images through "
            "h3_text_encode.py). Default: on when --memory-frames/--anchor-frames "
            "and a prompt are set."
        ),
    )
    ap.add_argument(
        "--no-te-memory",
        action="store_true",
        help="Disable TE vision memory even if the bank is tracked (bank-only / DiT refs).",
    )
    ap.add_argument(
        "--dit-memory-refs",
        action="store_true",
        help=(
            "Legacy stand-in: pack memory bank frames as DiT --ref-image alongside "
            "the FL2VA first-frame keyframe (needs Ref2VA checkpoint). Prefer TE memory."
        ),
    )
    args = ap.parse_args(argv)

    if args.overlap != 1:
        print(
            f"warn: WanGP FL2VA locks overlap=1; got {args.overlap}. "
            "Proceeding but quality is unproven.",
            flush=True,
        )

    if args.width is None and args.height is None and args.aspect is None and args.megapixels is None:
        # Owner convenience default (preview ladder), not a hard 16:9 lock
        canvas = resolve_canvas(width=864, height=480, mode=args.canvas_mode)
    else:
        canvas = resolve_canvas(
            width=args.width,
            height=args.height,
            aspect=args.aspect,
            megapixels=args.megapixels,
            short_edge=args.short_edge,
            mode=args.canvas_mode,
        )
    canvas_w, canvas_h = canvas.width, canvas.height
    # Pass adapted W/H into sample so encode/sample agree
    args.width, args.height = canvas_w, canvas_h
    window = snap_frames(args.window)
    overlap = max(1, int(args.overlap))
    texts = resolve_text_paths(args)
    prompts = parse_prompts(args)

    memory_frames = max(0, int(args.memory_frames))
    anchor_frames = max(0, min(2, int(args.anchor_frames)))
    dit_memory_refs = bool(args.dit_memory_refs)
    if dit_memory_refs and memory_frames == 0 and anchor_frames == 0:
        memory_frames = 2
        anchor_frames = 1
        print(
            "note: --dit-memory-refs implies Multishot defaults "
            f"memory_frames={memory_frames} anchor_frames={anchor_frames}",
            flush=True,
        )
    memory_on = memory_frames > 0 or anchor_frames > 0 or dit_memory_refs
    # TE vision memory is the real Multishot path; on by default when bank+prompt.
    if args.no_te_memory:
        te_memory = False
    elif args.te_memory:
        te_memory = True
    else:
        te_memory = bool(memory_on and prompts)
    if te_memory and not prompts:
        raise SystemExit(
            "TE vision memory needs --prompt / --prompt-file / --prompts "
            "(raw text to re-encode with memory images each window)"
        )
    pending_prompt_encode = False
    if not te_memory and not texts:
        if not prompts:
            raise SystemExit(
                "need --prompt / --prompt-file, or --text / --texts "
                "(pre-encoded safetensors), or TE memory with a prompt + "
                "--memory-frames/--anchor-frames"
            )
        pending_prompt_encode = True

    if args.num_windows is not None:
        num_windows = max(1, int(args.num_windows))
        # committed ≈ win1 + (n-1)*(window-overlap)
        total_target = window + (num_windows - 1) * (window - overlap)
    elif args.total_frames is not None:
        total_target = snap_frames(args.total_frames)
        # derive windows
        if total_target <= window:
            num_windows = 1
        else:
            rest = total_target - window
            num_windows = 1 + int(np.ceil(rest / float(window - overlap)))
    else:
        num_windows = 2
        total_target = window + (window - overlap)

    out: Path = args.out
    if out.exists():
        # don't wipe blindly if user points at something unexpected
        print(f"out dir {out} (existing files may be overwritten)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if pending_prompt_encode:
        dest = out / "text_prompt.safetensors"
        encode_text_with_memory(
            prompt=prompts[0],
            images=[],
            dest=dest,
            device=str(args.device),
        )
        texts = [dest]
        print(f"encoded prompt → {dest}", flush=True)

    meta = {
        "canvas": [canvas_w, canvas_h],
        "canvas_spec": canvas.as_dict(),
        "window": window,
        "overlap": overlap,
        "steps": args.steps,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "num_windows": num_windows,
        "total_target": total_target,
        "weights": str(args.weights),
        "protocol": "fl2va_last_rgb_keyframe_drop_overlap",
        "memory_frames": memory_frames,
        "anchor_frames": anchor_frames,
        "te_memory": te_memory,
        "dit_memory_refs": dit_memory_refs,
        "memory_note": (
            "Bank tracks Multishot-style anchor+recent ends (PNG + latent). "
            + (
                "TE vision memory ON: each window re-encodes via h3_text_encode.py "
                "with MiniMax Picture images (visual.* in TE ckpt)."
                if te_memory
                else "TE vision memory OFF."
            )
            + (
                " DiT --dit-memory-refs also on (Ref2VA stand-in)."
                if dit_memory_refs
                else ""
            )
            if memory_on
            else "off"
        ),
    }
    (out / "continue_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    if prompts:
        print(f"prompts (log): {len(prompts)} block(s)", flush=True)

    committed_v: List[np.ndarray] = []
    committed_a: List[np.ndarray] = []
    produced = 0
    prev_tail_u8: Optional[np.ndarray] = None  # [overlap,H,W,3] uint8
    exp_h, exp_w = canvas_h // 16, canvas_w // 16
    mem_dir = out / "memory"
    if memory_on:
        mem_dir.mkdir(parents=True, exist_ok=True)
    # Multishot bank: persistent anchor + recent shot-ends (PNG for TE, latent for DiT).
    anchor_latent: Optional[Path] = None
    anchor_png: Optional[Path] = None
    history_latents: List[Path] = []
    history_pngs: List[Path] = []

    # Optional start image → window-1 first-frame latent (+ optional anchor)
    start_latent: Optional[Path] = None
    if args.start_image is not None:
        print(f"encoding start image {args.start_image}…", flush=True)
        rgb = fit_image_rgb(load_image_rgb(args.start_image), canvas_w, canvas_h)
        m11 = (rgb.astype(np.float32) / 127.5) - 1.0
        z0 = encode_pixels_m11(
            m11,
            width=None,
            height=None,
            device=args.device,
            encode_seed=args.encode_seed,
        )
        start_latent = out / "start_first_frame.safetensors"
        save_keyframe_latent(z0, start_latent)
        print(f"start latent {tuple(z0.shape)} → {start_latent}", flush=True)
        if memory_on and anchor_frames > 0:
            anchor_latent = mem_dir / "anchor.safetensors"
            shutil.copy2(start_latent, anchor_latent)
            u8_start = np.clip((m11 + 1.0) * 127.5, 0, 255).astype(np.uint8)
            anchor_png = save_u8_png(u8_start, mem_dir / "anchor.png")
            print(f"memory anchor set from start image → {anchor_latent} + {anchor_png}", flush=True)

    t_all = time.perf_counter()
    for w in range(1, num_windows + 1):
        seg_dir = out / f"seg_{w:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed if args.seed_mode == "same" else args.seed + (w - 1)
        prompt_w = prompts[min(w - 1, len(prompts) - 1)] if prompts else ""

        need = total_target - produced
        if w == 1:
            frame_num = window if num_windows > 1 else snap_frames(min(window, total_target))
            if num_windows == 1:
                frame_num = snap_frames(total_target)
        else:
            # generate enough that after drop we cover remaining
            frame_num = snap_frames(min(window, need + overlap))

        # Build first-frame KEYFRAME for this window (seam lock — always last RGB).
        first_path: Optional[Path] = None
        if w == 1:
            first_path = start_latent
        else:
            assert prev_tail_u8 is not None
            first_path = encode_u8_frame_latent(
                prev_tail_u8[-1],
                dest=seg_dir / "first_frame_latent.safetensors",
                device=args.device,
                encode_seed=args.encode_seed,
                exp_h=exp_h,
                exp_w=exp_w,
            )
            print(f"win{w} keyframe from prev tail → {first_path}", flush=True)

        # Multishot MEMORY bank → TE vision images (primary) and/or DiT refs.
        mem_pngs: List[Path] = []
        if memory_on:
            if anchor_png is not None and anchor_frames > 0:
                mem_pngs.append(anchor_png)
            take = memory_frames if memory_frames > 0 else 0
            if take > 0 and history_pngs:
                mem_pngs.extend(history_pngs[-take:])
            mem_pngs = mem_pngs[:9]

        ref_images: List[Path] = []
        if dit_memory_refs:
            if anchor_latent is not None and anchor_frames > 0:
                ref_images.append(anchor_latent)
            take = memory_frames if memory_frames > 0 else (1 if history_latents else 0)
            if take > 0 and history_latents:
                ref_images.extend(history_latents[-take:])
            ref_images = ref_images[:9]

        # Text cond: TE vision re-encode when memory images exist; else static --text.
        if te_memory:
            text_path = seg_dir / "text_memory.safetensors"
            print(
                f"win{w} TE vision encode prompt + {len(mem_pngs)} memory image(s)…",
                flush=True,
            )
            encode_text_with_memory(
                prompt=prompt_w,
                images=mem_pngs,
                dest=text_path,
                device=args.device,
            )
        else:
            text_path = texts[min(w - 1, len(texts) - 1)]

        print(
            f"=== window {w}/{num_windows} frames={frame_num} seed={seed} "
            f"text={text_path.name} first={first_path} "
            f"te_mem_imgs={len(mem_pngs)} dit_refs={len(ref_images)} ===",
            flush=True,
        )
        t_w = time.perf_counter()
        sample_window(
            weights=args.weights,
            text=text_path,
            out_dir=seg_dir / "latents",
            steps=args.steps,
            width=args.width,
            height=args.height,
            length=frame_num,
            seed=seed,
            first_frame=first_path,
            device=args.device,
            attn=args.attn,
            sage_backend=args.sage_backend,
            extra_sample_args=args.extra_sample_arg,
            ref_images=ref_images or None,
            allow_keyframe_refs=dit_memory_refs,
        )
        u8, wav = decode_window(
            seg_dir / "latents",
            device=args.device,
            frames=frame_num,
            fps=FPS,
        )
        print(
            f"win{w} decoded video {u8.shape} audio {wav.shape} "
            f"in {time.perf_counter() - t_w:.1f}s",
            flush=True,
        )

        # next tail from FULL window before drop
        prev_tail_u8 = u8[-overlap:].copy()

        # Update Multishot bank from this window's pixels (full window).
        if memory_on:
            if anchor_latent is None and anchor_frames > 0:
                anchor_latent = encode_u8_frame_latent(
                    u8[0],
                    dest=mem_dir / "anchor.safetensors",
                    device=args.device,
                    encode_seed=args.encode_seed,
                    exp_h=exp_h,
                    exp_w=exp_w,
                )
                anchor_png = save_u8_png(u8[0], mem_dir / "anchor.png")
                print(f"memory anchor set from win{w} frame 0 → {anchor_latent}", flush=True)
            end_path = encode_u8_frame_latent(
                u8[-1],
                dest=mem_dir / f"end_w{w:02d}.safetensors",
                device=args.device,
                encode_seed=args.encode_seed,
                exp_h=exp_h,
                exp_w=exp_w,
            )
            end_png = save_u8_png(u8[-1], mem_dir / f"end_w{w:02d}.png")
            history_latents.append(end_path)
            history_pngs.append(end_png)
            # Multishot keeps a short rolling history (cap 8).
            if len(history_latents) > 8:
                history_latents = history_latents[-8:]
                history_pngs = history_pngs[-8:]
            (mem_dir / "bank.json").write_text(
                json.dumps(
                    {
                        "anchor": str(anchor_latent) if anchor_latent else None,
                        "anchor_png": str(anchor_png) if anchor_png else None,
                        "history": [str(p) for p in history_latents],
                        "history_pngs": [str(p) for p in history_pngs],
                        "memory_frames": memory_frames,
                        "anchor_frames": anchor_frames,
                        "te_memory": te_memory,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        if w == 1:
            commit_v = u8
            commit_a = wav
        else:
            commit_v = u8[overlap:]
            drop_a = int(round(overlap * SR / float(FPS)))
            commit_a = wav[drop_a:]

        if commit_v.shape[0] == 0:
            raise RuntimeError(f"win{w}: nothing to commit after drop")

        # trim if we overshoot total_target
        remain = total_target - produced
        if commit_v.shape[0] > remain:
            commit_v = commit_v[:remain]
            commit_a = commit_a[: int(round(remain * SR / float(FPS)))]

        committed_v.append(commit_v)
        committed_a.append(commit_a)
        produced += commit_v.shape[0]

        write_seg_media(seg_dir / "decode", commit_v, commit_a, fps=FPS)
        # also stash full-window decode for seam debug
        write_seg_media(seg_dir / "decode_full", u8, wav, fps=FPS)
        print(f"win{w} committed {commit_v.shape[0]} frames (total {produced})", flush=True)

        if produced >= total_target:
            break

    if not committed_v:
        print("FAIL: no frames", file=sys.stderr)
        return 1

    final_v = np.concatenate(committed_v, axis=0)
    final_a = np.concatenate(committed_a, axis=0)
    # final audio length sync
    target_a = int(round(final_v.shape[0] * SR / float(FPS)))
    if final_a.shape[0] > target_a:
        final_a = final_a[:target_a]
    elif final_a.shape[0] < target_a:
        pad = np.zeros((target_a - final_a.shape[0], final_a.shape[1]), dtype=np.float32)
        final_a = np.concatenate([final_a, pad], axis=0)

    final_dir = out / "final"
    write_seg_media(final_dir, final_v, final_a, fps=FPS)
    # convenience copy
    src_mp4 = final_dir / "output.mp4"
    if src_mp4.is_file():
        shutil.copy2(src_mp4, out / "output.mp4")
        print(f"wrote {out / 'output.mp4'}  ({final_v.shape[0]} frames, "
              f"{final_v.shape[0] / FPS:.2f}s) in {time.perf_counter() - t_all:.1f}s", flush=True)
    else:
        print(f"wrote {final_dir / 'video.mp4'} + audio.wav (mux failed)", flush=True)

    if not args.keep_segs:
        # keep latents + final; drop bulky full-window decode copies optional — keep for A/B
        pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise
