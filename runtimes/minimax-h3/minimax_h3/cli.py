"""Command-line interface for the PyTorch reference implementation.

Mirrors the Rust binary's commands (see src/main.rs): ``audit``,
``check-shapes``, ``infer``, ``list-params``, ``layout-test``. The VRAM /
offload / auto-vram flags are Candle-runtime concerns and have no PyTorch
equivalent here.

Usage:
    python -m minimax_h3 --tiny check-shapes
    python -m minimax_h3 --tiny bench  (runs the tiny forward + kernel checks)
    python -m minimax_h3 layout-test
    python -m minimax_h3 audit model.safetensors
    python -m minimax_h3 --quantize infer model.safetensors video.latent audio.latent text.latent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .attention import ATTN_CHOICES, AttnImpl, SAGE_BACKENDS, get_sage_backend, set_sage_backend
from .canvas import (
    ASPECT_RATIOS,
    MAX_MP,
    adapt_canvas,
    format_resolution_table,
    list_mp_ladder,
    list_official_short_edge_table,
    resolve_from_args,
)
from .conditioning import ConditioningError, load_text_embeds
from .config import H3Config
from .layout import Keyframe, PackedLayout, RefBlock, RefKind
from .model import MiniMaxH3Model, Payload
from .sample import SampleError, SampleRequest, run_sample
from .weights import (
    AdalnSvdSource,
    QuantizingSource,
    RandomSource,
    SafetensorsSource,
    expected_params,
    fmt_footprint,
    save_safetensors,
    total_params,
)


def build_source(cfg: H3Config, args, compute_dtype: torch.dtype, weight_dev="cpu"):
    """Build the (possibly wrapped) TensorSource chain the model reads from,
    mirroring the Rust CLI: RandomSource / SafetensorsSource → QuantizingSource
    → AdalnSvdSource."""
    weights_path = getattr(args, "weights", None)
    if weights_path is not None:
        inner = SafetensorsSource(weights_path, weight_dev, compute_dtype)
    else:
        inner = RandomSource(weight_dev, compute_dtype, seed=42)
    q = QuantizingSource(inner) if args.quantize else None
    qref = q if q is not None else inner
    s = AdalnSvdSource(qref, cfg) if cfg.adaln_svd_rank else None
    return s if s is not None else qref, inner


def load_first_tensor_f32(path: Path, device="cpu") -> torch.Tensor:
    """CLI alias for :func:`minimax_h3.conditioning.load_text_embeds`."""
    return load_text_embeds(path, device=device)


class _AppendTypedRef(argparse.Action):
    """Preserve interleaved ``--ref-*`` order on ``args.ref_specs``."""

    def __call__(self, parser, namespace, values, option_string=None):
        kind = {
            "--ref-image": "image",
            "--ref-video": "video",
            "--ref-audio": "audio",
            "--ref-av": "av",
        }.get(option_string or "", self.dest)
        specs = getattr(namespace, "ref_specs", None)
        if specs is None:
            specs = []
            setattr(namespace, "ref_specs", specs)
        specs.append((kind, values))


def all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


# ---------------------------------------------------------------------------


def cmd_check_shapes(cfg: H3Config, args) -> int:
    device = args.device
    compute = torch.bfloat16 if device == "cuda" else torch.float32
    src, _ = build_source(cfg, args, compute)
    model = MiniMaxH3Model(src, cfg, args.attn)
    model.to(device)

    video = torch.randn(1, cfg.latents_dim, 8, 8, 8, device=device)
    audio = torch.randn(1, cfg.audio_latents_dim, 2, 6, device=device)
    text = torch.randn(16, cfg.text_dim, device=device)
    payload = Payload(
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    with torch.no_grad():
        video_out, audio_out = model.forward(video, audio, text, None, 0.5, payload)
    vshape = list(video_out.shape)
    ashape = list(audio_out.shape)
    print(f"video_out {vshape} (expect [1, {cfg.latents_dim}, 8, 8, 8])")
    print(f"audio_out {ashape} (expect [1, {cfg.audio_latents_dim}, 2, 6])")
    print(f"video_out finite: {all_finite(video_out)}")
    print(f"audio_out finite: {all_finite(audio_out)}")
    if vshape != [1, cfg.latents_dim, 8, 8, 8] or ashape != [1, cfg.audio_latents_dim, 2, 6]:
        print("FAIL: unexpected output shape", file=sys.stderr)
        return 1
    print("OK: forward pass runs end to end")
    return 0


def cmd_layout_test(_cfg: H3Config, _args) -> int:
    # t2va
    l = PackedLayout.new(16, 32, 42, 24, 8)
    dump_layout("t2va", l)
    # fl2va: first + last keyframe anchors
    kfs = [Keyframe(0), Keyframe(123)]
    l = PackedLayout.new(16, 32, 42, 24, 8, kfs, [], 124)
    dump_layout("fl2va", l)
    # ref2va
    refs = [
        RefBlock(RefKind.IMAGE, 0, 16, 24, 0),
        RefBlock(RefKind.VIDEO_AUDIO, 5, 16, 24, 3),
        RefBlock(RefKind.AUDIO, 0, 0, 0, 4),
    ]
    l = PackedLayout.new(16, 32, 42, 24, 8, [], refs, None)
    dump_layout("ref2va", l)
    return 0


def dump_layout(name: str, l: PackedLayout) -> None:
    print(f"=== {name} ===")
    print(f"seq_len {l.seq_len}")
    for a, b, k in l.segments:
        print(f"  segment {k.name():>8} [{a:>5}, {b:>5}) n={b - a}")


def cmd_list_params(cfg: H3Config, args) -> int:
    params = expected_params(cfg)
    count = total_params(cfg)
    for name, shape, dtype in params:
        n = 1
        for s in shape:
            n *= s
        print(f"{name:>60} {shape} {dtype} ({n} elems)")
    print(f"\ntotal params: {count} ({count / 1e9:.2f} B)")
    print(f"fp16: {count * 2.0 / 1e9:.2f} GB | fp8/int8: {count / 1e9:.2f} GB | int4: {count * 0.5 / 1e9:.2f} GB")
    print(f"fp8 quantized (--quantize): {fmt_footprint(cfg)}")
    return 0


def cmd_audit(cfg: H3Config, args) -> int:
    # Probe-first, mirroring the Rust audit: read the header first, then drive
    # the expected contract from the format probe exactly like the model
    # (`resolve_curve_grid`): a checkpoint that ships `adaln_t_table` is
    # audited in curve mode with its stored grid, and a pre-projected (rank-r)
    # table — the pruned int8 checkpoints ship `[grid, rank]` — narrows the
    # expected adaln/table widths to rank too. `--curve-grid auto` (or even no
    # flag) thus audits curve-format checkpoints correctly.
    from safetensors import safe_open
    from .weights import strip_prefix, is_companion_key

    with safe_open(str(args.weights), framework="pt", device="cpu") as sf:
        file_meta = {
            strip_prefix(k): tuple(sf.get_slice(k).get_shape()) for k in sf.keys()
        }

    cfg = H3Config(**{**cfg.__dict__}) if hasattr(cfg, "__dict__") else cfg
    adaln_width = None
    table_shape = file_meta.get("adaln_t_table")
    if table_shape is not None:
        cfg.adaln_curve_grid = table_shape[0]
        if len(table_shape) == 2 and table_shape[1] < cfg.time_embed_dim:
            adaln_width = table_shape[1]

    expected = []
    for n, s, _dt in expected_params(cfg):
        if adaln_width is not None and (n.endswith("adaln_proj.linear.weight") or n == "adaln_t_table"):
            s = (s[0], adaln_width)
        expected.append((n, tuple(s)))
    expected = set(expected)

    file_keys = file_meta

    missing = [n for (n, s) in sorted(expected) if n not in file_keys]
    shape_mismatch = [
        (n, file_keys[n], s)
        for (n, s) in sorted(expected)
        if n in file_keys and tuple(file_keys[n]) != tuple(s)
    ]
    unused = sorted(k for k in file_keys if k not in {n for (n, _) in expected} and not is_companion_key(k))

    print(f"expected {len(expected)} params, {len(expected) - len(missing) - len(shape_mismatch)} matched")
    print(f"missing ({len(missing)}):")
    for m in missing:
        print(f"  {m}")
    print(f"shape mismatches ({len(shape_mismatch)}):")
    for n, got, want in shape_mismatch:
        print(f"  {n}: file {got} vs expected {want}")
    print(f"unused file keys ({len(unused)}):")
    for u in unused:
        print(f"  {u}")
    if not missing and not shape_mismatch:
        print(f"OK: checkpoint key table matches the port ({len(file_keys)} tensors)")
        return 0
    print("FAIL: checkpoint does not match the expected parameter table", file=sys.stderr)
    return 1


def cmd_infer(cfg: H3Config, args) -> int:
    device = args.device
    compute = torch.bfloat16 if device == "cuda" else torch.float32
    video = load_first_tensor_f32(args.video, device)
    audio = load_first_tensor_f32(args.audio, device)
    text = load_first_tensor_f32(args.text, device)
    print(f"video {tuple(video.shape)} audio {tuple(audio.shape)} text {tuple(text.shape)}")

    src, inner = build_source(cfg, args, compute, weight_dev="cpu" if args.offload else device)
    model = MiniMaxH3Model(src, cfg, args.attn)
    stream_gpu = device if (args.offload and device == "cuda") else None
    if not args.offload:
        model.to(device)

    missing, shape_mismatch, unused = inner.report()
    if missing:
        print(f"FAIL: checkpoint missing tensors: {missing}", file=sys.stderr)
        return 1
    if shape_mismatch:
        print(f"FAIL: checkpoint shape mismatches: {shape_mismatch}", file=sys.stderr)
        return 1
    if unused:
        print(f"note: unused checkpoint keys: {unused}", file=sys.stderr)

    payload = Payload(
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )
    with torch.no_grad():
        video_out, audio_out = model.forward(video, audio, text, None, args.sigma, payload, stream_gpu=stream_gpu)
    print(f"video_out {tuple(video_out.shape)} audio_out {tuple(audio_out.shape)}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_safetensors(Path(args.out) / "video_out.safetensors", [("video_out", video_out.detach().cpu().contiguous())])
    save_safetensors(Path(args.out) / "audio_out.safetensors", [("audio_out", audio_out.detach().cpu().contiguous())])
    print(f"wrote {Path(args.out) / 'video_out.safetensors'}")
    print(f"wrote {Path(args.out) / 'audio_out.safetensors'}")
    return 0


def cmd_canvas_table(_cfg: H3Config, args) -> int:
    """Print official short-edge + MP ladder tables (all aspects, Base-clamped)."""
    aspects = list(args.aspects) if getattr(args, "aspects", None) else None
    print(format_resolution_table(list_official_short_edge_table(aspects), title="=== official short-edge=768 ==="))
    print()
    aspect = getattr(args, "ladder_aspect", None) or "16:9"
    print(
        format_resolution_table(
            list_mp_ladder(aspect, clamp_base=not bool(getattr(args, "no_clamp", False))),
            title=f"=== MP ladder aspect={aspect} (clamp_base={not bool(getattr(args, 'no_clamp', False))}) ===",
        )
    )
    print()
    print("named aspects:", ", ".join(sorted(ASPECT_RATIOS, key=lambda k: ASPECT_RATIOS[k], reverse=True)))
    print(f"Base max ~{MAX_MP:.3f} MP — rows above that need H3-Regenerate-2K (closed) or post SR")
    if getattr(args, "resolve", False) or any(
        getattr(args, k, None) is not None for k in ("aspect", "megapixels", "short_edge")
    ):
        # Also print one resolved canvas from the same flags as sample
        try:
            spec = resolve_from_args(args)
            print()
            print("=== resolve ===")
            print(spec.as_dict())
        except Exception as e:
            print(f"resolve skip: {e}")
    return 0


def cmd_sample(cfg: H3Config, args) -> int:
    """Full Euler sample via :func:`minimax_h3.sample.run_sample`.

    Thin CLI adapter — packing, pin, denoise, and latent I/O live in the
    sample pipeline module (gemmy / scripts should call that, not re-fork here).
    """
    try:
        run_sample(
            SampleRequest(
                weights=args.weights,
                text=args.text,
                out=args.out,
                cfg=cfg,
                steps=int(args.steps),
                length=int(args.length),
                seed=int(args.seed),
                device=str(args.device),
                attn=AttnImpl.normalize(args.attn),
                canvas=resolve_from_args(args),
                first_frame=getattr(args, "first_frame", None),
                last_frame=getattr(args, "last_frame", None),
                ref_specs=list(getattr(args, "ref_specs", None) or []),
                allow_keyframe_refs=bool(getattr(args, "allow_keyframe_refs", False)),
                prefetch=int(getattr(args, "prefetch", 1)),
                precompute_adaln=bool(getattr(args, "precompute_adaln", True)),
                preproject_text=bool(getattr(args, "preproject_text", True)),
                ram_reserve_gb=float(getattr(args, "ram_reserve_gb", 8.0)),
                max_pin_gb=float(getattr(args, "max_pin_gb", 4.0)),
                save_every=int(args.save_every),
                profile=bool(getattr(args, "profile", False)),
                profile_json=getattr(args, "profile_json", None),
                log=True,
            )
        )
    except (SampleError, ConditioningError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_bench(cfg: H3Config, args) -> int:
    device = args.device
    compute = torch.bfloat16 if device == "cuda" else torch.float32
    src, _ = build_source(cfg, args, compute)
    model = MiniMaxH3Model(src, cfg, args.attn)
    model.to(device)

    video = torch.randn(1, cfg.latents_dim, 8, 8, 8, device=device)
    audio = torch.randn(1, cfg.audio_latents_dim, 2, 6, device=device)
    text = torch.randn(16, cfg.text_dim, device=device)
    payload = Payload(
        sigma_shift_video=cfg.sigma_shift_video,
        sigma_shift_audio=cfg.sigma_shift_audio,
    )

    def step():
        with torch.no_grad():
            return model.forward(video, audio, text, None, 0.5, payload)

    step()  # warmup (layout/rope caches)
    n = max(args.iters, 1)
    import time

    t0 = time.perf_counter()
    for _ in range(n):
        step()
    dt = (time.perf_counter() - t0) / n * 1e3
    print(f"full forward {dt:9.2f} ms/step ({cfg.num_layers} layers, attn {args.attn})")

    # attention-vs-MLP microbenchmarks at a controlled sequence length
    from .attention import Attention
    from .blocks import MLP

    hidden = cfg.hidden_size
    attn_eager = Attention(src, "bench.attn", hidden, cfg.num_attention_heads, cfg.attention_head_dim, cfg.qk_norm_eps, compute, AttnImpl.EAGER)
    attn_flash = Attention(src, "bench.attn", hidden, cfg.num_attention_heads, cfg.attention_head_dim, cfg.qk_norm_eps, compute, AttnImpl.FLASH)
    mlp = MLP(src, "bench.mlp", hidden, cfg.ffn_hidden_size, compute)
    if device == "cuda":
        attn_eager.to(device)
        attn_flash.to(device)
        mlp.to(device)
    x = torch.randn(args.seq, hidden, device=device).to(compute)

    def timeit(f, warmup=2):
        for _ in range(warmup):
            with torch.no_grad():
                f()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                f()
        return (time.perf_counter() - t0) / n * 1e3

    t_eager = timeit(lambda: attn_eager.forward(x, None, None))
    t_flash = timeit(lambda: attn_flash.forward(x, None, None))
    t_mlp = timeit(lambda: mlp.forward(x))
    print(f"attention (eager)   {t_eager:9.2f} ms @ seq {args.seq}")
    print(f"attention (flash)   {t_flash:9.2f} ms @ seq {args.seq}")
    print(f"mlp                 {t_mlp:9.2f} ms @ seq {args.seq}")
    print(f"attn/mlp  eager     {t_eager / max(t_mlp, 1e-9):9.2f}x")
    print(f"attn/mlp  flash     {t_flash / max(t_mlp, 1e-9):9.2f}x")
    print(f"flash vs eager      {t_flash / max(t_eager, 1e-9):9.2f}x")
    return 0


# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="minimax_h3",
        description="PyTorch reference implementation of the MiniMax H3 audio-video DiT (ComfyUI PR #15224)",
    )
    p.add_argument("--config", type=Path, default=None, help="architecture JSON (defaults to the real H3 numbers)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--tiny", action="store_true", help="smoke-test dims, small enough for CPU")
    p.add_argument("--curve-grid", default=None, help="AdaLN curve-table mode: a number forces that grid (overrides the config's adaln_curve_grid); 'auto' probes the checkpoint's stored adaln_t_table grid (a curve-format checkpoint loads correctly even without the flag)")
    p.add_argument("--adaln-svd-rank", type=int, default=None, help="load-time SVD pruning of adaln projections to RANK")
    p.add_argument("--quantize", action="store_true", help="fp8 (e4m3) weight quantization at load time")
    p.add_argument(
        "--attn",
        choices=list(ATTN_CHOICES),
        default="sage",
        help=(
            "attention kernel (default sage = fast/approx production). "
            "fa2 / flash-attn = real FlashAttention-2 HQ path (needs flash-attn wheel). "
            "flash = portable chunked online-softmax fallback (not Dao FA). "
            "sdpa/eager/auto = debug / short-seq."
        ),
    )
    p.add_argument(
        "--sage-backend",
        choices=list(SAGE_BACKENDS),
        default="auto",
        help=(
            "dense SageAttention kernel when --attn sage (default auto). "
            "auto=package arch dispatcher (SA2 sm120→fp8); "
            "fp8/fp8pp=explicit SA2 fp8 CUDA; fp16_cuda/fp16_triton=SA2; "
            "sa1=Triton SA1-style entry if present. Needs SA2 CUDA wheel on Windows."
        ),
    )
    p.add_argument("--offload", action="store_true", help="keep weights on CPU (reference stand-in for the Rust streaming path)")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-shapes", help="random-weight forward pass; validates shapes end to end")
    sub.add_parser("layout-test", help="print packed layouts for t2va / fl2va / ref2va")
    sub.add_parser("list-params", help="print the expected parameter table + totals")
    a = sub.add_parser("audit", help="check a checkpoint against the expected parameter table")
    a.add_argument("weights", type=Path)
    i = sub.add_parser("infer", help="one denoising step with real weights")
    i.add_argument("weights", type=Path)
    i.add_argument("video", type=Path)
    i.add_argument("audio", type=Path)
    i.add_argument("text", type=Path)
    i.add_argument("--sigma", type=float, default=0.5)
    i.add_argument("--out", type=Path, default=Path("out"))
    b = sub.add_parser("bench", help="random-weight timing: full forward + attention/MLP microbenchmarks")
    b.add_argument("--iters", type=int, default=3)
    b.add_argument("--seq", type=int, default=8192)
    ct = sub.add_parser(
        "canvas-table",
        help="print official short-edge + megapixel resolution tables (all aspects)",
    )
    ct.add_argument(
        "--aspects",
        nargs="*",
        default=None,
        help="aspects for the official short-edge table (default: 21:9 16:9 4:3 1:1 3:4 9:16)",
    )
    ct.add_argument(
        "--ladder-aspect",
        default="16:9",
        help="aspect for the MP ladder table (default 16:9; try 9:16 / 1:1 / …)",
    )
    ct.add_argument(
        "--no-clamp",
        action="store_true",
        help="show MP ladder rows above Base max without clamping (not runnable on open Base)",
    )
    ct.add_argument("--aspect", default=None, help="optional: also resolve one canvas like sample")
    ct.add_argument("--megapixels", "--mp", type=float, default=None, dest="megapixels")
    ct.add_argument("--short-edge", type=int, default=None)
    ct.add_argument("--width", type=int, default=None)
    ct.add_argument("--height", type=int, default=None)
    ct.add_argument(
        "--canvas-mode",
        choices=["preview", "official", "raw"],
        default="preview",
    )
    ct.add_argument("--resolve", action="store_true", help="print resolve block even without preset flags")

    s = sub.add_parser("sample", help="full Euler denoising loop on the real checkpoint (SageAttention + streamed int8)")
    s.add_argument("weights", type=Path)
    s.add_argument("text", type=Path)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument(
        "--width",
        type=int,
        default=None,
        help="canvas width before adapt (default with --height: official 1344×768 if no preset)",
    )
    s.add_argument(
        "--height",
        type=int,
        default=None,
        help="canvas height before adapt (default with --width: official 1344×768 if no preset)",
    )
    s.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="aspect preset (16:9, 9:16, 1:1, 21:9, …). Use with --megapixels or --short-edge",
    )
    s.add_argument(
        "--megapixels",
        "--mp",
        type=float,
        default=None,
        dest="megapixels",
        help=f"target MP with --aspect (clamped to Base max ~{MAX_MP:.3f})",
    )
    s.add_argument(
        "--short-edge",
        type=int,
        default=None,
        help="short edge with --aspect (official native uses 768)",
    )
    s.add_argument(
        "--canvas-mode",
        choices=["preview", "official", "raw"],
        default="preview",
        help="preview=allow <768 short edge; official=force 768 short; raw=×32+cap only",
    )
    s.add_argument("--length", type=int, default=124, help="output frames at 24fps (snaps to the 17k+5 grid)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--save-every", type=int, default=0)
    s.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "sample_out",
        help="output directory for latents/profile (default: outputs/sample_out; gitignored)",
    )
    s.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help=(
            "how many DiT layers to keep on the GPU at once while streaming "
            "(0=sync one-by-one, 1=current+load next, 5/10=keep more hot; default 1). "
            "Higher uses more VRAM; may or may not be faster once transfer is hidden."
        ),
    )
    s.add_argument(
        "--precompute-adaln",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lossless per-step AdaLN precompute (default on for cuda sample)",
    )
    s.add_argument(
        "--preproject-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lossless: run condition_proj+token refiner once before the Euler loop (default on)",
    )
    s.add_argument(
        "--ram-reserve-gb",
        type=float,
        default=8.0,
        help=(
            "minimum free system RAM to keep while pinning (default 8). "
            "Stops pinning before the desktop is starved. Does not change quality."
        ),
    )
    s.add_argument(
        "--max-pin-gb",
        type=float,
        default=4.0,
        help=(
            "max GB of weight tensors to lock (pin) in physical RAM for faster "
            "GPU upload (default 4). 0 = pin nothing. Negative = no byte cap "
            "(only --ram-reserve-gb applies). Pinning the full ~20 GB model is "
            "what drives Task Manager to ~99%% on a 64 GB box — leave the cap on."
        ),
    )
    s.add_argument("--profile", action="store_true", help="phase timers + peak VRAM for the sample run")
    s.add_argument("--profile-json", type=Path, default=None, help="write profile JSON (default: <out>/profile.json)")
    s.add_argument(
        "--first-frame",
        type=Path,
        default=None,
        help=(
            "FL2VA first-frame keyframe: DiT-normalized .safetensors latent "
            "[1,24,1,H/16,W/16] (from scripts/h3_vae_encode.py). "
            "Default T2VA when omitted. Not combinable with --ref-*."
        ),
    )
    s.add_argument(
        "--last-frame",
        type=Path,
        default=None,
        help=(
            "FL2VA last-frame keyframe: same latent contract as --first-frame. "
            "Can combine with --first-frame for first+last."
        ),
    )
    s.add_argument(
        "--ref-image",
        action=_AppendTypedRef,
        default=argparse.SUPPRESS,
        help=(
            "Ref2VA image ref latent .safetensors [1,24,1,lat_h,lat_w] "
            "(repeatable, ≤9). Encode via h3_vae_encode.py --mode image. "
            "Order preserved with other --ref-* flags. Needs Ref2VA checkpoint."
        ),
    )
    s.add_argument(
        "--ref-video",
        action=_AppendTypedRef,
        default=argparse.SUPPRESS,
        help=(
            "Ref2VA pure-video ref latent [1,24,T,lat_h,lat_w] "
            "(repeatable, ≤3 video slots with --ref-av). Encode --mode video."
        ),
    )
    s.add_argument(
        "--ref-audio",
        action=_AppendTypedRef,
        default=argparse.SUPPRESS,
        help=(
            "Ref2VA standalone audio ref [1,32,2,T] (repeatable, ≤3 audio total). "
            "Cannot be sole input — add image or video. Encode --mode audio."
        ),
    )
    s.add_argument(
        "--ref-av",
        action=_AppendTypedRef,
        default=argparse.SUPPRESS,
        help=(
            "Ref2VA paired video+audio: VIDEO.safetensors,AUDIO.safetensors "
            "(repeatable). Packs as VIDEO_AUDIO (audio rows then video rows)."
        ),
    )
    s.add_argument(
        "--allow-keyframe-refs",
        action="store_true",
        help=(
            "Experimental: pack FL2VA --first-frame/--last-frame together with "
            "--ref-image (Multishot-style DiT memory bank). Prefer Ref2VA "
            "weights. Not Multishot's TE vision memory (needs mmproj)."
        ),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tiny:
        cfg = H3Config.tiny()
    elif args.config is not None:
        cfg = H3Config.from_json(args.config)
    else:
        cfg = H3Config.full()
    if args.curve_grid is not None and args.curve_grid != "auto":
        cfg.adaln_curve_grid = int(args.curve_grid)
    if args.adaln_svd_rank is not None:
        cfg.adaln_svd_rank = args.adaln_svd_rank
    if args.attn == "sage":
        set_sage_backend(getattr(args, "sage_backend", "auto"))
    if args.curve_grid == "auto":
        # Leave cfg.adaln_curve_grid unset: the model's format probe
        # (ships_curve_table) drives the grid from the checkpoint, matching
        # the Rust CLI's `--curve-grid auto`.
        pass

    if args.cmd == "check-shapes":
        return cmd_check_shapes(cfg, args)
    if args.cmd == "layout-test":
        return cmd_layout_test(cfg, args)
    if args.cmd == "canvas-table":
        return cmd_canvas_table(cfg, args)
    if args.cmd == "list-params":
        return cmd_list_params(cfg, args)
    if args.cmd == "audit":
        return cmd_audit(cfg, args)
    if args.cmd == "infer":
        return cmd_infer(cfg, args)
    if args.cmd == "bench":
        return cmd_bench(cfg, args)
    if args.cmd == "sample":
        return cmd_sample(cfg, args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
