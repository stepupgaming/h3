"""Decode MiniMax H3 sampled latents through the official VAEs.

Replicates the exact decode conventions of the golden reference (ComfyUI PR
#15224, comfy/ldm/minimax/vae.py + audio_vae.py):

  video: z = z * latents_std + latents_mean            (per-channel, 24)
         decode_temporal (17k+5 grid chunking, token_drop overlap blend)
         dec = dec * pixel_std + pixel_mean, clamp(0,1), *2 - 1   (imagenet)
  audio: z = z.permute(0,2,1,3).reshape(2, 32, T)      (stereo split)
         z = z * latents_std + latents_mean            (per-channel, 32)
         dec_in_proj -> BigVGAN -> [2, 1, L], clamp[-1,1], reshape [1, 2, L]

Weights: checkpoints/vae/minimax_h3_video_vae_fp16.safetensors (video, fp16)
         checkpoints/vae/minimax_h3_audio_vae_fp32.safetensors (audio, fp32)
Model modules: the official bundle under MiniMax-H3-Official/FL2VA/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parent.parent


def _checkpoints_root() -> Path:
    """External weights tree (Gemmy) or ROOT/checkpoints (standalone H3 repo)."""
    import os
    env = (os.environ.get("GEMMY_H3_CHECKPOINTS") or "").strip()
    if env:
        return Path(env)
    return ROOT / "checkpoints"


# Import the official bundle as packages so its internal relative imports
# (video_vae.parallel, audio_vae.dac_*...) resolve.
FL2VA = ROOT / "MiniMax-H3-Official" / "FL2VA"
sys.path.insert(0, str(FL2VA))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# video VAE
# ---------------------------------------------------------------------------

def load_video_vae_state_dict(weights: Path) -> dict:
    """Load a video VAE checkpoint. Kijai's int8_convrot VAE (I8 linears +
    ``weight_scale`` + ``comfy_quant`` companions, decoder transformer blocks
    only) is dequantized to fp16 at load via the package's ConvRot decode.
    ComfyUI's ~1.5x int8 VAE claim rides their CUTLASS int8 GEMM, which loses
    to cast on owner sm120 — so this is a disk/load-time win, not a runtime
    speed path."""
    from minimax_h3.weights import dequantize_convrot_gs

    sd = safetensors.torch.load_file(str(weights))
    int8_names = [
        k
        for k, t in sd.items()
        if t.dtype == torch.int8
        and k.endswith(".weight")
        and (k[: -len(".weight")] + ".weight_scale") in sd
    ]
    for name in int8_names:
        base = name[: -len(".weight")]
        q = sd.pop(name)
        scale = sd.pop(base + ".weight_scale")
        blob = sd.pop(base + ".comfy_quant", None)
        gs = None
        if blob is not None:
            meta = json.loads(bytes(blob.cpu().numpy().tolist()).decode("utf-8"))
            if meta.get("convrot", False):
                gs = int(meta.get("convrot_groupsize", 0)) or None
        if gs is None:
            w = q.float() * scale.float()
        else:
            w = dequantize_convrot_gs(q, scale, gs)
        sd[name] = w.to(torch.float16)
    return sd


def build_video_vae(device: str, weights: Path | None = None) -> torch.nn.Module:
    from video_vae.klvae import AutoencoderKLLegacy
    from video_vae.parallel import get_parallel_state

    state = get_parallel_state()
    if not state:
        state.update({
            "group_size": 1, "group_rank": 0, "local_process_group": None,
            "sp_size": 1, "sp_rank": 0, "sp_enabled": False,
            "sp_process_group": None, "tp_size": 1, "tp_rank": 0,
        })

    source_dir = ROOT / "MiniMax-H3-Official" / "FL2VA" / "video_vae" / "source"
    source_config = AutoencoderKLLegacy.load_config(str(source_dir))
    model, _ = AutoencoderKLLegacy.from_config(
        source_config, return_unused_kwargs=True,
        clip_length=17, token_drop=3, encoder_tiling=1, decoder_tiling=1,
        parallel_tiling=1, tile_size=256, tile_overlap_min=64,
        encoder_parallel=0, decoder_parallel=0, chunk_dim=-1,
    )
    if weights is None:
        weights = _checkpoints_root() / "vae" / "minimax_h3_video_vae_fp16.safetensors"
    state_dict = load_video_vae_state_dict(weights)
    # latents_mean/std are baked into the checkpoint as buffers but are not
    # declared on AutoencoderKLLegacy; decode applies them from the wrapper
    # config (identical values), so drop them before the strict load.
    state_dict.pop("latents_mean", None)
    state_dict.pop("latents_std", None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device == "cuda":
        model.to("cuda", dtype=torch.float16)
    else:
        model.to("cpu", dtype=torch.float16)
    return model


def decode_video(vae, z: torch.Tensor, frame_num: int) -> torch.Tensor:
    """z: [1, 24, T, H, W] normalized -> pixels [1, 3, T, H, W] in [-1, 1]."""
    config = json.loads(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "video_vae" / "config.json").read_text()
    )
    dtype = next(vae.parameters()).dtype  # fp16 on cuda, fp32 on cpu
    z = z.to(device=next(vae.parameters()).device, dtype=dtype)
    mean = torch.tensor(config["latents_mean"], dtype=dtype, device=z.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(config["latents_std"], dtype=dtype, device=z.device).view(1, -1, 1, 1, 1)
    z = z * std + mean

    with torch.no_grad():
        dec = vae.decode_base(z, frame_num=frame_num)

    dec = dec.float()
    pmean = torch.tensor(IMAGENET_MEAN, dtype=dec.dtype, device=dec.device).view(1, 3, 1, 1, 1)
    pstd = torch.tensor(IMAGENET_STD, dtype=dec.dtype, device=dec.device).view(1, 3, 1, 1, 1)
    dec = dec * pstd + pmean
    dec = dec.clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)  # [-1, 1]
    return dec


# ---------------------------------------------------------------------------
# audio VAE
# ---------------------------------------------------------------------------

def build_audio_vae(device: str) -> torch.nn.Module:
    # The official bundle declares WNConv1d via torch weight_norm, but the
    # fp32 checkpoint ships weight-norm *folded* into plain conv weights (per
    # the Comfy reference: "Weight-norm parametrizations are folded into plain
    # conv weights"). Make weight_norm a pass-through so strict loading matches.
    import torch.nn.utils.parametrizations as _parametrize

    if not getattr(_parametrize, "_h3_patched", False):
        _parametrize.weight_norm = lambda module, *a, **k: module
        _parametrize._h3_patched = True

    from audio_vae.dac_audio_vae import DacAudioVAE
    import yaml

    config = json.loads(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "audio_vae" / "config.json").read_text()
    )
    metadata = json.loads(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "audio_vae" / "metadata.json").read_text()
    )["metadata"]["kwargs"]
    yaml_cfg = yaml.safe_load(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "audio_vae" / "config.yaml").read_text()
    )["model_config"]

    model = DacAudioVAE(
        encoder_rates=metadata["encoder_rates"],
        decoder_rates=metadata["decoder_rates"],
        attn_proj=metadata["attn_proj"],
        decoder_type=metadata["decoder_type"],
        decoder_dim=yaml_cfg["decoder_dim"],
        vae_latent_channels=yaml_cfg["vae_latent_channels"],
        sample_rate=metadata["sample_rate"],
    )
    weights = _checkpoints_root() / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
    state_dict = safetensors.torch.load_file(str(weights))
    # Same baked-in buffers as the video VAE; decode applies them from config.
    state_dict.pop("latents_mean", None)
    state_dict.pop("latents_std", None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def decode_audio(vae, z: torch.Tensor) -> torch.Tensor:
    """z: [1, 32, 2, T] normalized -> stereo [1, 2, L] at 32 kHz in [-1, 1]."""
    config = json.loads(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "audio_vae" / "config.json").read_text()
    )
    b, c, s, t = z.shape
    z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
    mean = torch.tensor(config["latents_mean"], dtype=z.dtype, device=z.device).view(1, -1, 1)
    std = torch.tensor(config["latents_std"], dtype=z.dtype, device=z.device).view(1, -1, 1)
    z = z * std + mean
    with torch.no_grad():
        x = vae.decode(z)  # [b*s, 1, L], clamped to [-1, 1]
    return x.reshape(b, s, -1)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def save_mp4(frames: np.ndarray, out_path: Path, fps: int = 24) -> None:
    """frames: [T, H, W, 3] uint8 -> H.264 mp4 via imageio-ffmpeg."""
    import imageio.v2 as imageio
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264",
                     quality=8, macro_block_size=2)


def write_png_sequence(
    frames: np.ndarray,
    dest_dir: Path,
    *,
    start_number: int = 1,
) -> list[Path]:
    """Lossless RGB PNG dump of VAE-decoded uint8 frames (before libx264)."""
    from PIL import Image

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, fr in enumerate(frames):
        path = dest / f"frame_{i + start_number:04d}.png"
        Image.fromarray(np.ascontiguousarray(fr, dtype=np.uint8), mode="RGB").save(path)
        written.append(path)
    return written


def _ffmpeg_exe() -> str:
    """Prefer PATH ffmpeg, else imageio-ffmpeg's bundled binary."""
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio_ffmpeg is unavailable; "
            "cannot mux video+audio"
        ) from e


def mux_av(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    audio_bitrate: str = "192k",
) -> Path:
    """Mux silent H.264 + WAV into one playable MP4 (AAC audio).

    Video stream is copied (no re-encode). Audio is AAC. Uses ``-shortest`` so
    a slight A/V length mismatch does not leave a trailing silent tail.
    """
    import subprocess

    ffmpeg = _ffmpeg_exe()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Decode H3 video+audio latents through the official VAEs and write "
            "a playable muxed MP4 (plus silent video.mp4 / audio.wav sidecars)."
        )
    )
    ap.add_argument("latent_dir", type=Path, help="dir with video_latent.safetensors + audio_latent.safetensors")
    ap.add_argument("--out", type=Path, default=Path("h3_vae_decode_out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=None, help="pixel frame count (default: infer from latent_t)")
    ap.add_argument(
        "--video-vae-weights",
        type=Path,
        default=None,
        help="video VAE checkpoint override (official Comfy-Org fp16 or int8_convrot under vae/)",
    )
    ap.add_argument(
        "--no-mux",
        action="store_true",
        help="skip writing output.mp4 (keep only silent video.mp4 + audio.wav)",
    )
    ap.add_argument(
        "--png-seq",
        type=Path,
        default=None,
        help="write lossless RGB PNG sequence of VAE uint8 frames (before libx264)",
    )
    ap.add_argument(
        "--no-h264",
        action="store_true",
        help="skip silent video.mp4; MP4 remains the default product unless this is set",
    )
    ap.add_argument(
        "--skip-audio",
        action="store_true",
        help="decode video only (skip audio VAE / mux). Use for chroma-key PNG dump.",
    )
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    video_z = safetensors.torch.load_file(str(args.latent_dir / "video_latent.safetensors"))["video_latent"]
    audio_path = args.latent_dir / "audio_latent.safetensors"
    audio_z = None
    if not args.skip_audio:
        if not audio_path.is_file():
            raise SystemExit(f"audio_latent.safetensors missing under {args.latent_dir} (pass --skip-audio to decode video only)")
        audio_z = safetensors.torch.load_file(str(audio_path))["audio_latent"]
        print(f"video latent {tuple(video_z.shape)}  audio latent {tuple(audio_z.shape)}")
    else:
        print(f"video latent {tuple(video_z.shape)}  (audio skipped)")

    torch.set_grad_enabled(False)

    # ---- video ----
    print(f"loading video VAE ({args.video_vae_weights or 'fp16 default'})...")
    vvae = build_video_vae(args.device, args.video_vae_weights)
    # Infer pixel frame count from latent_t on the 17k+5 / f16t4 grid used by the
    # sample path. Callers can override with --frames when needed.
    latent_t = int(video_z.shape[2])
    inferred_frames = 5 if latent_t <= 2 else ((latent_t - 2) // 5) * 17 + 5
    frame_count = args.frames if args.frames is not None else inferred_frames
    vz = video_z.to(vvae.decoder.proj_out.weight.device)
    pixels = decode_video(vvae, vz, frame_count)
    pixels = pixels.cpu()
    print(f"decoded video {tuple(pixels.shape)} in [-1, 1] (requested frames={frame_count})")

    # stats / coherence checks — use the tensor's actual T, not the request
    frames_np = pixels.squeeze(0).permute(1, 2, 3, 0).numpy()  # [T, H, W, 3] [-1,1]
    actual_t = frames_np.shape[0]
    fin = np.isfinite(frames_np).all()
    frame_mean = frames_np.mean()
    frame_std = frames_np.std()
    step = max(1, actual_t // 12)
    diff = np.abs(np.diff(frames_np.reshape(actual_t, -1)[::step], axis=0)).mean() if actual_t > 1 else 0.0
    print(f"video: finite={fin} mean={frame_mean:.4f} std={frame_std:.4f} "
          f"inter-frame diff (subsampled)={diff:.4f}")

    u8 = np.clip((frames_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if args.png_seq is not None:
        png_written = write_png_sequence(u8, Path(args.png_seq))
        print(f"wrote {len(png_written)} lossless RGB PNG frames under {Path(args.png_seq)}")
    if not args.no_h264:
        save_mp4(u8, out / "video.mp4", fps=args.fps)
        print(f"wrote {out / 'video.mp4'}  ({len(u8)} frames, {u8.shape[1]}x{u8.shape[2]})")
    else:
        print("skipped silent video.mp4 (--no-h264)")

    # a few contact-sheet thumbnails for a quick eyeball
    try:
        from PIL import Image
        idxs = sorted({0, actual_t // 2, actual_t - 1})
        for i in idxs:
            Image.fromarray(u8[i]).save(out / f"frame_{i:04d}.png")
        print("wrote thumbnails:", [f"frame_{i:04d}.png" for i in idxs])
    except Exception as e:
        print(f"thumbnail skip: {e}")

    # ---- audio ----
    if args.skip_audio:
        print("skipped audio VAE (--skip-audio)")
        return 0

    print("loading audio VAE (fp32)...")
    avae = build_audio_vae(args.device)
    az = audio_z.to(next(avae.parameters()).device)
    stereo = decode_audio(avae, az).cpu()  # [1, 2, L]
    wav = stereo[0].numpy()  # [2, L] in [-1, 1]
    sr = 32000
    print(f"decoded audio {tuple(stereo.shape)} @ {sr} Hz ({wav.shape[-1] / sr:.3f}s)")

    rms_l = np.sqrt((wav[0] ** 2).mean())
    rms_r = np.sqrt((wav[1] ** 2).mean())
    peak = np.abs(wav).max()
    print(f"audio: finite={np.isfinite(wav).all()} rms L={rms_l:.5f} R={rms_r:.5f} peak={peak:.4f}")

    from scipy.io import wavfile
    # scipy expects (samples, channels); wav is (channels, samples)
    wav_t = np.clip(wav.T * 32767, -32768, 32767).astype(np.int16)
    wav_path = out / "audio.wav"
    video_path = out / "video.mp4"
    wavfile.write(str(wav_path), sr, wav_t)
    print(f"wrote {wav_path}")

    # Normal deliverable: one playable file with picture + sound.
    if not args.no_mux and not args.no_h264:
        mux_path = out / "output.mp4"
        try:
            mux_av(video_path, wav_path, mux_path)
            print(f"wrote {mux_path}  (muxed video+audio — open this)")
        except Exception as e:
            print(f"FAIL: mux to {mux_path} failed: {e}", file=sys.stderr)
            print(
                "sidecars video.mp4 + audio.wav are still present; "
                "install/fix ffmpeg or imageio-ffmpeg and re-run, or mux manually.",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
