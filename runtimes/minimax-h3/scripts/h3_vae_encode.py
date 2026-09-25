"""Encode a still image into a DiT-normalized H3 FL2VA keyframe latent.

Mirrors WanGP / official condition encode conventions:

  pixels (uint8 or [-1,1])
    → ImageNet normalize
    → VAE image path (process_image=True / adaptive encode)
    → DiagonalGaussian **sample** (not mode), optional fixed RNG
    → z_dit = (z_raw - latents_mean) / latents_std

Output shape: ``[1, 24, 1, H/16, W/16]`` float32 safetensors
(``video_latent`` or ``first_frame_latent`` key).

Canvas must match the sample path (``adapt_canvas`` / ``--width --height``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from h3_vae_decode import IMAGENET_MEAN, IMAGENET_STD, ROOT as DECODE_ROOT, build_video_vae  # noqa: E402

assert DECODE_ROOT == ROOT

FL2VA_CFG = ROOT / "MiniMax-H3-Official" / "FL2VA" / "video_vae" / "config.json"


def video_latent_stats(device=None, dtype=torch.float32):
    cfg = json.loads(FL2VA_CFG.read_text())
    mean = torch.tensor(cfg["latents_mean"], dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(cfg["latents_std"], dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def fit_image_rgb(
    rgb: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Cover-scale + center-crop to exact ``width x height`` (uint8 RGB)."""
    from PIL import Image

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    h0, w0 = rgb.shape[:2]
    if w0 <= 0 or h0 <= 0:
        raise ValueError("empty image")
    scale = max(width / w0, height / h0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    im = Image.fromarray(rgb, mode="RGB").resize((nw, nh), Image.Resampling.BICUBIC)
    arr = np.asarray(im)
    top = max(0, (nh - height) // 2)
    left = max(0, (nw - width) // 2)
    out = arr[top : top + height, left : left + width]
    if out.shape[0] != height or out.shape[1] != width:
        # edge case: rounding shortfall — pad
        pad = np.zeros((height, width, 3), dtype=np.uint8)
        pad[: out.shape[0], : out.shape[1]] = out
        out = pad
    return out


def load_image_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def pixels_minus1_1_to_nchw(rgb_m11: np.ndarray) -> torch.Tensor:
    """[H,W,3] or [3,H,W] in [-1,1] → float32 NCHW [1,3,H,W] ImageNet-normalized."""
    if rgb_m11.ndim == 3 and rgb_m11.shape[0] == 3:
        x = torch.from_numpy(np.asarray(rgb_m11, dtype=np.float32)).unsqueeze(0)
    elif rgb_m11.ndim == 3 and rgb_m11.shape[2] == 3:
        x = torch.from_numpy(np.asarray(rgb_m11, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"bad pixel shape {getattr(rgb_m11, 'shape', None)}")
    # [-1,1] → [0,1] → ImageNet
    x = (x + 1.0) * 0.5
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def uint8_rgb_to_nchw_imagenet(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 → ImageNet-normalized NCHW float32."""
    x = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def encode_image_condition(
    vae,
    nchw_imagenet: torch.Tensor,
    *,
    encode_seed: int = 42,
) -> torch.Tensor:
    """ImageNet-normalized NCHW → DiT-normalized keyframe ``[1,24,1,lh,lw]`` f32.

    Uses the official image encode path (adaptive + trim T=1) and a **seeded**
    posterior sample so condition latents are reproducible (WanGP uses seed 42).
    """
    from video_vae.vae_module import DiagonalGaussianDistribution

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    x = nchw_imagenet.to(device=device, dtype=dtype)
    if x.ndim != 4:
        raise ValueError(f"expected NCHW, got {tuple(x.shape)}")

    # Match encode_base(..., process_image=True): add temporal dim for 3d conv.
    if getattr(vae, "use_3d_conv", True) and x.ndim == 4:
        x5 = x.unsqueeze(2)  # [B,3,1,H,W]
    else:
        x5 = x

    moments = vae._adaptive_encode(x5)
    dist = DiagonalGaussianDistribution(moments)
    # Sample noise on CPU with fixed seed (WanGP encode_condition), then apply.
    g = torch.Generator(device="cpu").manual_seed(int(encode_seed))
    noise = torch.randn(dist.mean.shape, generator=g, dtype=torch.float32)
    z = dist.mean + dist.std * noise.to(device=dist.mean.device, dtype=dist.mean.dtype)

    if getattr(vae, "use_3d_conv", True):
        z = vae.trim_code(z, 1)

    # Official image latents: [B,D,1,H',W'] or squeezed; force 5D batch.
    if z.ndim == 4:
        z = z.unsqueeze(2)
    if z.ndim != 5:
        raise RuntimeError(f"unexpected encode rank {z.ndim} shape {tuple(z.shape)}")

    mean, std = video_latent_stats(device=z.device, dtype=torch.float32)
    z = z.float()
    z_dit = (z - mean) / std
    return z_dit.detach().cpu().contiguous()


def encode_path(
    image: Path,
    *,
    width: int,
    height: int,
    device: str = "cuda",
    encode_seed: int = 42,
    vae=None,
) -> torch.Tensor:
    """Load image file, fit canvas, encode → DiT latent on CPU."""
    own_vae = vae is None
    if own_vae:
        vae = build_video_vae(device)
    try:
        rgb = fit_image_rgb(load_image_rgb(image), width, height)
        nchw = uint8_rgb_to_nchw_imagenet(rgb)
        return encode_image_condition(vae, nchw, encode_seed=encode_seed)
    finally:
        if own_vae:
            del vae
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def encode_pixels_m11(
    pixels_m11: Union[np.ndarray, torch.Tensor],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    device: str = "cuda",
    encode_seed: int = 42,
    vae=None,
) -> torch.Tensor:
    """Encode a [-1,1] frame (HWC or CHW or 1CHW) to DiT latent.

    If width/height given, cover-crop uint8 path after converting to u8.
    Otherwise encode at native resolution (must already match sample canvas).
    """
    if isinstance(pixels_m11, torch.Tensor):
        t = pixels_m11.detach().float().cpu()
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 3:
            arr = t.permute(1, 2, 0).numpy()
        elif t.ndim == 3 and t.shape[2] == 3:
            arr = t.numpy()
        else:
            raise ValueError(f"bad tensor shape {tuple(pixels_m11.shape)}")
    else:
        arr = np.asarray(pixels_m11, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))

    own_vae = vae is None
    if own_vae:
        vae = build_video_vae(device)
    try:
        if width is not None and height is not None:
            u8 = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            u8 = fit_image_rgb(u8, width, height)
            nchw = uint8_rgb_to_nchw_imagenet(u8)
        else:
            nchw = pixels_minus1_1_to_nchw(arr)
        return encode_image_condition(vae, nchw, encode_seed=encode_seed)
    finally:
        if own_vae:
            del vae
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def save_keyframe_latent(z: torch.Tensor, path: Path, key: str = "video_latent") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if z.ndim == 4:
        z = z.unsqueeze(0)
    if z.shape[0] != 1 or z.shape[1] != 24 or z.shape[2] != 1:
        raise ValueError(f"expected [1,24,1,H,W], got {tuple(z.shape)}")
    safetensors.torch.save_file({key: z.contiguous().float().cpu()}, str(path))
    return path


def save_video_latent(z: torch.Tensor, path: Path, key: str = "video_latent") -> Path:
    """Save DiT-normalized video/image latent ``[1,24,T,H,W]`` (T≥1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if z.ndim == 4:
        z = z.unsqueeze(0)
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 24:
        raise ValueError(f"expected [1,24,T,H,W], got {tuple(z.shape)}")
    safetensors.torch.save_file({key: z.contiguous().float().cpu()}, str(path))
    return path


def save_audio_latent(z: torch.Tensor, path: Path, key: str = "audio_latent") -> Path:
    """Save DiT-normalized stereo audio latent ``[1,32,2,T]``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if z.ndim == 3:
        # [32,2,T] → batch
        z = z.unsqueeze(0)
    if z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != 32 or z.shape[2] != 2:
        raise ValueError(f"expected [1,32,2,T], got {tuple(z.shape)}")
    safetensors.torch.save_file({key: z.contiguous().float().cpu()}, str(path))
    return path


def snap_frame_count(n: int) -> int:
    """Snap pixel frame count up to the 17k+5 grid (same as sample CLI)."""
    fc = max(int(n), 5)
    while fc % 17 != 5:
        fc += 1
    return fc


def latent_t_from_frames(frame_count: int) -> int:
    fc = snap_frame_count(frame_count)
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def fit_video_rgb(
    frames: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Cover-scale + center-crop each frame. ``frames`` [T,H,W,3] uint8 → same."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {frames.shape}")
    out = np.empty((frames.shape[0], height, width, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = fit_image_rgb(frames[i], width, height)
    return out


def load_video_rgb(path: Path, *, max_frames: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Load video → (uint8 [T,H,W,3], fps). Uses imageio."""
    import imageio.v2 as imageio

    path = Path(path)
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 24.0)
        frames = []
        for i, fr in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            arr = np.asarray(fr)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            frames.append(arr.astype(np.uint8))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path}: no frames")
    return np.stack(frames, axis=0), fps


@torch.no_grad()
def encode_video_condition(
    vae,
    ncthw_imagenet: torch.Tensor,
    *,
    encode_seed: int = 42,
) -> torch.Tensor:
    """ImageNet-normalized ``[1,3,T,H,W]`` → DiT-normalized ``[1,24,T',H',W']`` f32.

    Uses the official multi-frame encode path (``encode_base`` / ``process_image=False``)
    with a seeded posterior sample, then ``(z-mean)/std``. Caller should already
    snap T via ``get_suitable_video_length`` / 17k+5 grid.
    """
    from video_vae.vae_module import DiagonalGaussianDistribution

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    x = ncthw_imagenet.to(device=device, dtype=dtype)
    if x.ndim != 5:
        raise ValueError(f"expected NCTHW, got {tuple(x.shape)}")

    # encode_base samples with its own RNG; we re-sample with a fixed seed for
    # reproducibility (same as image path).
    if hasattr(vae, "encode_temporal"):
        moments = vae.encode_temporal(x)
    else:
        moments = vae._adaptive_encode(x)
    dist = DiagonalGaussianDistribution(moments)
    g = torch.Generator(device="cpu").manual_seed(int(encode_seed))
    noise = torch.randn(dist.mean.shape, generator=g, dtype=torch.float32)
    z = dist.mean + dist.std * noise.to(device=dist.mean.device, dtype=dist.mean.dtype)

    if z.ndim == 4:
        z = z.unsqueeze(2)
    mean, std = video_latent_stats(device=z.device, dtype=torch.float32)
    z_dit = (z.float() - mean) / std
    return z_dit.detach().cpu().contiguous()


def encode_video_path(
    video: Path,
    *,
    width: int,
    height: int,
    device: str = "cuda",
    encode_seed: int = 42,
    max_seconds: float = 15.0,
    vae=None,
) -> torch.Tensor:
    """Load mp4/webm, fit canvas, snap length, encode → DiT video latent on CPU."""
    own_vae = vae is None
    if own_vae:
        vae = build_video_vae(device)
    try:
        max_frames = int(max_seconds * 24) + 8
        frames, fps = load_video_rgb(video, max_frames=max_frames)
        # Resample-ish: if fps far from 24, stride/repeat roughly to 24fps budget
        if abs(fps - 24.0) > 0.5 and fps > 1e-3:
            idx = np.linspace(0, len(frames) - 1, num=max(1, int(round(len(frames) * 24.0 / fps))))
            idx = np.clip(np.round(idx).astype(int), 0, len(frames) - 1)
            frames = frames[idx]
        # Cap duration at max_seconds @ 24fps then snap to VAE-suitable length
        max_fc = int(max_seconds * 24)
        if frames.shape[0] > max_fc:
            frames = frames[:max_fc]
        # Prefer official processor grid when VAE is loaded; else 17k+5 snap.
        try:
            proc = getattr(vae, "processor", None)
            if proc is not None and hasattr(proc, "get_suitable_video_length"):
                fc = int(proc.get_suitable_video_length(frames.shape[0], verbose=False))
            else:
                fc = snap_frame_count(frames.shape[0])
        except Exception:
            fc = snap_frame_count(frames.shape[0])
        if frames.shape[0] < fc:
            pad = np.repeat(frames[-1:], fc - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        elif frames.shape[0] > fc:
            frames = frames[:fc]
        frames = fit_video_rgb(frames, width, height)
        # [T,H,W,3] uint8 → [1,3,T,H,W] ImageNet
        x = torch.from_numpy(frames.astype(np.float32) / 255.0)  # T,H,W,3
        x = x.permute(3, 0, 1, 2).unsqueeze(0)  # 1,3,T,H,W
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
        x = (x - mean) / std
        return encode_video_condition(vae, x, encode_seed=encode_seed)
    finally:
        if own_vae:
            del vae
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def audio_latent_stats(device=None, dtype=torch.float32):
    cfg = json.loads(
        (ROOT / "MiniMax-H3-Official" / "FL2VA" / "audio_vae" / "config.json").read_text()
    )
    mean = torch.tensor(cfg["latents_mean"], dtype=dtype, device=device).view(1, -1, 1)
    std = torch.tensor(cfg["latents_std"], dtype=dtype, device=device).view(1, -1, 1)
    return mean, std


@torch.no_grad()
def encode_audio_mono(vae, mono: torch.Tensor, *, encode_seed: int = 42) -> torch.Tensor:
    """Encode one channel ``[1, L]`` or ``[1,1,L]`` → DiT-normalized ``[1,32,T]``."""
    if mono.ndim == 2:
        mono = mono.unsqueeze(1)  # [B,1,L]
    if mono.ndim != 3 or mono.shape[1] != 1:
        raise ValueError(f"expected [B,1,L], got {tuple(mono.shape)}")
    device = next(vae.parameters()).device
    x = mono.to(device=device, dtype=torch.float32)
    x = vae.preprocess(x, getattr(vae, "sample_rate", 32000))
    h = vae.encoder(x)  # [B, latent_dim, T]
    if getattr(vae, "attn_proj", False) and hasattr(vae, "pre_block"):
        # AttnProjection: [B, T, C]
        h = vae.pre_block(h.transpose(1, 2)).transpose(1, 2)
    mean = vae.mean_proj(h)
    logs = vae.logs_proj(h)
    g = torch.Generator(device="cpu").manual_seed(int(encode_seed))
    eps = torch.randn(mean.shape, generator=g, dtype=torch.float32)
    z = mean + torch.exp(0.5 * logs) * eps.to(device=mean.device, dtype=mean.dtype)
    m, s = audio_latent_stats(device=z.device, dtype=torch.float32)
    z = (z.float() - m) / s
    return z.detach().cpu().contiguous()


@torch.no_grad()
def encode_audio_stereo(
    vae,
    stereo: torch.Tensor,
    *,
    encode_seed: int = 42,
) -> torch.Tensor:
    """Stereo ``[2, L]`` or ``[1,2,L]`` → DiT-normalized ``[1,32,2,T]``."""
    if stereo.ndim == 3 and stereo.shape[0] == 1:
        stereo = stereo[0]
    if stereo.ndim != 2 or stereo.shape[0] not in (1, 2):
        raise ValueError(f"expected [2,L] or [1,L], got {tuple(stereo.shape)}")
    if stereo.shape[0] == 1:
        stereo = stereo.repeat(2, 1)
    left = encode_audio_mono(vae, stereo[0:1], encode_seed=encode_seed)
    right = encode_audio_mono(vae, stereo[1:2], encode_seed=encode_seed + 1)
    # left/right: [1,32,T] → stack channels dim → [1,32,2,T]
    return torch.stack([left[0], right[0]], dim=1).unsqueeze(0).contiguous()


def _resample_linear(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resample for ``[C, L]`` float32 (no torchaudio required)."""
    if orig_sr == target_sr:
        return wav
    n_in = wav.shape[-1]
    n_out = max(1, int(round(n_in * float(target_sr) / float(orig_sr))))
    x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    out = np.empty((wav.shape[0], n_out), dtype=np.float32)
    for c in range(wav.shape[0]):
        out[c] = np.interp(x_new, x_old, wav[c].astype(np.float64)).astype(np.float32)
    return out


def load_audio_stereo(path: Path, *, sr: int = 32000, max_seconds: float = 15.0) -> torch.Tensor:
    """Load wav/flac/ogg/mp3 → float32 ``[2, L]`` at ``sr`` Hz, mono duplicated if needed.

    Prefer ``soundfile`` / ``scipy.io.wavfile`` (uv env); fall back to stdlib ``wave``
    for PCM wav. No torchaudio dependency.
    """
    path = Path(path)
    wav_np: Optional[np.ndarray] = None
    file_sr = 0

    # soundfile handles wav/flac/ogg; scipy wavfile for basic wav
    try:
        import soundfile as sf

        data, file_sr = sf.read(str(path), always_2d=True)  # [L, C]
        wav_np = np.asarray(data, dtype=np.float32).T  # [C, L]
    except Exception:
        wav_np = None

    if wav_np is None:
        try:
            from scipy.io import wavfile as scipy_wav

            file_sr, data = scipy_wav.read(str(path))
            data = np.asarray(data)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)
            if data.ndim == 1:
                data = data[:, None]
            wav_np = data.T  # [C, L]
        except Exception:
            wav_np = None

    if wav_np is None:
        import wave

        with wave.open(str(path), "rb") as wf:
            file_sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"{path}: unsupported sample width {sw}; install soundfile")
        if nch > 1:
            data = data.reshape(-1, nch).T
        else:
            data = data.reshape(1, -1)
        wav_np = data

    wav_np = _resample_linear(wav_np, int(file_sr), int(sr))
    if wav_np.shape[0] == 1:
        wav_np = np.repeat(wav_np, 2, axis=0)
    elif wav_np.shape[0] > 2:
        wav_np = wav_np[:2]
    max_len = int(max_seconds * sr)
    if wav_np.shape[-1] > max_len:
        wav_np = wav_np[:, :max_len]
    return torch.from_numpy(np.ascontiguousarray(wav_np)).float()


def encode_audio_path(
    audio: Path,
    *,
    device: str = "cuda",
    encode_seed: int = 42,
    max_seconds: float = 15.0,
    vae=None,
) -> torch.Tensor:
    from h3_vae_decode import build_audio_vae

    own_vae = vae is None
    if own_vae:
        vae = build_audio_vae(device)
    try:
        wav = load_audio_stereo(audio, max_seconds=max_seconds)
        return encode_audio_stereo(vae, wav, encode_seed=encode_seed)
    finally:
        if own_vae:
            del vae
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Encode still / video / audio into DiT-normalized H3 condition latents"
    )
    ap.add_argument(
        "input",
        type=Path,
        help="image (png/jpg), video (mp4/…), or audio (wav/mp3) depending on --mode",
    )
    ap.add_argument(
        "--mode",
        choices=["image", "video", "audio", "auto"],
        default="auto",
        help="auto detects from suffix (default auto)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .safetensors path (default under outputs/ from input stem)",
    )
    ap.add_argument("--width", type=int, default=None, help="canvas width (required for image/video)")
    ap.add_argument("--height", type=int, default=None, help="canvas height (required for image/video)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--video-vae-weights",
        type=Path,
        default=None,
        help="video VAE checkpoint override (official Comfy-Org fp16 or int8_convrot under vae/)",
    )
    ap.add_argument(
        "--encode-seed",
        type=int,
        default=42,
        help="RNG seed for VAE posterior sample (default 42, WanGP-compatible)",
    )
    ap.add_argument(
        "--key",
        default=None,
        help="safetensors key (default video_latent / audio_latent)",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="cap for video/audio ref encode (official ≤15s)",
    )
    args = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    path = Path(args.input)
    suffix = path.suffix.lower()
    mode = args.mode
    if mode == "auto":
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            mode = "image"
        elif suffix in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}:
            mode = "video"
        elif suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            mode = "audio"
        else:
            print(f"FAIL: cannot auto-detect mode for {suffix}; pass --mode", file=sys.stderr)
            return 1

    if mode in ("image", "video") and (args.width is None or args.height is None):
        print("FAIL: --width and --height required for image/video encode", file=sys.stderr)
        return 1

    if args.out is None:
        tag = {"image": "image_ref", "video": "video_ref", "audio": "audio_ref"}[mode]
        args.out = Path("outputs") / f"{path.stem}_{tag}.safetensors"

    vae = None
    if args.video_vae_weights is not None and mode in ("image", "video"):
        vae = build_video_vae(args.device, args.video_vae_weights)

    if mode == "image":
        print(f"encoding image {path} → canvas {args.width}x{args.height} on {args.device}")
        z = encode_path(
            path,
            width=args.width,
            height=args.height,
            device=args.device,
            encode_seed=args.encode_seed,
            vae=vae,
        )
        key = args.key or "video_latent"
        print(f"latent {tuple(z.shape)} finite={bool(torch.isfinite(z).all())} "
              f"mean={z.mean():.4f} std={z.std():.4f}")
        dest = save_keyframe_latent(z, args.out, key=key)
    elif mode == "video":
        print(f"encoding video {path} → canvas {args.width}x{args.height} on {args.device}")
        z = encode_video_path(
            path,
            width=args.width,
            height=args.height,
            device=args.device,
            encode_seed=args.encode_seed,
            max_seconds=args.max_seconds,
            vae=vae,
        )
        key = args.key or "video_latent"
        print(f"latent {tuple(z.shape)} finite={bool(torch.isfinite(z).all())} "
              f"mean={z.mean():.4f} std={z.std():.4f}")
        dest = save_video_latent(z, args.out, key=key)
    else:
        print(f"encoding audio {path} on {args.device}")
        z = encode_audio_path(
            path,
            device=args.device,
            encode_seed=args.encode_seed,
            max_seconds=args.max_seconds,
        )
        key = args.key or "audio_latent"
        print(f"latent {tuple(z.shape)} finite={bool(torch.isfinite(z).all())} "
              f"mean={z.mean():.4f} std={z.std():.4f}")
        dest = save_audio_latent(z, args.out, key=key)

    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
