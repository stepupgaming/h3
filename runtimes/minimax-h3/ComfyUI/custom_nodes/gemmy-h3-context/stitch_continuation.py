"""SatoDive MiniMax-H3 Native Guide stitch (MIT, nkxx188 / SatoDive).

Ported from SatoDive/Minimax-H3-Latent-Continuation `stitch_continuation.py`.
The Easy VIDEO/context helpers are replaced with tensor + MP4 I/O so this
lives in gemmy-h3-context without pinning the Easy director.

Native Guide re-renders the first ``context_frames`` of a continuation as the
source tail. Stitching cuts the source at that re-render and appends the
continuation whole, with colour match on the overlap and source audio through
the overlap.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

FPS = 24.0


def _audio_sample_rate(audio: Mapping) -> int:
    return int(audio.get("sample_rate") or audio.get("samplerate") or audio.get("sampler_rate") or 32000)


def _normalize_video_frames(frames: torch.Tensor) -> torch.Tensor:
    """Normalize RGB frames to CPU float RGB in [0, 1]. Shape [T,H,W,C]."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("Video must provide frames with shape [frames, height, width, channels]")
    if frames.shape[0] < 1 or frames.shape[1] < 1 or frames.shape[2] < 1 or frames.shape[-1] < 3:
        raise ValueError("Video contains no usable RGB frames")
    frames = frames[..., :3].detach().to(device="cpu", dtype=torch.float32).contiguous()
    try:
        if float(frames.max()) > 1.5:
            frames = frames / 255.0
    except Exception:
        pass
    return frames.clamp(0.0, 1.0)


def _resize_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if int(frames.shape[1]) == height and int(frames.shape[2]) == width:
        return frames
    x = frames.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(
        x, size=(int(height), int(width)), mode="bilinear", align_corners=False
    )
    return x.permute(0, 2, 3, 1).contiguous()


def _match_features(frames: torch.Tensor, size: int = 48) -> torch.Tensor:
    """Small grayscale signature per frame, for fast and robust comparison.

    Downsampling suppresses the codec noise and fine re-render differences that
    would otherwise dominate a raw pixel distance, while keeping composition and
    subject position -- which is what alignment actually depends on.
    """
    value = frames.permute(0, 3, 1, 2).float()
    if value.shape[1] >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=value.device).view(1, 3, 1, 1)
        value = (value[:, :3] * weights).sum(dim=1, keepdim=True)
    else:
        value = value[:, :1]
    value = torch.nn.functional.interpolate(
        value, size=(size, size), mode="area"
    ).flatten(1)
    value = value - value.mean(dim=1, keepdim=True)
    norm = value.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return value / norm


def find_cut_frame(
    source: torch.Tensor,
    continuation: torch.Tensor,
    search_frames: int,
    match_window: int,
) -> tuple[int, float]:
    """Frame in ``source`` where ``continuation`` begins.

    Returns ``(cut_frame, similarity)``. ``cut_frame`` is the first source frame
    to discard; keeping ``source[:cut_frame]`` and appending the continuation
    whole gives a continuous join. Similarity is cosine, 1.0 being identical.
    """
    total = int(source.shape[0])
    window = max(1, min(int(match_window), int(continuation.shape[0]), total))
    latest = total - window
    if latest < 0:
        return total, 0.0
    earliest = max(0, latest - max(0, int(search_frames)))

    target = _match_features(continuation[:window])
    best_frame, best_score = latest, float("-inf")
    for start in range(earliest, latest + 1):
        candidate = _match_features(source[start:start + window])
        score = float((candidate * target).sum(dim=1).mean())
        if score > best_score:
            best_score, best_frame = score, start
    return best_frame, best_score


def _trim_audio(audio: Any, keep_frames: int, fps: float) -> Any:
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        return None
    rate = int(_audio_sample_rate(audio))
    keep = max(0, int(round(float(keep_frames) / max(1e-6, float(fps)) * rate)))
    keep = min(keep, int(waveform.shape[-1]))
    return {"waveform": waveform[..., :keep].contiguous(), "sample_rate": rate}


def _audio_tail_from(audio: Any, skip_frames: int, fps: float) -> Any:
    """Drop the first ``skip_frames`` worth of audio, keeping the rest."""
    if not isinstance(audio, Mapping) or not isinstance(audio.get("waveform"), torch.Tensor):
        return None
    waveform = audio["waveform"]
    if waveform.ndim != 3:
        return None
    rate = int(_audio_sample_rate(audio))
    skip = max(0, int(round(float(skip_frames) / max(1e-6, float(fps)) * rate)))
    skip = min(skip, int(waveform.shape[-1]))
    return {"waveform": waveform[..., skip:].contiguous(), "sample_rate": rate}


def _concat_audio(first: Any, second: Any) -> Any:
    tracks = [a for a in (first, second)
              if isinstance(a, Mapping) and isinstance(a.get("waveform"), torch.Tensor)]
    if not tracks:
        return None
    if len(tracks) == 1:
        return tracks[0]
    rate = int(_audio_sample_rate(tracks[0]))
    parts = []
    for track in tracks:
        wave = track["waveform"]
        if int(_audio_sample_rate(track)) != rate:
            wave = torch.nn.functional.interpolate(
                wave, scale_factor=rate / float(_audio_sample_rate(track)),
                mode="linear", align_corners=False,
            )
        parts.append(wave)
    channels = max(int(p.shape[1]) for p in parts)
    parts = [p.repeat(1, channels // int(p.shape[1]), 1) if int(p.shape[1]) < channels else p
             for p in parts]
    return {"waveform": torch.cat(parts, dim=-1).contiguous(), "sample_rate": rate}


def _crossfade_join(first: Any, second: Any, millis: float = 12.0) -> Any:
    """Concatenate two tracks with a short fade at the seam, preserving length."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return _concat_audio(first, second)
    a, b = first.get("waveform"), second.get("waveform")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return _concat_audio(first, second)
    rate = int(_audio_sample_rate(first))
    n = int(round(max(0.0, float(millis)) / 1000.0 * rate))
    n = min(n, int(a.shape[-1]), int(b.shape[-1]))
    if n < 8:
        return _concat_audio(first, second)
    channels = max(int(a.shape[1]), int(b.shape[1]))
    if int(a.shape[1]) < channels:
        a = a.repeat(1, channels // int(a.shape[1]), 1)
    if int(b.shape[1]) < channels:
        b = b.repeat(1, channels // int(b.shape[1]), 1)
    a, b = a.clone(), b.clone()
    ramp = torch.linspace(0.0, 1.0, n, dtype=a.dtype, device=a.device)
    a[..., -n:] = a[..., -n:] * torch.cos(ramp * 1.5707963)
    b[..., :n] = b[..., :n] * torch.sin(ramp * 1.5707963)
    return {"waveform": torch.cat([a, b], dim=-1).contiguous(), "sample_rate": rate}


def stitch_tensors(
    source: torch.Tensor,
    follow: torch.Tensor,
    *,
    source_audio: Any = None,
    follow_audio: Any = None,
    fps: float = FPS,
    alignment: str = "auto",
    overlap_frames: int = 39,
    search_frames: int = 30,
    match_window: int = 8,
    color_match: str = "mean + contrast",
    window_audio: str = "from source",
) -> dict[str, Any]:
    """SatoDive stitch on RGB tensors. Returns frames, audio, fps, cut_frame, report."""
    source = _normalize_video_frames(source)
    follow = _normalize_video_frames(follow)
    fps = float(fps or FPS)

    if int(source.shape[0]) == 0 or int(follow.shape[0]) == 0:
        raise ValueError("Stitch Continuation needs two non-empty videos")

    resized = False
    if follow.shape[1:3] != source.shape[1:3]:
        follow = _resize_frames(follow, int(source.shape[1]), int(source.shape[2]))
        resized = True

    total = int(source.shape[0])
    if str(alignment) == "fixed overlap":
        cut = max(0, total - int(overlap_frames))
        score = float("nan")
    else:
        nominal = max(0, total - int(overlap_frames))
        span = max(0, int(search_frames))
        window = max(1, min(int(match_window), int(follow.shape[0])))
        upper = min(total - window, nominal + span)
        lower = max(0, nominal - span)
        if upper < lower:
            cut, score = max(0, min(nominal, total)), float("nan")
        else:
            cut, score = find_cut_frame(
                source[:upper + window], follow, upper - lower, window
            )

    cut = max(0, min(int(cut), total))

    matched = "off"
    if str(color_match) != "off" and cut < total:
        window = min(int(match_window), total - cut, int(follow.shape[0]))
        if window >= 1:
            ref = source[cut:cut + window]
            cur = follow[:window]
            ref_mean = ref.mean(dim=(0, 1, 2), keepdim=True)
            cur_mean = cur.mean(dim=(0, 1, 2), keepdim=True)
            if str(color_match) == "mean + contrast":
                ref_std = ref.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
                cur_std = cur.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-5)
                scale = (ref_std / cur_std).clamp(0.5, 2.0)
            else:
                scale = torch.ones_like(ref_mean)
            follow = ((follow - cur_mean) * scale + ref_mean).clamp(0.0, 1.0)
            matched = f"{color_match} (gain {float(scale.mean()):.3f})"

    joined = torch.cat([source[:cut], follow], dim=0).contiguous()

    overlap_used = min(int(total - cut), int(follow.shape[0]))
    audio_source = "continuation"
    if str(window_audio) == "from source" and overlap_used > 0 and source_audio is not None:
        head = _trim_audio(source_audio, cut + overlap_used, fps)
        tail = _audio_tail_from(follow_audio, overlap_used, fps)
        audio = _crossfade_join(head, tail) if tail is not None else head
        audio_source = f"source through the {overlap_used}-frame overlap, then continuation"
    else:
        audio = _concat_audio(_trim_audio(source_audio, cut, fps), follow_audio)

    dropped_all = cut == 0 and total > 0
    report = (
        f"source {total}f + continuation {int(follow.shape[0])}f -> {int(joined.shape[0])}f\n"
        f"cut source at frame {cut} (dropped {total - cut} overlapping frames)\n"
        f"overlap_frames: {int(overlap_frames)}\n"
        + f"colour match: {matched}\n"
        + f"overlap audio: {audio_source}\n"
        + f"alignment: {alignment}"
        + ("" if score != score else f", match {score:.4f}")
        + (f"\ncontinuation resized to {int(source.shape[2])}x{int(source.shape[1])}"
           if resized else "")
        + ("\nWARNING: the entire source video was dropped. overlap_frames "
           f"({int(overlap_frames)}) is >= the source length ({total}); lower it "
           "or supply a longer source." if dropped_all else "")
    )
    return {
        "frames": joined,
        "audio": audio,
        "fps": fps,
        "cut_frame": int(cut),
        "report": report,
        "match": None if score != score else float(score),
    }


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found") from exc


def load_mp4(path: Path) -> tuple[torch.Tensor, dict | None, float]:
    import imageio.v2 as imageio

    path = Path(path)
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or FPS)
        frames = []
        for fr in reader:
            arr = np.asarray(fr)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            frames.append(arr)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{path}: no frames")
    video = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32) / 255.0)
    audio = _load_audio_pcm(path)
    return video, audio, fps


def _load_audio_pcm(path: Path) -> dict | None:
    cmd = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-vn", "-ac", "2", "-ar", "32000", "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size < 2:
        return None
    if data.size % 2:
        data = data[: data.size // 2 * 2]
    wave = torch.from_numpy(np.ascontiguousarray(data.reshape(-1, 2).T)).unsqueeze(0)
    return {"waveform": wave, "sample_rate": 32000}


def write_mp4(frames: torch.Tensor, audio: Any, fps: float, output: Path) -> None:
    import imageio.v2 as imageio
    import wave

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = _normalize_video_frames(frames)
    uint8 = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    tmp_dir = Path(tempfile.mkdtemp(prefix="h3_stitch_"))
    try:
        silent = tmp_dir / "v.mp4"
        imageio.mimwrite(str(silent), uint8, fps=float(fps or FPS), codec="libx264",
                         quality=8, macro_block_size=2)
        if not (isinstance(audio, Mapping) and isinstance(audio.get("waveform"), torch.Tensor)):
            shutil.copy2(silent, output)
            return
        waveform = audio["waveform"].detach().cpu()
        if waveform.ndim != 3:
            shutil.copy2(silent, output)
            return
        rate = int(_audio_sample_rate(audio))
        pcm = waveform.squeeze(0)
        if pcm.shape[0] == 1:
            pcm = pcm.repeat(2, 1)
        interleaved = pcm[:2].transpose(0, 1).contiguous().numpy()
        interleaved = np.clip(interleaved, -1.0, 1.0)
        raw = (interleaved * 32767.0).astype(np.int16).tobytes()
        wav_path = tmp_dir / "a.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(raw)
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent), "-i", str(wav_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(output),
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def stitch_mp4s(
    source: Path,
    continuation: Path,
    output: Path,
    *,
    overlap_frames: int = 39,
    alignment: str = "auto",
    search_frames: int = 30,
    match_window: int = 8,
    color_match: str = "mean + contrast",
    window_audio: str = "from source",
) -> dict[str, Any]:
    src_f, src_a, src_fps = load_mp4(Path(source))
    fol_f, fol_a, _fol_fps = load_mp4(Path(continuation))
    result = stitch_tensors(
        src_f, fol_f,
        source_audio=src_a,
        follow_audio=fol_a,
        fps=src_fps,
        alignment=alignment,
        overlap_frames=overlap_frames,
        search_frames=search_frames,
        match_window=match_window,
        color_match=color_match,
        window_audio=window_audio,
    )
    write_mp4(result["frames"], result["audio"], result["fps"], Path(output))
    result["output"] = str(Path(output))
    print(f"[h3-stitch] {result['report']}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SatoDive Native Guide stitch")
    ap.add_argument("--source", required=True)
    ap.add_argument("--continuation", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overlap-frames", type=int, default=39)
    ap.add_argument("--alignment", default="auto", choices=["auto", "fixed overlap"])
    args = ap.parse_args(argv)
    stitch_mp4s(
        Path(args.source), Path(args.continuation), Path(args.output),
        overlap_frames=args.overlap_frames, alignment=args.alignment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
