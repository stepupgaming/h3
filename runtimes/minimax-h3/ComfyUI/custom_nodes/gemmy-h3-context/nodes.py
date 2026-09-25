"""Gemmy-owned MiniMax-H3 AV latent persist + continuation nodes.

Does not vendor Context Loop / MultiRef UI packs. Implements:
- persist/load of joint NestedTensor video+audio latents
- masked_av: copy tail into the target + 0/1 denoise masks
- native_guide: previous tail as MiniMaxH3AddGuide keyframes (no VAE, no mask)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy.nested_tensor
import folder_paths
import node_helpers
from comfy.utils import load_torch_file, save_torch_file

from .av_math import (
    LATENT_SCHEMA,
    MAX_FRACTIONAL_CHANGE,
    audio_fractional_change,
    audio_latent_t,
    audio_tail_slice,
    expected_audio_samples,
    fit_audio_time,
    frames_from_video_tokens,
    native_guide_tail_spec,
    snap_context_frames,
    video_latent_t,
)


def _unpack_av(latent: dict):
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError("Gemmy H3 context nodes expect a NestedTensor video+audio latent")
    video, audio = samples.tensors
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(f"expected video [B,24,T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(f"expected audio [B,32,2,T], got {tuple(audio.shape)}")
    return video, audio


def _pack_av(video: torch.Tensor, audio: torch.Tensor, extra: dict | None = None) -> dict:
    out = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if extra:
        out.update(extra)
    return out


def build_native_tail_keyframe(
    prev_video: torch.Tensor,
    prev_audio: torch.Tensor,
    target_video: torch.Tensor,
    context_frames: int = 39,
) -> tuple[dict, int]:
    """Slice the previous AV tail into a MiniMaxH3AddGuide keyframe dict.

    No VAE round trip. The sampler target stays empty; this only conditions.
    """
    if prev_video.ndim != 5 or prev_video.shape[1] != 24:
        raise ValueError(f"expected previous video [B,24,T,H,W], got {tuple(prev_video.shape)}")
    if target_video.ndim != 5 or target_video.shape[1] != 24:
        raise ValueError(f"expected target video [B,24,T,H,W], got {tuple(target_video.shape)}")
    if tuple(prev_video.shape[3:]) != tuple(target_video.shape[3:]):
        raise ValueError(
            "previous video spatial shape "
            f"{tuple(prev_video.shape)} does not match target {tuple(target_video.shape)}"
        )
    tgt_frames = frames_from_video_tokens(int(target_video.shape[2]))
    avail = frames_from_video_tokens(int(prev_video.shape[2]))
    ctx, vt, at = native_guide_tail_spec(context_frames, avail, tgt_frames)
    vt = min(vt, int(prev_video.shape[2]), int(target_video.shape[2]))
    at = min(at, int(prev_audio.shape[-1]) if prev_audio.ndim == 4 else 0)
    if vt < 1:
        raise ValueError("native guide needs at least one previous video token")
    video_tail = prev_video[:, :, -vt:].detach().contiguous()
    keyframe = {
        "resolved_frame_index": 0,
        "latent": video_tail,
    }
    if prev_audio.ndim == 4 and at > 0:
        keyframe["audio_latent"] = prev_audio[..., -at:].detach().contiguous()
    return keyframe, ctx


class GemmyH3SaveAVLatent:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "h3/gemmy_av"}),
                "fingerprint": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "gemmy/h3"
    OUTPUT_NODE = True

    def save(self, samples, filename_prefix="h3/gemmy_av", fingerprint="{}"):
        video, audio = _unpack_av(samples)
        folder, filename, counter, _sub, _prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        stem = f"{filename}_{counter:05}_h3av"
        st_path = Path(folder) / f"{stem}.safetensors"
        meta_path = Path(folder) / f"{stem}.json"
        try:
            fp = json.loads(fingerprint) if fingerprint.strip() else {}
        except json.JSONDecodeError:
            fp = {"raw": fingerprint}
        fp.setdefault("schema", LATENT_SCHEMA)
        fp["video_shape"] = list(video.shape)
        fp["audio_shape"] = list(audio.shape)
        fp["frames"] = frames_from_video_tokens(int(video.shape[2]))
        payload = {
            "video": video.detach().contiguous().cpu(),
            "audio": audio.detach().contiguous().cpu(),
        }
        save_torch_file(payload, str(st_path))
        meta_path.write_text(json.dumps(fp, indent=2), encoding="utf-8")
        return (str(st_path),)


class GemmyH3LoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "fingerprint")
    FUNCTION = "load"
    CATEGORY = "gemmy/h3"

    def load(self, path):
        st = Path(path)
        if st.suffix.lower() != ".safetensors":
            raise ValueError(f"H3 AV latent must be a .safetensors file: {st}")
        if not st.is_file():
            raise FileNotFoundError(f"H3 AV latent missing: {st}")
        blob = load_torch_file(str(st), safe_load=True)
        video = blob["video"]
        audio = blob["audio"]
        meta = st.with_suffix(".json")
        fp = "{}"
        if meta.is_file():
            fp = meta.read_text(encoding="utf-8")
        return (_pack_av(video, audio), fp)


class GemmyH3MaskedAVContext:
    """Copy an AV-safe tail into the target latent and protect it with 0/1 masks.

    This writes the predecessor into the *target* latent (masked_av). It is not
    COND-only last-frame / Ref2VA tail-ref.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("LATENT",),
                "context_frames": ("INT", {"default": 39, "min": 5, "max": 396}),
                "audio_mode": (["generated_audio", "source_track"],),
            },
            "optional": {
                "previous": ("LATENT",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "tail_image": ("IMAGE",),
                "tail_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "protected_frames")
    FUNCTION = "apply"
    CATEGORY = "gemmy/h3"

    def apply(
        self,
        target,
        context_frames=39,
        audio_mode="generated_audio",
        previous=None,
        vae=None,
        audio_vae=None,
        tail_image=None,
        tail_audio=None,
    ):
        tgt_v, tgt_a = _unpack_av(target)
        tgt_v = tgt_v.clone()
        tgt_a = tgt_a.clone()
        tgt_frames = frames_from_video_tokens(int(tgt_v.shape[2]))

        if previous is not None:
            prev_v, prev_a = _unpack_av(previous)
            if tuple(prev_v.shape[3:]) != tuple(tgt_v.shape[3:]):
                raise ValueError(
                    "predecessor video spatial shape "
                    f"{tuple(prev_v.shape)} does not match target {tuple(tgt_v.shape)}"
                )
            avail = frames_from_video_tokens(int(prev_v.shape[2]))
            ctx = snap_context_frames(context_frames, min(avail, tgt_frames))
            vt = video_latent_t(ctx)
            at = audio_latent_t(ctx)
            tgt_v[:, :, :vt] = prev_v[:, :, -vt:]
            tgt_a[..., :at] = prev_a[..., -at:]
        elif tail_image is not None:
            if vae is None:
                raise ValueError("VAE-tail import needs the video vae")
            frames = tail_image
            avail = int(frames.shape[0])
            ctx = snap_context_frames(context_frames, min(avail, tgt_frames))
            while frames.shape[0] > ctx:
                frames = frames[-ctx:]
            while frames.shape[0] % 17 != 5 and frames.shape[0] > 5:
                frames = frames[1:]
            encoded = vae.encode(frames)
            vt = int(encoded.shape[2])
            ctx = frames_from_video_tokens(vt)
            tgt_v[:, :, :vt] = encoded.to(device=tgt_v.device, dtype=tgt_v.dtype)
            at = audio_latent_t(ctx)
            if tail_audio is not None:
                if audio_vae is None:
                    raise ValueError("imported soundtrack needs the audio vae")
                waveform = tail_audio["waveform"]
                sr = int(tail_audio["sample_rate"])
                vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
                if sr != vae_sr:
                    import torchaudio

                    waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
                    sr = vae_sr
                expected = expected_audio_samples(ctx, sr)
                start, end = audio_tail_slice(int(waveform.shape[-1]), ctx, sr)
                waveform = waveform[..., start:end]
                actual = int(waveform.shape[-1])
                frac = audio_fractional_change(actual, expected)
                if frac > MAX_FRACTIONAL_CHANGE:
                    raise ValueError(
                        f"imported audio length mismatch: actual={actual} expected={expected} "
                        f"fractional_change={frac:.4f} (max {MAX_FRACTIONAL_CHANGE})"
                    )
                z = audio_vae.encode(waveform[:1].movedim(1, -1))
                at = min(at, int(z.shape[-1]), int(tgt_a.shape[-1]))
                tgt_a[..., :at] = z[..., :at].to(device=tgt_a.device, dtype=tgt_a.dtype)
        else:
            raise ValueError("GemmyH3MaskedAVContext needs previous latent or tail_image")

        vmask = torch.ones(
            (tgt_v.shape[0], 1, tgt_v.shape[2], tgt_v.shape[3], tgt_v.shape[4]),
            device=tgt_v.device,
            dtype=tgt_v.dtype,
        )
        amask = torch.ones(
            (tgt_a.shape[0], 1, 1, tgt_a.shape[-1]),
            device=tgt_a.device,
            dtype=tgt_a.dtype,
        )
        vt = video_latent_t(ctx)
        at = audio_latent_t(ctx)
        vmask[:, :, :vt] = 0
        if audio_mode == "source_track":
            amask.zero_()
        else:
            amask[..., :at] = 0

        out = _pack_av(
            tgt_v,
            tgt_a,
            extra={"noise_mask": comfy.nested_tensor.NestedTensor((vmask, amask))},
        )
        return (out, int(ctx))


class GemmyH3LatentTailGuide:
    """Previous AV tail as native H3 keyframes. No denoise mask, no VAE encode.

    Matches MiniMaxH3AddGuide's minimax_keyframes payload, using the saved
    latent tokens instead of re-encoding pixels. Target latent stays empty;
    every delivered frame is generated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "previous": ("LATENT",),
                "latent": ("LATENT",),
                "context_frames": ("INT", {"default": 39, "min": 5, "max": 396}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply"
    CATEGORY = "gemmy/h3"

    def apply(self, positive, previous, latent, context_frames=39):
        prev_v, prev_a = _unpack_av(previous)
        tgt_v, _tgt_a = _unpack_av(latent)
        keyframe, _ctx = build_native_tail_keyframe(
            prev_v, prev_a, tgt_v, context_frames
        )
        existing = []
        if positive and isinstance(positive[0], (list, tuple)) and len(positive[0]) > 1:
            existing = list(positive[0][1].get("minimax_keyframes") or [])
        keyframes = existing + [keyframe]
        out = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
        return (out,)


class GemmyH3TrimProtectedPrefix:
    """Drop the protected head after decode (Context Loop anchor_mode=head)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "protected_frames": ("INT", {"default": 39, "min": 0, "max": 396}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "gemmy/h3"

    def trim(self, images, protected_frames=39, audio=None):
        n = int(protected_frames)
        if n <= 0:
            return (images, audio)
        if images.shape[0] <= n:
            raise ValueError(f"cannot trim {n} protected frames from {images.shape[0]} decoded frames")
        out_img = images[n:]
        out_audio = audio
        if audio is not None:
            sr = int(audio["sample_rate"])
            drop = expected_audio_samples(n, sr)
            waveform = audio["waveform"]
            if waveform.shape[-1] <= drop:
                raise ValueError(
                    f"cannot trim {drop} audio samples from {waveform.shape[-1]}"
                )
            remain = waveform[..., drop:]
            expected = expected_audio_samples(int(out_img.shape[0]), sr)
            frac = audio_fractional_change(int(remain.shape[-1]), expected)
            if frac > MAX_FRACTIONAL_CHANGE:
                raise ValueError(
                    f"trimmed audio mismatch: actual={remain.shape[-1]} expected={expected} "
                    f"fractional_change={frac:.4f} (max {MAX_FRACTIONAL_CHANGE})"
                )
            if remain.shape[-1] != expected:
                # Bound is tight enough that a 1-sample pad/trim is allowed.
                if remain.shape[-1] > expected:
                    remain = remain[..., :expected]
                else:
                    remain = torch.nn.functional.pad(remain, (0, expected - remain.shape[-1]))
            out_audio = {"waveform": remain, "sample_rate": sr}
        return (out_img, out_audio)


class GemmyH3SplitAV:
    """Unpack NestedTensor video+audio so a 3D latent upscaler can see video only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "split"
    CATEGORY = "gemmy/h3"

    def split(self, samples):
        video, audio = _unpack_av(samples)
        return ({"samples": video}, {"samples": audio})


class GemmyH3InitFromAudio:
    """Put a wav into the *target* audio half of an H3 AV latent.

    Frozen ``ref_audio`` is context the sampler never rewrites. This node is
    the opposite: the performance becomes the starting latent, and KSampler
    ``denoise`` < 1 walks it (img2img). Do not set a 0-mask here — that is
    ``audio_mode=source_track`` lock, which keeps the source voice exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "apply"
    CATEGORY = "gemmy/h3"

    def apply(self, samples, audio, audio_vae):
        video, template = _unpack_av(samples)
        if audio is None:
            raise ValueError("GemmyH3InitFromAudio needs a connected AUDIO")
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sr != vae_sr:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        waveform = waveform[:1]
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2]
        z = audio_vae.encode(waveform.movedim(1, -1))
        keep, pad = fit_audio_time(int(z.shape[-1]), int(template.shape[-1]))
        z = z[..., :keep]
        if pad:
            z = torch.nn.functional.pad(z, (0, pad))
        z = z.to(device=template.device, dtype=template.dtype)
        extra = {k: v for k, v in samples.items() if k not in ("samples", "noise_mask")}
        return (_pack_av(video, z, extra=extra or None),)


class GemmyH3JoinAV:
    """Re-pack video+audio after a video-only spatial move."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("LATENT",),
                "audio": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "join"
    CATEGORY = "gemmy/h3"

    def join(self, video, audio):
        v = video["samples"]
        a = audio["samples"]
        if getattr(v, "is_nested", False):
            raise ValueError("GemmyH3JoinAV video input must be a plain [B,24,T,H,W] latent")
        if getattr(a, "is_nested", False):
            raise ValueError("GemmyH3JoinAV audio input must be a plain [B,32,2,T] latent")
        return (_pack_av(v, a),)


def _save_blob(prefix: str, output_dir: str, payload: dict, meta: dict, suffix: str) -> str:
    folder, filename, counter, _sub, _prefix = folder_paths.get_save_image_path(prefix, output_dir)
    stem = f"{filename}_{counter:05}_{suffix}"
    st_path = Path(folder) / f"{stem}.safetensors"
    meta_path = Path(folder) / f"{stem}.json"
    save_torch_file(payload, str(st_path))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return str(st_path)


def _wrap_nested(value):
    if isinstance(value, dict) and set(value) == {"$nested"}:
        parts = tuple(_wrap_nested(part) for part in value["$nested"])
        return comfy.nested_tensor.NestedTensor(parts)
    if isinstance(value, dict):
        return {k: _wrap_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_wrap_nested(v) for v in value]
    return value


class GemmyH3SaveConditioning:
    """Write conditioning, including nested minimax_refs latents, then let the process exit."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "h3/gemmy_cond"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "gemmy/h3"
    OUTPUT_NODE = True

    def save(self, conditioning, filename_prefix="h3/gemmy_cond"):
        from .conditioning_io import pack_tree

        blobs, tree = pack_tree(conditioning)
        path = _save_blob(
            filename_prefix,
            self.output_dir,
            blobs,
            {"schema": "gemmy-h3-conditioning-v1", "tree": tree},
            "h3cond",
        )
        return (path,)


class GemmyH3LoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "gemmy/h3"

    def load(self, path):
        from .conditioning_io import unpack_tree

        st = Path(path)
        if st.suffix.lower() != ".safetensors":
            raise ValueError(f"H3 conditioning must be a .safetensors file: {st}")
        if not st.is_file():
            raise FileNotFoundError(f"H3 conditioning missing: {st}")
        meta_path = st.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != "gemmy-h3-conditioning-v1":
            raise ValueError(f"unexpected conditioning schema in {meta_path}")
        blobs = load_torch_file(str(st), safe_load=True)
        return (_wrap_nested(unpack_tree(blobs, meta["tree"])),)


class GemmyH3SaveSigmas:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "filename_prefix": ("STRING", {"default": "h3/gemmy_sigmas"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "gemmy/h3"
    OUTPUT_NODE = True

    def save(self, sigmas, filename_prefix="h3/gemmy_sigmas"):
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"SIGMAS must be a tensor, got {type(sigmas).__name__}")
        path = _save_blob(
            filename_prefix,
            self.output_dir,
            {"sigmas": sigmas.detach().cpu().contiguous()},
            {"schema": "gemmy-h3-sigmas-v1"},
            "h3sig",
        )
        return (path,)


class GemmyH3LoadSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "load"
    CATEGORY = "gemmy/h3"

    def load(self, path):
        st = Path(path)
        if st.suffix.lower() != ".safetensors":
            raise ValueError(f"H3 sigmas must be a .safetensors file: {st}")
        if not st.is_file():
            raise FileNotFoundError(f"H3 sigmas missing: {st}")
        blob = load_torch_file(str(st), safe_load=True)
        if "sigmas" not in blob:
            raise KeyError(f"sigmas tensor missing in {st}")
        return (blob["sigmas"],)
