"""Gemmy H3 mask-edit Comfy nodes (encode crop + NestedTensor noise_mask).

Crop/uncrop stay in runtimes/minimax-h3/h3_mask_edit.py so the 16 GB worker
can unload SAM3 before Eros loads. These nodes only run inside the H3 sample
graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .nodes import _pack_av, _unpack_av

_H3_ROOT = Path(__file__).resolve().parents[3]
if str(_H3_ROOT) not in sys.path:
    sys.path.insert(0, str(_H3_ROOT))

from h3_mask_edit import pixel_mask_to_h3_latent_mask  # noqa: E402


class GemmyH3EncodeVideoFrames:
    """Encode an IMAGE batch as H3 video latent [B,24,T,H/16,W/16]."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video",)
    FUNCTION = "encode"
    CATEGORY = "gemmy/h3"

    def encode(self, images, vae):
        if images.ndim != 4:
            raise ValueError(f"expected IMAGE [N,H,W,C], got {tuple(images.shape)}")
        # Comfy IMAGE is 0..1 HWC. H3 VAE encode wants [B,3,T,H,W] in [-1,1]
        # through the wrapper; vae.encode(IMAGE) already handles the wrapper.
        z = vae.encode(images)
        if getattr(z, "is_nested", False):
            z = z.unbind()[0]
        return ({"samples": z},)


class GemmyH3SetAVNoiseMask:
    """Attach a pixel-space video mask as NestedTensor noise_mask.

    White = denoise (edit). Audio stays 0 so the unused audio half is not
    the product soundtrack — the worker muxes the source wav after uncrop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "apply"
    CATEGORY = "gemmy/h3"

    def apply(self, samples, mask):
        video, audio = _unpack_av(samples)
        m = mask
        if m.ndim == 2:
            m = m.unsqueeze(0)
        arr = m.detach().float().cpu().numpy()
        latent_m = pixel_mask_to_h3_latent_mask(arr)
        video_mask = torch.from_numpy(latent_m).to(device=video.device, dtype=video.dtype)
        # Broadcast to video channels if prepare_mask does not: [1,1,T,H,W]
        if video_mask.shape[2] != video.shape[2]:
            # Snap by cropping/padding T
            t = int(video.shape[2])
            if video_mask.shape[2] > t:
                video_mask = video_mask[:, :, :t]
            else:
                video_mask = torch.nn.functional.pad(
                    video_mask, (0, 0, 0, 0, 0, t - video_mask.shape[2])
                )
        if video_mask.shape[-2:] != video.shape[-2:]:
            video_mask = torch.nn.functional.interpolate(
                video_mask[0], size=video.shape[-2:], mode="nearest"
            ).unsqueeze(0)
        audio_mask = torch.zeros(
            audio.shape[0], 1, audio.shape[2], audio.shape[3],
            device=audio.device, dtype=audio.dtype,
        )
        extra = {k: v for k, v in samples.items() if k not in ("samples", "noise_mask")}
        packed = _pack_av(video, audio, extra=extra or None)
        packed["noise_mask"] = __import__("comfy.nested_tensor", fromlist=["NestedTensor"]).NestedTensor(
            (video_mask, audio_mask)
        )
        return (packed,)
