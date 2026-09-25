"""Gemmy-owned MiniMax-H3 AV context nodes (persist + masked/native continue)."""

from .mask_nodes import GemmyH3EncodeVideoFrames, GemmyH3SetAVNoiseMask
from .nodes import (
    GemmyH3InitFromAudio,
    GemmyH3JoinAV,
    GemmyH3LatentTailGuide,
    GemmyH3LoadAVLatent,
    GemmyH3LoadConditioning,
    GemmyH3LoadSigmas,
    GemmyH3MaskedAVContext,
    GemmyH3SaveAVLatent,
    GemmyH3SaveConditioning,
    GemmyH3SaveSigmas,
    GemmyH3SplitAV,
    GemmyH3TrimProtectedPrefix,
)

NODE_CLASS_MAPPINGS = {
    "GemmyH3SaveAVLatent": GemmyH3SaveAVLatent,
    "GemmyH3LoadAVLatent": GemmyH3LoadAVLatent,
    "GemmyH3MaskedAVContext": GemmyH3MaskedAVContext,
    "GemmyH3LatentTailGuide": GemmyH3LatentTailGuide,
    "GemmyH3TrimProtectedPrefix": GemmyH3TrimProtectedPrefix,
    "GemmyH3SplitAV": GemmyH3SplitAV,
    "GemmyH3JoinAV": GemmyH3JoinAV,
    "GemmyH3InitFromAudio": GemmyH3InitFromAudio,
    "GemmyH3EncodeVideoFrames": GemmyH3EncodeVideoFrames,
    "GemmyH3SetAVNoiseMask": GemmyH3SetAVNoiseMask,
    "GemmyH3SaveConditioning": GemmyH3SaveConditioning,
    "GemmyH3LoadConditioning": GemmyH3LoadConditioning,
    "GemmyH3SaveSigmas": GemmyH3SaveSigmas,
    "GemmyH3LoadSigmas": GemmyH3LoadSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmyH3SaveAVLatent": "Gemmy H3 Save AV Latent",
    "GemmyH3LoadAVLatent": "Gemmy H3 Load AV Latent",
    "GemmyH3MaskedAVContext": "Gemmy H3 Masked AV Context",
    "GemmyH3LatentTailGuide": "Gemmy H3 Latent Tail Guide",
    "GemmyH3TrimProtectedPrefix": "Gemmy H3 Trim Protected Prefix",
    "GemmyH3SplitAV": "Gemmy H3 Split AV",
    "GemmyH3JoinAV": "Gemmy H3 Join AV",
    "GemmyH3InitFromAudio": "Gemmy H3 Init From Audio",
    "GemmyH3EncodeVideoFrames": "Gemmy H3 Encode Video Frames",
    "GemmyH3SetAVNoiseMask": "Gemmy H3 Set AV Noise Mask",
    "GemmyH3SaveConditioning": "Gemmy H3 Save Conditioning",
    "GemmyH3LoadConditioning": "Gemmy H3 Load Conditioning",
    "GemmyH3SaveSigmas": "Gemmy H3 Save Sigmas",
    "GemmyH3LoadSigmas": "Gemmy H3 Load Sigmas",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
