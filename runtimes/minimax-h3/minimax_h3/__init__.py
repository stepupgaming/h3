"""MiniMax H3 — standalone PyTorch reference + owner-box production sample path.

A 1:1 Python port of the Candle crate in ``src/`` (itself a port of ComfyUI
PR #15224), independent of Comfy. Readable reference for parity **and** the
production runtime used on the owner 16 GB / 64 GB box (streamed pruned int8 +
SageAttention).

**Library surface for product / gemmy integration**

- :class:`SampleRequest` / :func:`run_sample` — full DiT sample (latents out)
- :func:`build_conditioning` / loaders — FL2VA / Ref2VA packing
- :mod:`minimax_h3.frame_grid` — 17k+5 snap, latent_t, flow_sigmas
- :class:`MiniMaxH3Model` / :class:`Payload` — single-step forward

TE encode, VAE encode/decode, and multi-window continue remain under
``scripts/`` (separate VRAM sequencing); call those as subprocesses or import
helpers from the scripts until they grow package adapters.
"""

from .attention import AttnImpl
from .conditioning import (
    ConditioningError,
    ConditioningPlan,
    build_conditioning,
    load_cond_audio_latent,
    load_cond_video_latent,
    load_text_embeds,
)
from .config import H3Config
from .frame_grid import (
    FPS,
    TemporalShape,
    audio_t_from_frames,
    flow_sigmas,
    floor_frames,
    latent_t_from_frames,
    snap_frames,
    temporal_shape,
)
from .layout import Keyframe, PackedLayout, RefBlock, RefKind, SegKind
from .model import MiniMaxH3Model, Payload
from .prompt_ir import compile_from_brief, compile_prompt, is_ir_format
from .sample import SampleError, SampleRequest, SampleResult, euler_denoise, run_sample

__version__ = "0.1.0"

__all__ = [
    "H3Config",
    "MiniMaxH3Model",
    "Payload",
    "PackedLayout",
    "SegKind",
    "RefKind",
    "RefBlock",
    "Keyframe",
    "AttnImpl",
    "SampleRequest",
    "SampleResult",
    "SampleError",
    "run_sample",
    "euler_denoise",
    "ConditioningPlan",
    "ConditioningError",
    "build_conditioning",
    "load_text_embeds",
    "load_cond_video_latent",
    "load_cond_audio_latent",
    "FPS",
    "TemporalShape",
    "snap_frames",
    "floor_frames",
    "latent_t_from_frames",
    "audio_t_from_frames",
    "temporal_shape",
    "flow_sigmas",
    "compile_from_brief",
    "compile_prompt",
    "is_ir_format",
]
