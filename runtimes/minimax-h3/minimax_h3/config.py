"""Architecture configuration for the MiniMax H3 audio-video DiT.

A 1:1 Python port of ``src/config.rs`` (which itself mirrors the
``MiniMaxH3Model.__init__`` defaults in ComfyUI's
``comfy/ldm/minimax/model.py``, PR #15224). Keep this file in lockstep with
the Rust config so the PyTorch reference and the Candle runtime share one
architecture definition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CONFIG_JSON = Path(__file__).resolve().parent.parent / "configs" / "h3.json"


@dataclass
class H3Config:
    # Non-default fields first (Python dataclass ordering); the defaults then
    # mirror the serde defaults in src/config.rs.
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    ffn_hidden_size: int
    latents_dim: int
    audio_latents_dim: int
    text_dim: int
    token_refiner_num_layers: int = 2
    patch_size: list = field(default_factory=lambda: [1, 2, 2])
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0
    rope_theta: float = 10000.0
    # Curve-form checkpoints: a shared [grid, time_embed_dim] fp32 basis
    # (`adaln_t_table`) replaces the time embedder + full-width adaln weights.
    adaln_curve_grid: Optional[int] = None
    # Load-time SVD pruning of the per-block adaln projections (rank R).
    adaln_svd_rank: Optional[int] = None

    # -- derived -----------------------------------------------------------

    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    def video_patch_dim(self) -> int:
        p = self.patch_size
        return self.latents_dim * p[0] * p[1] * p[2]

    def use_adaln_curves(self) -> bool:
        return self.adaln_curve_grid is not None

    def runs_curves(self) -> bool:
        """Model runs curve semantics when the checkpoint ships the table OR
        SVD pruning synthesizes one."""
        return self.adaln_curve_grid is not None or self.adaln_svd_rank is not None

    def curve_grid(self) -> int:
        """Grid rows for the curve table; the default used when SVD pruning
        synthesizes a table from a classic (time-embedder) checkpoint."""
        return self.adaln_curve_grid or 64

    # -- constructors ------------------------------------------------------

    @classmethod
    def full(cls) -> "H3Config":
        """The real H3 architecture (from model.py defaults / configs/h3.json)."""
        return cls.from_json(_CONFIG_JSON)

    @classmethod
    def from_json(cls, path: Path | str) -> "H3Config":
        with open(path) as f:
            raw = json.load(f)
        return cls(**raw)

    @classmethod
    def tiny(cls) -> "H3Config":
        """Smoke-test dimensions: every shape path runs, but the model fits on
        CPU. Mirrors ``MiniMaxH3Config::tiny`` in src/config.rs."""
        return cls(
            hidden_size=128,
            num_layers=2,
            token_refiner_num_layers=1,
            num_attention_heads=4,
            attention_head_dim=32,
            ffn_hidden_size=256,
            latents_dim=24,
            audio_latents_dim=32,
            patch_size=[1, 2, 2],
            text_dim=64,
            timestep_input_dim=32,
            time_embed_hidden_size=128,
            time_embed_dim=64,
            rope_inv_freq_len=4,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
            rope_theta=10000.0,
            adaln_curve_grid=None,
            adaln_svd_rank=None,
        )
