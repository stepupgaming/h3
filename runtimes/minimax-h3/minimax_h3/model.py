"""MiniMax H3 audio-video DiT forward pass.

A 1:1 Python port of ``src/model.rs`` (which ports
``MiniMaxH3Model._forward`` from ComfyUI ``comfy/ldm/minimax/model.py``,
PR #15224). Shared numerics match the Candle runtime. Owner-box streaming
(RAM↔VRAM block offload, deferred int8, optional prefetch) lives here too —
those seams are runtime concerns, not golden-math changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .attention import AttnImpl
from .blocks import DiTBlock, FinalLayer, TokenRefiner
from .config import H3Config
from .layout import Keyframe, PackedLayout, RefBlock, SegKind, pad_latent_dims
from .rotary import rope_cos_sin, rope_freqs
from .timestep import TimeEmbedder, time_shift_sigma, time_shift_slope

VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def _dump_act(name: str, t: torch.Tensor) -> None:
    """Debug: dump an intermediate activation to `$MH3_DUMP_DIR/<name>.bin`
    as raw f32 (u64 ndim, ndim×u64 dims, then row-major f32) — byte-compatible
    with the Rust `dump_act` in src/model.rs. No-op unless the env var is set."""
    import os

    d = os.environ.get("MH3_DUMP_DIR")
    if not d:
        return
    t = t.detach().float().cpu().contiguous()
    dims = list(t.shape)
    vals = t.reshape(-1).numpy().astype("<f4")
    import struct

    buf = struct.pack(f"<Q{len(dims)}Q", len(dims), *dims)
    buf += vals.tobytes()
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.bin"), "wb") as f:
        f.write(buf)


@dataclass
class Payload:
    """Condition latents + sampling parameters handed to the model each step."""

    cond_video_latents: List[Tensor] = field(default_factory=list)
    cond_audio_latents: List[Tensor] = field(default_factory=list)
    visual_cond_noise_aug: float = VISUAL_COND_TIMESTEP
    audio_cond_noise_aug: float = AUDIO_COND_TIMESTEP
    seed: int = 0
    keyframes: List[Keyframe] = field(default_factory=list)
    refs: List[RefBlock] = field(default_factory=list)
    frame_count: Optional[int] = None
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0


# ---------------------------------------------------------------------------
# Patch / pack helpers (exact ports)


def patchify_video(latent: Tensor, patch) -> Tensor:
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]"""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    # nctrhpwq -> nthwcrpq: keep the (c, r, p, q) channel+patch block together at
    # the END of each row, so the token order is exactly (t, h, w). The earlier
    # permute (0, 2, 3, 4, 5, 6, 7, 1) scrambled the intra-token layout.
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows: Tensor, t: int, h: int, w: int, c: int, patch) -> Tensor:
    pt, ph, pw = patch
    n = rows.shape[0]
    b = n // (t * h * w)
    x = rows.reshape(b, t, h, w, c, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # nthwcrpq -> nctrhpwq
    return x.reshape(b, c, t * pt, h * ph, w * pw)


def pack_audio(latent: Tensor) -> Tensor:
    """[B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)"""
    b, c, ch, t = latent.shape
    if b != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    return latent.permute(0, 2, 3, 1).reshape(ch * t, c)


def unpack_audio(rows: Tensor, ch: int) -> Tensor:
    ch_t, c = rows.shape
    t = ch_t // ch
    x = rows.reshape(ch, t, c).permute(2, 0, 1)  # [C, ch, T]
    return x.unsqueeze(0)


def pad_to_patch_size(x: Tensor, patch) -> Tensor:
    """Zero-pad T/H/W up to multiples of the patch size (crop handled by
    caller)."""
    t, h, w = x.shape[2], x.shape[3], x.shape[4]
    tp, hp, wp = pad_latent_dims(t, h, w, patch)
    out = x
    if wp > w:
        out = torch.nn.functional.pad(out, (0, wp - w))
    if hp > h:
        out = torch.nn.functional.pad(out, (0, 0, 0, hp - h))
    if tp > t:
        out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, tp - t))
    return out


def aug_rows(rows: Tensor, aug: float, seed: int, dtype: torch.dtype) -> Tensor:
    """Condition-row noise augmentation, deterministic per seed (Box-Muller)."""
    if aug >= 1.0:
        return rows.to(dtype)
    from .weights import _randn_std_normal

    n, d = rows.shape
    noise = _randn_std_normal((n, d), seed).to(rows.device).to(dtype)
    rows = rows.to(dtype)
    return rows * aug + noise * (1.0 - aug)


# ---------------------------------------------------------------------------


class MiniMaxH3Model:
    def __init__(self, src, cfg: H3Config, attn_impl: str = AttnImpl.AUTO):
        from .weights import ships_curve_table, resolve_curve_grid

        self.cfg = cfg
        self.compute_dtype = src.compute_dtype()
        hidden = cfg.hidden_size
        t_dim = cfg.time_embed_dim

        from .blocks import linear, rms_norm

        self.video_patch_proj = linear(
            src, "video_patch_proj", hidden, cfg.video_patch_dim(), True, torch.float32
        )
        self.audio_patch_proj = linear(
            src, "audio_patch_proj", hidden, cfg.audio_latents_dim, True, torch.float32
        )
        self.condition_proj = linear(src, "condition_proj", hidden, cfg.text_dim, True, self.compute_dtype)

        # Curve semantics run whenever the checkpoint actually ships the table
        # (format probe) OR SVD pruning synthesizes one. Curve mode: adaln
        # projections run fp32 and skip the silu (the curve rows encode it).
        use_curves = cfg.runs_curves() or ships_curve_table(src) is not None
        adaln_dtype = torch.float32 if use_curves else self.compute_dtype
        apply_silu = not use_curves
        if use_curves:
            grid = resolve_curve_grid(src, cfg)
            if grid < 2:
                raise ValueError(f"curve grid must be >= 2, got {grid}")
            # The stored table width wins over t_dim: pre-projected tables
            # (the pruned int8 checkpoints ship `adaln_t_table` at
            # `[grid, rank]` — the SVD basis already applied) keep their
            # stored width, and `adaln_in` below follows it, so the per-block
            # adaln projections are requested at `[out, rank]` and match the
            # file. Mirrors the Rust loader's stored-width request.
            stored = src.stored_shape("adaln_t_table")
            width = stored[1] if stored is not None else t_dim
            self.adaln_t_table = src.get("adaln_t_table", (grid, width), torch.float32)
            self.time_embedder = None
        else:
            self.adaln_t_table = None
            self.time_embedder = TimeEmbedder(
                src, cfg.timestep_input_dim, cfg.time_embed_hidden_size, t_dim
            )
        # The adaln input width follows the loaded table.
        adaln_in = self.adaln_t_table.shape[1] if self.adaln_t_table is not None else t_dim

        # rope.inv_freq: from the checkpoint if present, else synthesize from
        # rope_theta (mirrors MiniMaxH3Model::new in src/model.rs).
        if src.stored_shape("rope.inv_freq") is None:
            n = cfg.rope_inv_freq_len
            vals = [1.0 / (cfg.rope_theta ** (2.0 * i / n)) for i in range(n)]
            self.rope_inv_freq = torch.tensor(vals, dtype=torch.float32, device=src.device())
        else:
            self.rope_inv_freq = src.get("rope.inv_freq", (cfg.rope_inv_freq_len,), torch.float32)

        self.token_refiner = TokenRefiner(
            src,
            cfg.token_refiner_num_layers,
            hidden,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.ffn_hidden_size,
            cfg.norm_eps,
            cfg.qk_norm_eps,
            cfg.final_norm_eps,
            self.compute_dtype,
            attn_impl,
        )
        self.blocks: List[DiTBlock] = []
        for i in range(cfg.num_layers):
            self.blocks.append(
                DiTBlock(
                    src,
                    f"blocks.{i}",
                    hidden,
                    cfg.num_attention_heads,
                    cfg.attention_head_dim,
                    cfg.ffn_hidden_size,
                    adaln_in,
                    cfg.norm_eps,
                    cfg.qk_norm_eps,
                    apply_silu,
                    self.compute_dtype,
                    adaln_dtype,
                    attn_impl,
                )
            )
        self.final_layer = FinalLayer(
            src,
            hidden,
            adaln_in,
            cfg.video_patch_dim(),
            cfg.audio_latents_dim,
            cfg.final_norm_eps,
            apply_silu,
            self.compute_dtype,
            adaln_dtype,
        )

        self._layout_cache: dict = {}
        self._rope_cache: dict = {}
        # Prefetch depth for streamed runs: 0 = sync H2D per block (legacy),
        # 1 = overlap next block transfer with current compute (default when
        # stream_gpu is set). Depth >1 keeps more blocks in flight (VRAM cost).
        self.stream_prefetch_depth: int = 1
        # When True, DiTBlock.adaln_proj weights are not moved with the block
        # (precomputed mods are already on device / installed on the proj).
        self._adaln_precomputed: bool = False
        self._transfer_stream: Optional[torch.cuda.Stream] = None

    def to(self, device=None, dtype=None):
        for p in (self.video_patch_proj, self.audio_patch_proj, self.condition_proj):
            p.to(device=device, dtype=dtype)
        if self.time_embedder is not None:
            self.time_embedder.to(device=device, dtype=dtype)
        if self.adaln_t_table is not None:
            self.adaln_t_table = self.adaln_t_table.to(device=device, dtype=dtype)
        self.rope_inv_freq = self.rope_inv_freq.to(device=device, dtype=dtype)
        self.token_refiner.to(device=device, dtype=dtype)
        for b in self.blocks:
            b.to(device=device, dtype=dtype)
        self.final_layer.to(device=device, dtype=dtype)
        return self

    def pin_stream_hosts(
        self,
        reserve_gb: float = 8.0,
        max_pin_gb: float = 4.0,
    ) -> dict:
        """Pin some deferred int8 host storages before a streamed sample.

        Pinned pages are locked in physical RAM (Windows Task Manager "In use"
        climbs hard). We deliberately do **not** pin the whole ~20 GB weight
        set — only enough for a faster upload runway.

        ``reserve_gb``: never pin if free RAM would fall below this (default 8).
        ``max_pin_gb``: hard cap on how many GB of weight bytes we pin (default 4).
        Remaining layers stay ordinary pageable CPU memory — still correct,
        H2D may be a bit slower; quality unchanged. ``max_pin_gb=0`` skips
        pinning entirely. ``max_pin_gb < 0`` means no byte cap (reserve only).
        """
        import gc

        def _mem() -> tuple[float | None, float | None, float | None]:
            """(available_gb, used_gb, total_gb)"""
            try:
                import psutil

                vm = psutil.virtual_memory()
                return (
                    vm.available / (1024**3),
                    vm.used / (1024**3),
                    vm.total / (1024**3),
                )
            except Exception:
                return None, None, None

        def _deferred_bytes(obj) -> int:
            """Rough host payload size for budget accounting."""
            n = 0
            stack = [obj]
            seen: set[int] = set()
            while stack:
                o = stack.pop()
                if o is None or id(o) in seen:
                    continue
                seen.add(id(o))
                for attr in ("deferred_int8",):
                    d = getattr(o, attr, None)
                    if d is not None and hasattr(d, "host_parts"):
                        try:
                            for _attr, t in d.host_parts():
                                if t is not None:
                                    n += int(t.numel()) * int(t.element_size())
                        except Exception:
                            pass
                b = getattr(o, "b", None)
                if isinstance(b, torch.Tensor) and b.device.type == "cpu":
                    n += int(b.numel()) * int(b.element_size())
                for name in (
                    "qkv_proj",
                    "out_proj",
                    "fc1",
                    "fc2",
                    "linear",
                    "attn",
                    "mlp",
                    "adaln_proj",
                    "blocks",
                    "video_out",
                    "audio_out",
                ):
                    child = getattr(o, name, None)
                    if child is None:
                        continue
                    if isinstance(child, (list, tuple)):
                        stack.extend(child)
                    else:
                        stack.append(child)
            return n

        pinned = 0
        pageable = 0
        pinned_bytes = 0
        stop_reason = None  # None | "reserve" | "max_pin" | "no_pin"
        avail_before, used_before, total_gb = _mem()

        if max_pin_gb == 0:
            stop_reason = "no_pin"

        def _can_pin_obj(obj) -> bool:
            nonlocal stop_reason
            if stop_reason is not None:
                return False
            need = _deferred_bytes(obj)
            # Pin path allocates a second copy briefly → need free RAM for
            # reserve + this tensor (+ small slack).
            avail, _, _ = _mem()
            if avail is not None and reserve_gb > 0:
                if avail < float(reserve_gb) + need / (1024**3) + 0.5:
                    stop_reason = "reserve"
                    return False
            if max_pin_gb >= 0:
                if (pinned_bytes + need) / (1024**3) > float(max_pin_gb) + 1e-9:
                    stop_reason = "max_pin"
                    return False
            return True

        def _pin_or_adopt(obj) -> None:
            nonlocal pinned, pageable, pinned_bytes
            if obj is None:
                return
            if _can_pin_obj(obj) and hasattr(obj, "pin_host"):
                before = _deferred_bytes(obj)
                obj.pin_host()
                pinned += 1
                pinned_bytes += before
                gc.collect()
                return
            if hasattr(obj, "adopt_host_pageable"):
                obj.adopt_host_pageable()
            pageable += 1

        for p in (self.video_patch_proj, self.audio_patch_proj, self.condition_proj):
            _pin_or_adopt(p)
        _pin_or_adopt(self.token_refiner)
        for b in self.blocks:
            _pin_or_adopt(b)
        for p in (self.final_layer.video_out, self.final_layer.audio_out):
            _pin_or_adopt(p)
        _pin_or_adopt(self.final_layer.adaln_proj)

        gc.collect()
        avail_after, used_after, _ = _mem()
        return {
            "pinned_groups": pinned,
            "pageable_groups": pageable,
            "pinned_gb": pinned_bytes / (1024**3),
            "max_pin_gb": float(max_pin_gb),
            "reserve_gb": float(reserve_gb),
            "stop_reason": stop_reason,
            "stopped_for_reserve": stop_reason == "reserve",
            "avail_gb_before": avail_before,
            "avail_gb_after": avail_after,
            "used_gb_before": used_before,
            "used_gb_after": used_after,
            "total_gb": total_gb,
        }

    def prepare_text_states(
        self,
        text: Tensor,
        stream_gpu: Optional[str] = None,
    ) -> Tensor:
        """Project + refine text once to ``[L, hidden]`` (lossless cache).

        If ``text`` is already ``[L, hidden_size]``, returns it cast to the
        compute dtype on the target device. Otherwise runs ``condition_proj``
        and the token refiner (streamed when ``stream_gpu`` is set) exactly as
        ``forward`` would on the first step, then leaves refiner weights on CPU.

        The multi-step sampler should call this once and pass the result as
        ``text`` every denoise step so proj+refiner are not repeated.
        """
        dtype = self.compute_dtype
        dev = stream_gpu or str(text.device)
        if text.shape[1] == self.cfg.hidden_size:
            return text.to(device=dev, dtype=dtype)

        if text.shape[1] != self.cfg.text_dim:
            raise ValueError(
                f"text last dim must be text_dim={self.cfg.text_dim} or "
                f"hidden_size={self.cfg.hidden_size}, got {text.shape[1]}"
            )

        # Resident pieces needed for text path.
        self.condition_proj.to(dev)
        self.token_refiner.final_norm.to(dev)
        x = text.to(device=dev, dtype=dtype)
        proj = self.condition_proj.forward(x)
        if stream_gpu is not None:
            for rb in self.token_refiner.blocks:
                rb.to(stream_gpu)
                proj = rb.forward(proj, None, None)
                rb.to("cpu")
            states = self.token_refiner.final_norm.forward(proj)
        else:
            self.token_refiner.to(dev)
            states = self.token_refiner.forward(proj)
            # Keep parity with stream path: refiner blocks need not stay resident
            # for the denoise loop (only used again if someone re-prepares text).
            for rb in self.token_refiner.blocks:
                rb.to("cpu")
        return states

    def clear_adaln_precompute(self) -> None:
        """Drop any installed AdaLN precompute and restore weight streaming."""
        for b in self.blocks:
            b.adaln_proj.clear_precomputed()
        self.final_layer.adaln_proj.clear_precomputed()
        self._adaln_precomputed = False

    def precompute_adaln_for_t_emb(self, t_emb: Tensor, device: Optional[str] = None) -> None:
        """Lossless AdaLN precompute for one step's ``t_emb`` [M, t_dim].

        Runs each block's (and the final layer's) adaln projection once, stores
        the modulation rows on the proj, and marks adaln weights skippable in
        the stream plan. Same arithmetic as computing mods inside each block
        forward — just hoisted so the big adaln linears are not re-read every
        layer from a cold stream.

        Call once per denoising step (t_emb changes with sigma). Safe under
        curve mode (fp32 island) and classic silu mode.
        """
        dev = device or str(t_emb.device)
        t_emb = t_emb.to(dev)
        # Clear any previous step's cache so forward actually runs the linear.
        for b in self.blocks:
            b.adaln_proj.clear_precomputed()
        self.final_layer.adaln_proj.clear_precomputed()
        self._adaln_precomputed = False

        # Keep tiny AdaLN linears resident on the compute device for the sample
        # when streaming — bouncing 50 small projs CPU↔GPU every step is pure
        # tax. Mods are cast once to the stream dtype so mod_* can skip per-row
        # casts (bit-identical once rounded to compute_dtype).
        cdtype = self.compute_dtype
        for b in self.blocks:
            b.adaln_proj.to(dev)
            mods = b.adaln_proj.forward(t_emb)
            mods = [m.detach().to(dtype=cdtype) for m in mods]
            b.adaln_proj.set_precomputed(mods)
        self.final_layer.adaln_proj.to(dev)
        fmods = self.final_layer.adaln_proj.forward(t_emb)
        fmods = [m.detach().to(dtype=cdtype) for m in fmods]
        self.final_layer.adaln_proj.set_precomputed(fmods)
        self._adaln_precomputed = True

    def _block_to(
        self,
        block: DiTBlock,
        device,
        non_blocking: bool = False,
        skip_adaln: bool = False,
    ) -> None:
        """Move a DiT block, optionally skipping adaln_proj weights."""
        block.norm1.to(device=device)
        block.norm2.to(device=device)
        block.attn.to(device=device, non_blocking=non_blocking)
        block.mlp.to(device=device, non_blocking=non_blocking)
        if not skip_adaln:
            block.adaln_proj.to(device=device, non_blocking=non_blocking)
        elif block.adaln_proj._precomputed is not None and device is not None:
            # Keep precomputed mods on the compute device.
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type != "cpu":
                block.adaln_proj._precomputed = [
                    t.to(device=dev, non_blocking=non_blocking)
                    for t in block.adaln_proj._precomputed
                ]

    # -- layout / rope helpers (cached by signature, like the Rust port) ----

    def layout(self, text_len, latent_t, lat_h, lat_w, audio_t, payload: Payload) -> PackedLayout:
        # Cache key must match PackedLayout.signature (includes refs fingerprint).
        # Keying only on len(refs) collided distinct Ref2VA packs.
        refs_sig = tuple(
            (
                int(blk.kind.value),
                int(blk.latent_t),
                int(blk.latent_h),
                int(blk.latent_w),
                int(blk.ref_audio_t),
            )
            for blk in payload.refs
        )
        sig = (
            int(text_len),
            int(latent_t),
            int(lat_h),
            int(lat_w),
            int(audio_t),
            len(payload.keyframes),
            int(payload.frame_count or 0),
            refs_sig,
        )
        if sig in self._layout_cache:
            return self._layout_cache[sig]
        layout = PackedLayout.new(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            payload.keyframes,
            payload.refs,
            payload.frame_count,
        )
        self._layout_cache[sig] = layout
        return layout

    def live_packed_seq_len(self, video, audio, text, payload: Payload) -> int:
        vd = video.shape
        tp, hp, wp = pad_latent_dims(vd[2], vd[3], vd[4], self.cfg.patch_size)
        return self.layout(text.shape[0], tp, hp, wp, audio.shape[3], payload).seq_len

    def rope_cos_sin(self, layout: PackedLayout, device, dtype: torch.dtype):
        sig = (layout.signature, str(device))
        if sig in self._rope_cache:
            return self._rope_cache[sig]
        ids = layout.position_ids_tensor(device)
        inv = self.rope_inv_freq.to(device)
        freqs = rope_freqs(ids, inv)
        cos, sin = rope_cos_sin(freqs, dtype)
        self._rope_cache[sig] = (cos, sin)
        return cos, sin

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        text: Tensor,
        text_tags: Optional[Sequence[int]] = None,
        sigma: float = 0.5,
        payload: Optional[Payload] = None,
        stream_gpu: Optional[str] = None,
        precompute_adaln: bool = False,
    ):
        """One denoising step. `video` [1, 24, T, H, W], `audio` [1, 32, 2, Ta],
        `text` [L, text_dim] Qwen3-VL hidden states (or [L, hidden] to skip the
        condition projection + refiner), `sigma` in [0, 1] (the video-flow
        sigma from the sampler).

        `stream_gpu` (e.g. "cuda") runs the block-by-block offload path: the
        resident pieces (patch/condition projections, time embedder / curve
        table, rope, the token refiner's final norm, the final layer) migrate
        to the GPU once, and every DiT / refiner block is moved to the GPU for
        its own forward and back to CPU after — the PyTorch mirror of the
        Candle runtime's `forward_streamed` engine, so a full 33B checkpoint
        stays in RAM and only one block (+ optional in-flight prefetch) is
        ever resident in VRAM. Prefetch depth is ``self.stream_prefetch_depth``.

        `precompute_adaln`: when True, project AdaLN mods once for this step's
        ``t_emb`` and skip streaming adaln weights per block (lossless).

        Returns (video_out [1, 24, T, H, W], audio_out [1, 32, 2, Ta]) with the
        reference's sign/slope conventions (velocity-like output).
        """
        from .profile import get_profiler

        prof = get_profiler()

        if payload is None:
            payload = Payload()
        if stream_gpu is not None:
            # Resident pieces migrate once; the block stacks stream below.
            for p in (self.video_patch_proj, self.audio_patch_proj, self.condition_proj):
                p.to(stream_gpu)
            if self.time_embedder is not None:
                self.time_embedder.to(stream_gpu)
            if self.adaln_t_table is not None:
                self.adaln_t_table = self.adaln_t_table.to(stream_gpu)
            self.rope_inv_freq = self.rope_inv_freq.to(stream_gpu)
            self.token_refiner.final_norm.to(stream_gpu)
            # Final layer: always need norm + heads; adaln may be precomputed.
            self.final_layer.norm.to(stream_gpu)
            self.final_layer.video_out.to(stream_gpu)
            self.final_layer.audio_out.to(stream_gpu)
            if not precompute_adaln and not self._adaln_precomputed:
                self.final_layer.adaln_proj.to(stream_gpu)
        device = video.device
        dtype = self.compute_dtype
        orig_t, orig_h, orig_w = video.shape[2], video.shape[3], video.shape[4]
        video_x = pad_to_patch_size(video, self.cfg.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio.shape[3]
        text_len = text.shape[0]

        layout = self.layout(text_len, latent_t, lat_h, lat_w, audio_t, payload)

        return self._forward_body(
            video=video,
            audio=audio,
            text=text,
            text_tags=text_tags,
            sigma=sigma,
            payload=payload,
            stream_gpu=stream_gpu,
            precompute_adaln=precompute_adaln,
            layout=layout,
            device=device,
            dtype=dtype,
            orig_t=orig_t,
            orig_h=orig_h,
            orig_w=orig_w,
            video_x=video_x,
            latent_t=latent_t,
            lat_h=lat_h,
            lat_w=lat_w,
            audio_t=audio_t,
            text_len=text_len,
            prof=prof,
        )

    def _forward_body(
        self,
        *,
        video: Tensor,
        audio: Tensor,
        text: Tensor,
        text_tags: Optional[Sequence[int]],
        sigma: float,
        payload: Payload,
        stream_gpu: Optional[str],
        precompute_adaln: bool,
        layout: PackedLayout,
        device,
        dtype: torch.dtype,
        orig_t: int,
        orig_h: int,
        orig_w: int,
        video_x: Tensor,
        latent_t: int,
        lat_h: int,
        lat_w: int,
        audio_t: int,
        text_len: int,
        prof,
    ):
        shift_v = payload.sigma_shift_video
        shift_a = payload.sigma_shift_audio
        sigma_v = max(sigma, 1e-6)
        t_v = 1.0 - sigma_v
        t_a = 1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)

        vis_aug = payload.visual_cond_noise_aug
        aud_aug = payload.audio_cond_noise_aug
        has_vis_cond = any(k in (SegKind.COND, SegKind.REF_IMG) for (_, _, k) in layout.segments)
        has_aud_cond = any(k == SegKind.REF_AUDIO for (_, _, k) in layout.segments)

        def seg_t(kind: SegKind) -> float:
            if kind in (SegKind.TEXT, SegKind.VIDEO):
                return t_v
            if kind == SegKind.AUDIO:
                return t_a
            if kind in (SegKind.COND, SegKind.REF_IMG):
                return max(t_v, vis_aug)
            return max(t_a, aud_aug)  # REF_AUDIO

        unique_t = sorted({t_v, t_a} | ({max(t_v, vis_aug)} if has_vis_cond else set()) | ({max(t_a, aud_aug)} if has_aud_cond else set()))

        def t_row(t: float) -> int:
            return min(range(len(unique_t)), key=lambda i: abs(unique_t[i] - t))

        def mod_row(kind: SegKind) -> int:
            return t_row(seg_t(kind)) * 3 + kind.tag()

        # mod segments (text split into tag runs when tags are given)
        mod_segments: List[Tuple[int, int, int]] = []
        for a, b, kind in layout.segments:
            if kind == SegKind.TEXT and text_tags is not None:
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or text_tags[i] != text_tags[run_start]:
                        mod_segments.append((a + run_start, a + i, t_row(seg_t(kind)) * 3 + int(text_tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, mod_row(kind)))

        # ---- embed rows ---------------------------------------------------
        def _embed() -> Tensor:
            video_rows = patchify_video(video_x.to(torch.float32), self.cfg.patch_size)
            audio_rows = pack_audio(audio.to(torch.float32))

            vcond_iter = iter(payload.cond_video_latents)
            acond_iter = iter(payload.cond_audio_latents)

            parts: List[Tensor] = []
            for _a, _b, kind in layout.segments:
                if kind == SegKind.TEXT:
                    if text.shape[1] != self.cfg.hidden_size:
                        proj = self.condition_proj.forward(text.to(dtype))
                        if stream_gpu is not None:
                            for rb in self.token_refiner.blocks:
                                rb.to(stream_gpu)
                                proj = rb.forward(proj, None, None)
                                rb.to("cpu")
                            states = self.token_refiner.final_norm.forward(proj)
                        else:
                            states = self.token_refiner.forward(proj)
                    else:
                        states = text.to(dtype)
                    parts.append(states)
                elif kind in (SegKind.COND, SegKind.REF_IMG):
                    z = next(vcond_iter)
                    rows = patchify_video(z.to(torch.float32), self.cfg.patch_size)
                    rows = aug_rows(rows, vis_aug, payload.seed, dtype)
                    emb = self.video_patch_proj.forward(rows.to(torch.float32)).to(dtype)
                    parts.append(emb)
                elif kind == SegKind.VIDEO:
                    emb = self.video_patch_proj.forward(video_rows).to(dtype)
                    parts.append(emb)
                elif kind == SegKind.REF_AUDIO:
                    z = next(acond_iter)
                    rows = pack_audio(z.to(torch.float32))
                    rows = aug_rows(rows, aud_aug, payload.seed + 1, dtype)
                    emb = self.audio_patch_proj.forward(rows.to(torch.float32)).to(dtype)
                    parts.append(emb)
                else:  # AUDIO
                    emb = self.audio_patch_proj.forward(audio_rows).to(dtype)
                    parts.append(emb)
            return torch.cat(parts, dim=0)

        if prof is not None and prof.enabled:
            with prof.phase("pack_embed"):
                h = _embed()
        else:
            h = _embed()
        _dump_act("h0", h)

        # ---- timestep embedding -------------------------------------------
        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        _dump_act("t_vals", t_vals)
        if self.adaln_t_table is not None:
            from .weights import lerp_curve

            t_emb = lerp_curve(self.adaln_t_table.to(device), t_vals)
        else:
            t_emb = self.time_embedder.forward(t_vals).to(dtype)
        _dump_act("t_emb", t_emb)

        if precompute_adaln:
            if prof is not None and prof.enabled:
                with prof.phase("adaln_precompute"):
                    self.precompute_adaln_for_t_emb(t_emb, device=stream_gpu or str(device))
            else:
                self.precompute_adaln_for_t_emb(t_emb, device=stream_gpu or str(device))

        skip_adaln = self._adaln_precomputed

        rope_cos, rope_sin = self.rope_cos_sin(layout, device, dtype)

        # ---- DiT blocks (optional streamed + prefetched) ------------------
        n_blocks = len(self.blocks)
        use_prefetch = (
            stream_gpu is not None
            and self.stream_prefetch_depth > 0
            and str(stream_gpu).startswith("cuda")
            and torch.cuda.is_available()
        )

        if stream_gpu is None:
            for i, block in enumerate(self.blocks):
                h = block.forward(h, t_emb, mod_segments, rope_cos, rope_sin)
                _dump_act(f"h_block_{i:03d}", h)
        elif not use_prefetch:
            for i, block in enumerate(self.blocks):
                if prof is not None and prof.enabled:
                    with prof.phase("h2d"):
                        self._block_to(block, stream_gpu, skip_adaln=skip_adaln)
                else:
                    self._block_to(block, stream_gpu, skip_adaln=skip_adaln)
                h = block.forward(h, t_emb, mod_segments, rope_cos, rope_sin)
                _dump_act(f"h_block_{i:03d}", h)
                if prof is not None and prof.enabled:
                    with prof.phase("d2h"):
                        self._block_to(block, "cpu", skip_adaln=skip_adaln)
                else:
                    self._block_to(block, "cpu", skip_adaln=skip_adaln)
        else:
            # Prefetch depth D = how many DiT blocks may sit on the GPU:
            # current compute block + up to (D-1) already-loaded ahead, while
            # the next one uploads. Past blocks free once their compute event
            # is complete — preferably without a host stall on the hot path.
            #   D=1 → classic: compute i, load i+1 under it (~2 resident)
            #   D=2..N → longer runway so PCIe is less likely to stall compute
            # Host tensors are pinned by Linear.pin_host when RAM allows.
            #
            # Sync policy:
            # - ready/done are CUDA Events (compute waits via wait_event)
            # - kick H2D of i+D before waiting on ready[i]
            # - free completed blocks via event.query() when possible; only
            #   host-synchronize when VRAM pressure requires it (backlog)
            # - profiler h2d_enqueue/d2h use host timing (sync=False) so
            #   measuring does not destroy overlap
            depth = max(1, int(self.stream_prefetch_depth))
            if self._transfer_stream is None:
                self._transfer_stream = torch.cuda.Stream()
            tstream = self._transfer_stream
            compute_stream = torch.cuda.current_stream()

            def h2d_async(idx: int) -> None:
                with torch.cuda.stream(tstream):
                    self._block_to(
                        self.blocks[idx],
                        stream_gpu,
                        non_blocking=True,
                        skip_adaln=skip_adaln,
                    )

            def release_block(idx: int) -> None:
                # Restore host int8 pointers / drop device weight storages.
                # Caller must ensure compute on this block has finished.
                self._block_to(self.blocks[idx], "cpu", skip_adaln=skip_adaln)

            def record_ready(idx: int) -> None:
                with torch.cuda.stream(tstream):
                    ready_events[idx].record(tstream)

            ready_events = [torch.cuda.Event() for _ in range(n_blocks)]
            done_events = [torch.cuda.Event() for _ in range(n_blocks)]
            # Indices whose compute finished recording but device storage may
            # still be live. Pop when event is ready (non-blocking) so free
            # rarely hits the host critical path.
            free_q: List[int] = []

            def drain_free(*, block: bool) -> None:
                """Release finished blocks. ``block=False`` only pops ready ones."""
                while free_q:
                    idx = free_q[0]
                    if not block and not done_events[idx].query():
                        return
                    if block:
                        done_events[idx].synchronize()
                    free_q.pop(0)
                    if prof is not None and prof.enabled:
                        with prof.phase("d2h", sync=False):
                            release_block(idx)
                    else:
                        release_block(idx)

            # Prime first `depth` blocks so compute 0 starts with runway.
            prime_n = min(depth, n_blocks)
            if prof is not None and prof.enabled:
                with prof.phase("h2d"):
                    for j in range(prime_n):
                        h2d_async(j)
                        record_ready(j)
                    ready_events[0].synchronize()
            else:
                for j in range(prime_n):
                    h2d_async(j)
                    record_ready(j)
                ready_events[0].synchronize()

            for i in range(n_blocks):
                # Kick the block entering the far end of the prefetch runway
                # *before* waiting on this block's weights (overlap).
                nxt = i + depth
                if nxt < n_blocks:
                    if prof is not None and prof.enabled:
                        with prof.phase("h2d_enqueue", sync=False):
                            h2d_async(nxt)
                            record_ready(nxt)
                    else:
                        h2d_async(nxt)
                        record_ready(nxt)

                # Weights for this block must be on device before GEMMs.
                compute_stream.wait_event(ready_events[i])

                # Free any earlier blocks whose compute is already done. Keep at
                # most one completed block live (i-1) so VRAM stays for ahead
                # loads; if the free queue grows (slow free / fast compute),
                # block on the oldest to bound residency.
                if i > 0:
                    free_q.append(i - 1)
                drain_free(block=False)
                # Bound live completed blocks: if still holding more than 1
                # finished block, host-wait the oldest (rare when overlap works).
                while len(free_q) > 1:
                    drain_free(block=True)

                h = self.blocks[i].forward(h, t_emb, mod_segments, rope_cos, rope_sin)
                _dump_act(f"h_block_{i:03d}", h)
                done_events[i].record(compute_stream)

            # Drain compute + free for remaining blocks.
            if n_blocks > 0:
                free_q.append(n_blocks - 1)
            drain_free(block=True)

        # ---- final layer --------------------------------------------------
        video_seg = next((a, b, t_row(seg_t(SegKind.VIDEO))) for (a, b, k) in layout.segments if k == SegKind.VIDEO)
        audio_seg = next((a, b, t_row(seg_t(SegKind.AUDIO))) for (a, b, k) in layout.segments if k == SegKind.AUDIO)
        if skip_adaln and self.final_layer.adaln_proj._precomputed is not None:
            # mods already on device from precompute
            pass
        elif stream_gpu is not None:
            self.final_layer.adaln_proj.to(stream_gpu)
        if prof is not None and prof.enabled:
            with prof.phase("final"):
                v, a = self.final_layer.forward(h, t_emb, video_seg, audio_seg)
        else:
            v, a = self.final_layer.forward(h, t_emb, video_seg, audio_seg)
        _dump_act("h_blocks", h)
        _dump_act("v", v)
        _dump_act("a", a)

        video_out = unpatchify_video(
            v, latent_t, lat_h // 2, lat_w // 2, self.cfg.latents_dim, self.cfg.patch_size
        )
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a, 2)

        slope_a = time_shift_slope(sigma_v, shift_v, shift_a)
        # Reference returns outputs cast back to the input latent dtypes.
        video_out = (-video_out).to(video.dtype)
        audio_out = (-slope_a * audio_out).to(audio.dtype)
        return video_out, audio_out


__all__ = ["MiniMaxH3Model", "Payload", "patchify_video", "unpatchify_video", "pack_audio", "unpack_audio"]
