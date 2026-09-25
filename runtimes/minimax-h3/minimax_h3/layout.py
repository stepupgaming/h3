"""Packed-sequence layout for the H3 DiT.

A 1:1 Python port of ``src/layout.rs``, which is itself a 1:1 port of
``PackedLayout`` in ComfyUI's ``comfy/ldm/minimax/model.py`` (PR #15224).

The packed sequence is ``[text | cond/ref rows | audio | video]``; every row
carries a 3D (t, h, w) coordinate that becomes the split-half RoPE input. All
grid math runs in f64 exactly like the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

FRAME_PER_TOKEN: List[float] = [1.0, 4.0, 4.0, 4.0, 4.0]
FRAME_RESCALE: float = 5.0 / 3.0


def pad_latent_dims(t: int, h: int, w: int, patch) -> Tuple[int, int, int]:
    """Ceil t/h/w up to multiples of the patch size — exactly what the
    forward's `pad_to_patch_size` does, so the packed-layout dims always match
    the forward's own layout. Shared by the plan-time `packed_seq_len` and the
    live re-check, so the assumed and live sequence lengths can never diverge
    over the padding math (mirrors `layout::pad_latent_dims`)."""
    tp = (t + patch[0] - 1) // patch[0] * patch[0]
    hp = (h + patch[1] - 1) // patch[1] * patch[1]
    wp = (w + patch[2] - 1) // patch[2] * patch[2]
    return tp, hp, wp


class SegKind(Enum):
    TEXT = 0
    COND = 1
    REF_IMG = 2
    REF_AUDIO = 3
    AUDIO = 4
    VIDEO = 5

    def name(self) -> str:  # type: ignore[override]
        return {
            SegKind.TEXT: "text",
            SegKind.COND: "cond",
            SegKind.REF_IMG: "ref_img",
            SegKind.REF_AUDIO: "ref_audio",
            SegKind.AUDIO: "audio",
            SegKind.VIDEO: "video",
        }[self]

    def tag(self) -> int:
        """Modality index into the [M*3, hidden] adaln rows."""
        if self is SegKind.TEXT:
            return 1
        if self in (SegKind.VIDEO, SegKind.COND, SegKind.REF_IMG):
            return 0
        return 2  # AUDIO | REF_AUDIO


@dataclass(frozen=True)
class Keyframe:
    resolved_frame_index: int


class RefKind(Enum):
    IMAGE = 0
    AUDIO = 1
    VIDEO = 2  # pure-video ref blocks are accepted by the reference
    VIDEO_AUDIO = 3


@dataclass
class RefBlock:
    kind: RefKind
    latent_t: int
    latent_h: int
    latent_w: int
    ref_audio_t: int


@dataclass
class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    seq_len: int
    position_ids: List[float]  # [seq_len, 3] f64 (t, h, w)
    img_update: List[bool]
    audio_update: List[bool]
    segments: List[Tuple[int, int, SegKind]]
    signature: Tuple

    def position_ids_tensor(self, device=None, dtype=None):
        import torch

        t = torch.tensor(self.position_ids, dtype=torch.float32, device=device).reshape(
            self.seq_len, 3
        )
        return t.to(dtype) if dtype is not None else t

    @classmethod
    def new(
        cls,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        keyframes: Sequence[Keyframe] = (),
        refs: Sequence[RefBlock] = (),
        frame_count: Optional[int] = None,
    ) -> "PackedLayout":
        frame, w_grid = frame_grid(latent_h, latent_w)
        frame_rows = len(frame) // 2

        segments: List[Tuple[int, int, SegKind]] = []
        pos: List[List[float]] = []
        img_pos: List[int] = []
        img_update: List[bool] = []
        audio_pos: List[int] = []
        audio_update: List[bool] = []

        cursor = float(text_len)
        row = text_len

        # ---- text -------------------------------------------------------
        g = []
        for i in range(text_len):
            g.extend([float(i), 0.0, 0.0])
        segments.append((0, text_len, SegKind.TEXT))
        pos.append(g)

        # ---- keyframes (fl2va: cond rows right after text) -------------
        for kf in keyframes:
            if kf.resolved_frame_index == 0:
                cond_t = float(text_len)
            elif frame_count is not None:
                if kf.resolved_frame_index == frame_count - 1:
                    cond_t = float(text_len) + sum(video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
            else:
                raise ValueError("keyframe layout requires frame_count")
            g = []
            for r in range(frame_rows):
                g.extend([cond_t, frame[r * 2], frame[r * 2 + 1]])
            segments.append((row, row + frame_rows, SegKind.COND))
            pos.append(g)
            for i in range(frame_rows):
                img_pos.append(row + i)
                img_update.append(False)
            row += frame_rows

        target_audio_w = (w_grid[0], w_grid[-1])

        # ---- refs --------------------------------------------------------
        if refs:
            cursor = float(text_len)
            for blk in refs:
                if blk.kind is RefKind.IMAGE:
                    r_frame, _ = frame_grid(blk.latent_h, blk.latent_w)
                    n = len(r_frame) // 2
                    g = []
                    for r in range(n):
                        g.extend([cursor, r_frame[r * 2], r_frame[r * 2 + 1]])
                    segments.append((row, row + n, SegKind.REF_IMG))
                    pos.append(g)
                    for i in range(n):
                        img_pos.append(row + i)
                        img_update.append(False)
                    row += n
                    cursor += 1.0
                elif blk.kind is RefKind.AUDIO:
                    rt = blk.ref_audio_t
                    if rt > 0:
                        segments.append((row, row + rt * 2, SegKind.REF_AUDIO))
                        pos.append(audio_grid(cursor, rt, target_audio_w[0], target_audio_w[1]))
                        for i in range(rt * 2):
                            audio_pos.append(row + i)
                            audio_update.append(False)
                        row += rt * 2
                        cursor += float(rt)
                else:  # VIDEO | VIDEO_AUDIO
                    rt = blk.ref_audio_t
                    vt = blk.latent_t
                    r_frame, r_w_grid = frame_grid(blk.latent_h, blk.latent_w)
                    r_frame_rows = len(r_frame) // 2
                    if rt > 0:
                        segments.append((row, row + rt * 2, SegKind.REF_AUDIO))
                        pos.append(audio_grid(cursor, rt, r_w_grid[0], r_w_grid[-1]))
                        for i in range(rt * 2):
                            audio_pos.append(row + i)
                            audio_update.append(False)
                        row += rt * 2
                    n = vt * r_frame_rows
                    segments.append((row, row + n, SegKind.REF_IMG))
                    pos.append(video_grid(vt, r_frame, r_frame_rows, cursor))
                    for i in range(n):
                        img_pos.append(row + i)
                        img_update.append(False)
                    row += n
                    cursor += max(float(rt), sum(video_t_spans(vt)))

        # ---- target audio, then target video (always last two) ----------
        segments.append((row, row + audio_t * 2, SegKind.AUDIO))
        pos.append(audio_grid(cursor, audio_t, target_audio_w[0], target_audio_w[1]))
        for i in range(audio_t * 2):
            audio_pos.append(row + i)
            audio_update.append(True)
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append((row, row + n_video, SegKind.VIDEO))
        pos.append(video_grid(latent_t, frame, frame_rows, cursor))
        for i in range(n_video):
            img_pos.append(row + i)
            img_update.append(True)
        row += n_video

        position_ids: List[float] = [x for g in pos for x in g]
        # Signature must distinguish T2VA / FL2VA / Ref2VA packs that share the
        # same text+latent+audio dims. Rope caches key on this tuple; omitting
        # cond structure made T2VA→FL2VA reuse a short rope table and crash (or
        # silently mis-rotate if lengths had matched by chance).
        refs_sig = tuple(
            (int(blk.kind.value), int(blk.latent_t), int(blk.latent_h), int(blk.latent_w), int(blk.ref_audio_t))
            for blk in refs
        )
        return cls(
            seq_len=row,
            position_ids=position_ids,
            img_update=img_update,
            audio_update=audio_update,
            segments=segments,
            signature=(
                text_len,
                latent_t,
                latent_h,
                latent_w,
                audio_t,
                len(keyframes),
                int(frame_count or 0),
                refs_sig,
            ),
        )


# ---------------------------------------------------------------------------
# grid helpers (f64, exactly like the reference)


def axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> List[float]:
    ratio = dim / sqrt_area
    n = dim // patch
    return [
        ((i * (ratio / n)) + (1.0 - ratio) / 2.0) * 32.0 for i in range(n)
    ]


def frame_grid(h: int, w: int) -> Tuple[List[float], List[float]]:
    """Area-normalized (h, w) coordinates of one latent frame's 2x2-patch
    rows. Returns (frame rows flattened [n_h*n_w, 2], w axis grid)."""
    area = (h * w) ** 0.5
    hh = axis_from_sqrt_area(h, 2, area)
    ww = axis_from_sqrt_area(w, 2, area)
    frame: List[float] = []
    for hv in hh:
        for wv in ww:
            frame.extend([hv, wv])
    return frame, ww


def video_t_spans(n: int) -> List[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def video_t_grid(n: int, origin: float) -> List[float]:
    """origin + exclusive cumsum of the spans."""
    spans = video_t_spans(n)
    out: List[float] = []
    acc = 0.0
    for k in range(n):
        out.append(origin + acc)
        if k + 1 < n:
            acc += spans[k]
    return out


def audio_grid(cursor: float, t: int, w_low: float, w_high: float) -> List[float]:
    """Channel-major stereo rows: t advances per latent frame, w pinned to the
    grid extremes per stereo channel, h stays 0."""
    g = [0.0] * (t * 2 * 3)
    for k in range(t):
        g[k * 3] = cursor + k
        g[(t + k) * 3] = cursor + k
        g[k * 3 + 2] = w_low
        g[(t + k) * 3 + 2] = w_high
    return g


def video_grid(vt: int, frame: List[float], frame_rows: int, cursor: float) -> List[float]:
    t_grid = video_t_grid(vt, cursor)
    g: List[float] = []
    for tv in t_grid:
        for r in range(frame_rows):
            g.extend([tv, frame[r * 2], frame[r * 2 + 1]])
    return g
