"""Gemmy-owned H3 masked video edit helpers.

Veteran AI / ganloss `minimax_h3_r2v_video_mask_edit` (video PJWfUAO1Oco):
SAM3 tracks a subject, Gemmy crops it (the JSON 1 MP video resize), Eros
two-stage Ref2VA uses that crop as `ref_video`, then uncrop pastes back.
Crop / cleanup / uncrop live here so the product path does not vendor GPL
MaskVidExperiments.

Arrays are numpy `float32`:
  frames  [N, H, W, 3] in 0..1
  masks   [N, H, W] in 0..1  (1 = edit)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

# H3 video VAE causal groups. Same tuple as gemmy-h3-context/av_math.py.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def video_latent_t(frame_count: int) -> int:
    n = int(frame_count)
    return 2 if n <= 5 else ((n - 5) // 17) * 5 + 2


def snap_div(value: int, div: int) -> int:
    div = max(1, int(div))
    return max(div, int(round(int(value) / div) * div))


def cleanup_masks(
    masks: np.ndarray,
    *,
    min_area: int = 32,
    min_frame_presence: int = 2,
    grow: int = 0,
) -> np.ndarray:
    """Drop flash fragments and optionally dilate the keep-region.

    A connected component that appears on fewer than ``min_frame_presence``
    frames, or covers fewer than ``min_area`` pixels on its peak frame, is
    treated as segmentation noise.
    """
    m = np.asarray(masks, dtype=np.float32)
    if m.ndim != 3:
        raise ValueError(f"masks must be [N,H,W], got {m.shape}")
    binary = m > 0.1
    n, h, w = binary.shape
    kept = np.zeros_like(binary)
    # Cheap temporal gate: a pixel must be on in at least min_frame_presence
    # frames in a ±2 window, or we drop it as a flash.
    if n == 1:
        kept[0] = binary[0]
    else:
        counts = binary.astype(np.int16)
        window = counts.copy()
        for dt in (1, 2):
            window[dt:] += counts[:-dt]
            window[:-dt] += counts[dt:]
        kept = binary & (window >= min_frame_presence)
    areas = kept.reshape(n, -1).sum(axis=1)
    if min_area > 0:
        for i in range(n):
            if int(areas[i]) < min_area:
                kept[i] = False
    out = kept.astype(np.float32)
    if grow > 0:
        out = _dilate(out, grow)
    return out


def mask_coverage(masks: np.ndarray) -> dict[str, float | int]:
    """Per-clip binary coverage (fraction of pixels on)."""
    m = np.asarray(masks, dtype=np.float32) > 0.5
    if m.ndim != 3 or m.size == 0:
        raise ValueError(f"masks must be [N,H,W], got {getattr(m, 'shape', None)}")
    per = m.reshape(m.shape[0], -1).mean(axis=1)
    return {
        "frames": int(m.shape[0]),
        "mean": float(per.mean()),
        "min": float(per.min()),
        "max": float(per.max()),
    }


def compose_overlay(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    tint: tuple[float, float, float] = (1.0, 0.12, 0.12),
    alpha: float = 0.5,
) -> np.ndarray:
    """Tint the keep-region so a human or `gemmy analyze` can see the selection."""
    f = np.asarray(frames, dtype=np.float32)
    m = np.asarray(masks, dtype=np.float32)
    if f.ndim != 4 or m.ndim != 3:
        raise ValueError(f"frames [N,H,W,3] and masks [N,H,W], got {f.shape} / {m.shape}")
    if f.shape[:3] != m.shape:
        raise ValueError(f"frame/mask spatial mismatch {f.shape} vs {m.shape}")
    w = np.clip(m[..., None], 0.0, 1.0) * float(alpha)
    t = np.array(tint, dtype=np.float32).reshape(1, 1, 1, 3)
    return np.clip(f * (1.0 - w) + t * w, 0.0, 1.0)


def _dilate(masks: np.ndarray, radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return masks
    n, h, w = masks.shape
    out = masks.copy()
    on = masks > 0.5
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            shifted = np.zeros_like(on)
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            shifted[:, y0:y1, x0:x1] = on[:, sy0:sy1, sx0:sx1]
            out = np.maximum(out, shifted.astype(np.float32))
    return out


def _frame_bbox(mask: np.ndarray, thresh: float = 0.1) -> tuple[int, int, int, int] | None:
    on = np.where(mask > thresh)
    if on[0].size == 0:
        return None
    y0, y1 = int(on[0].min()), int(on[0].max()) + 1
    x0, x1 = int(on[1].min()), int(on[1].max()) + 1
    return x0, y0, x1, y1


def _pad_box(
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    scale: float,
    div: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cw = max(1, x1 - x0)
    ch = max(1, y1 - y0)
    cx = x0 + cw / 2.0
    cy = y0 + ch / 2.0
    nw = snap_div(int(math.ceil(cw * float(scale))), div)
    nh = snap_div(int(math.ceil(ch * float(scale))), div)
    nw = min(nw, snap_div(img_w, div) if img_w >= div else img_w)
    nh = min(nh, snap_div(img_h, div) if img_h >= div else img_h)
    nx0 = int(round(cx - nw / 2.0))
    ny0 = int(round(cy - nh / 2.0))
    nx0 = max(0, min(nx0, img_w - nw))
    ny0 = max(0, min(ny0, img_h - nh))
    return nx0, ny0, nx0 + nw, ny0 + nh


def plan_combined_crop(
    masks: np.ndarray,
    *,
    crop_scale: float = 1.5,
    divisible_by: int = 32,
    upscale_megapixels: float = 0.0,
) -> dict[str, Any]:
    """One static crop covering the subject's whole travel."""
    m = np.asarray(masks, dtype=np.float32)
    n, h, w = m.shape
    union: tuple[int, int, int, int] | None = None
    for i in range(n):
        b = _frame_bbox(m[i])
        if b is None:
            continue
        if union is None:
            union = b
        else:
            union = (
                min(union[0], b[0]),
                min(union[1], b[1]),
                max(union[2], b[2]),
                max(union[3], b[3]),
            )
    if union is None:
        raise ValueError("SAM3 mask is empty — no subject to crop (try a different --mask-prompt)")
    box = _pad_box(union, w, h, crop_scale, divisible_by)
    out_w, out_h = box[2] - box[0], box[3] - box[1]
    if upscale_megapixels and upscale_megapixels > 0:
        out_w, out_h = _size_for_mp(out_w, out_h, upscale_megapixels, divisible_by)
    boxes = [list(box) for _ in range(n)]
    return {
        "mode": "combined",
        "boxes": boxes,
        "out_width": int(out_w),
        "out_height": int(out_h),
        "source_width": int(w),
        "source_height": int(h),
    }


def plan_tracked_crop(
    masks: np.ndarray,
    *,
    crop_scale: float = 1.5,
    divisible_by: int = 32,
    upscale_megapixels: float = 0.0,
) -> dict[str, Any]:
    """Constant-size crop that stays still until the subject would leave it."""
    m = np.asarray(masks, dtype=np.float32)
    n, h, w = m.shape
    raw: list[tuple[int, int, int, int] | None] = [_frame_bbox(m[i]) for i in range(n)]
    present = [b for b in raw if b is not None]
    if not present:
        raise ValueError("SAM3 mask is empty — no subject to crop (try a different --mask-prompt)")
    max_w = max(b[2] - b[0] for b in present)
    max_h = max(b[3] - b[1] for b in present)
    dummy = (0, 0, max_w, max_h)
    sized = _pad_box(dummy, w, h, crop_scale, divisible_by)
    cw, ch = sized[2] - sized[0], sized[3] - sized[1]
    boxes: list[list[int]] = []
    last: tuple[int, int, int, int] | None = None
    for b in raw:
        if b is None:
            if last is None:
                last = (0, 0, cw, ch)
            boxes.append(list(last))
            continue
        sx0, sy0, sx1, sy1 = b
        if last is not None and sx0 >= last[0] and sy0 >= last[1] and sx1 <= last[2] and sy1 <= last[3]:
            boxes.append(list(last))
            continue
        cx = (sx0 + sx1) / 2.0
        cy = (sy0 + sy1) / 2.0
        nx0 = int(round(cx - cw / 2.0))
        ny0 = int(round(cy - ch / 2.0))
        nx0 = max(0, min(nx0, w - cw))
        ny0 = max(0, min(ny0, h - ch))
        last = (nx0, ny0, nx0 + cw, ny0 + ch)
        boxes.append(list(last))
    out_w, out_h = cw, ch
    if upscale_megapixels and upscale_megapixels > 0:
        out_w, out_h = _size_for_mp(out_w, out_h, upscale_megapixels, divisible_by)
    return {
        "mode": "tracked",
        "boxes": boxes,
        "out_width": int(out_w),
        "out_height": int(out_h),
        "source_width": int(w),
        "source_height": int(h),
    }


def _size_for_mp(width: int, height: int, megapixels: float, div: int) -> tuple[int, int]:
    """Enlarge toward ``megapixels`` (crop planner). Never shrinks."""
    area = max(1, int(width) * int(height))
    target = max(area, int(float(megapixels) * 1_000_000))
    scale = math.sqrt(target / area)
    nw = snap_div(max(div, int(round(width * scale))), div)
    nh = snap_div(max(div, int(round(height * scale))), div)
    return nw, nh


def size_for_megapixels(
    width: int, height: int, megapixels: float, div: int = 32
) -> tuple[int, int]:
    """ResolutionSelector-style size at ``megapixels``, ×``div``. Can shrink."""
    area = max(1, int(width) * int(height))
    target = max(int(div) * int(div), int(round(float(megapixels) * 1_000_000)))
    scale = math.sqrt(target / area)
    nw = snap_div(max(div, int(round(width * scale))), div)
    nh = snap_div(max(div, int(round(height * scale))), div)
    return nw, nh


def apply_crop(
    frames: np.ndarray,
    masks: np.ndarray,
    plan: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice / resize every frame to the planned crop. Returns cropped frames+masks."""
    frames = np.asarray(frames, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    boxes = plan["boxes"]
    out_w = int(plan["out_width"])
    out_h = int(plan["out_height"])
    n = min(frames.shape[0], masks.shape[0], len(boxes))
    out_f = np.zeros((n, out_h, out_w, 3), dtype=np.float32)
    out_m = np.zeros((n, out_h, out_w), dtype=np.float32)
    for i in range(n):
        x0, y0, x1, y1 = (int(v) for v in boxes[i])
        patch = frames[i, y0:y1, x0:x1]
        mp = masks[i, y0:y1, x0:x1]
        if patch.shape[0] != out_h or patch.shape[1] != out_w:
            out_f[i] = _resize_hwc(patch, out_h, out_w)
            out_m[i] = _resize_hw(mp, out_h, out_w)
        else:
            out_f[i] = patch
            out_m[i] = mp
    return out_f, out_m


def uncrop(
    processed: np.ndarray,
    originals: np.ndarray,
    plan: dict[str, Any],
    cropped_masks: np.ndarray | None = None,
    *,
    feather: int = 8,
) -> np.ndarray:
    """Paste processed crops back into the original frames with a feathered edge.

    The SAM3 matte is the edit region, not a tight key. Matching background
    in the crop is pasted too so leftover source hair volume cannot print as
    a hedge cutout. Unmasked pixels the crop actually changed stay original.
    Feather keeps the subject interior opaque.
    """
    processed = np.asarray(processed, dtype=np.float32)
    originals = np.asarray(originals, dtype=np.float32)
    boxes = plan["boxes"]
    n = min(processed.shape[0], originals.shape[0], len(boxes))
    out = originals[:n].copy()
    img_h, img_w = out.shape[1], out.shape[2]
    for i in range(n):
        x0, y0, x1, y1 = (int(v) for v in boxes[i])
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        crop = processed[i]
        if crop.shape[0] != bh or crop.shape[1] != bw:
            crop = _resize_hwc(crop, bh, bw)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(img_w, x0 + bw), min(img_h, y0 + bh)
        sx0, sy0 = x0c - x0, y0c - y0
        region = crop[sy0 : sy0 + (y1c - y0c), sx0 : sx0 + (x1c - x0c)]
        dest = out[i, y0c:y1c, x0c:x1c]
        if cropped_masks is not None and i < cropped_masks.shape[0]:
            alpha = cropped_masks[i]
            if alpha.shape[0] != bh or alpha.shape[1] != bw:
                alpha = _resize_hw(alpha, bh, bw)
            alpha = alpha[sy0 : sy0 + (y1c - y0c), sx0 : sx0 + (x1c - x0c)]
            alpha = _paste_alpha(alpha, dest, region, feather)
        else:
            alpha = np.ones((y1c - y0c, x1c - x0c), dtype=np.float32)
            if feather > 0:
                alpha = _feather_alpha(alpha, feather)
        a = alpha[..., None]
        out[i, y0c:y1c, x0c:x1c] = region * a + dest * (1.0 - a)
    return np.clip(out, 0.0, 1.0)


# Outside the tracked region, keep source pixels that the crop actually
# changed (another person). Matching background is taken from the crop so
# the old hair silhouette does not print as a hedge cutout.
_PROTECT_THRESH = 0.15


def _paste_alpha(
    subject: np.ndarray,
    original: np.ndarray,
    processed: np.ndarray,
    feather: int,
) -> np.ndarray:
    """Processed everywhere in the crop except protected unmasked pixels.

    The SAM3 matte answers *where we must edit*, not *the only pixels we
    paste*. Tight-keying it leaves a hair-shaped seam: leftover source
    volume is generated background sitting on original background.
    """
    subject_on = np.asarray(subject, dtype=np.float32) > 0.5
    diff = np.mean(
        np.abs(
            np.asarray(original, dtype=np.float32) - np.asarray(processed, dtype=np.float32)
        ),
        axis=-1,
    )
    high = diff > _PROTECT_THRESH
    # Missed hair/coat is high-diff and touches the subject. Absorb it.
    # Another person is high-diff across matching background and stays original.
    subject_ex = _geodesic_dilate(subject_on, subject_on | high)
    protect = high & ~subject_ex
    keep = protect.astype(np.float32)
    if int(feather) > 0 and keep.any():
        # Expand *original* around the protected person, not processed into them.
        keep = _feather_alpha(keep, int(feather))
        keep = np.maximum(keep, protect.astype(np.float32))
    take = 1.0 - keep
    take = np.maximum(take, subject_ex.astype(np.float32))
    if int(feather) > 0 and not protect.any():
        take = _feather_alpha(take, int(feather))
        take = np.maximum(take, subject_ex.astype(np.float32))
    return take


def _geodesic_dilate(
    seed: np.ndarray, allowed: np.ndarray, *, max_iter: int = 64
) -> np.ndarray:
    cur = np.asarray(seed, dtype=bool)
    allowed_b = np.asarray(allowed, dtype=bool)
    for _ in range(int(max_iter)):
        nxt = (_dilate(cur.astype(np.float32)[None, ...], 1)[0] > 0.5) & allowed_b
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def _feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Soft edge in a band *outside* the keep-region. Interior stays opaque.

    ``alpha * box_blur(alpha)`` erodes the matte: source hair in that inward
    band shows through as a dark outline. Dilate by ``radius`` first, blur the
    grown mask, then clamp the original interior back to 1 so concave hair
    edges cannot dip below opaque.
    """
    r = max(1, int(radius))
    on = (alpha > 0.5).astype(np.float32)
    if not on.any():
        return np.clip(alpha, 0.0, 1.0)
    grown = _dilate(on[None, ...], r)[0]
    h, w = grown.shape
    padded = np.pad(grown, r, mode="edge")
    acc = np.zeros((h, w), dtype=np.float32)
    k = (2 * r + 1) ** 2
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            acc += padded[dy : dy + h, dx : dx + w]
    blur = acc / k
    return np.clip(np.maximum(on, blur), 0.0, 1.0)


def _resize_hwc(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((tw, th), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _resize_hw(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((tw, th), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def pixel_mask_to_h3_latent_mask(masks: np.ndarray, *, spatial_div: int = 16) -> np.ndarray:
    """Unify a pixel mask onto the H3 video VAE token grid.

    Causal frame groups are ``(1,4,4,4,4)``. Spatial tokens are 16×16 source
    pixels (2×2 of the 8×8 VAE cells). Output shape ``[1, 1, T, H/16, W/16]``.
    """
    m = np.asarray(masks, dtype=np.float32)
    if m.ndim != 3:
        raise ValueError(f"masks must be [N,H,W], got {m.shape}")
    n, h, w = m.shape
    t = video_latent_t(n)
    lh = max(1, h // spatial_div)
    lw = max(1, w // spatial_div)
    out = np.zeros((1, 1, t, lh, lw), dtype=np.float32)
    frame_idx = 0
    for ti in range(t):
        nframes = FRAME_PER_TOKEN[ti % 5]
        take = min(nframes, n - frame_idx)
        if take <= 0:
            break
        chunk = m[frame_idx : frame_idx + take]
        frame_idx += take
        # Max-pool each 16×16 cell across the token's frames.
        cropped = chunk[:, : lh * spatial_div, : lw * spatial_div]
        pooled = cropped.reshape(take, lh, spatial_div, lw, spatial_div).max(axis=(0, 2, 4))
        out[0, 0, ti] = pooled
    return out
