"""H3 canvas / resolution presets.

Official Base rules (HF / Comfy PR #15224):
  - pixel axes multiple of 32 (VAE 16× × DiT patch 2)
  - native short edge 768, area ≤ 768×1344 (~1.032 MP)
  - aspects including 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 (+ more)
  - 2K is H3-Regenerate-2K (closed) — not a Base canvas

Two adapt modes:
  - ``preview`` (default here): honor short edges < 768 for low-VRAM ladders
  - ``official``: always short edge 768 (Comfy adapt_canvas parity)

Megapixel ladders are **generators**, not a 16:9-only lock. Rows above the
Base area cap are clamped (never invent native 1080p/2K on open Base).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344  # ≈ 1.032192 MP
MAX_MP = MAX_PIXELS / 1_000_000.0

# Official + common community aspects (w/h). Portrait is inverse of landscape.
ASPECT_RATIOS: Dict[str, float] = {
    "32:9": 32 / 9,
    "21:9": 21 / 9,
    "2.35:1": 2.35,
    "2.39:1": 2.39,
    "16:9": 16 / 9,
    "3:2": 3 / 2,
    "4:3": 4 / 3,
    "5:4": 5 / 4,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "9:16": 9 / 16,
    "9:21": 9 / 21,
    "9:32": 9 / 32,
}

# Comfy-style MP steps for preview ladders (user table + a few extras).
# Values above MAX_MP are kept for display but **clamped** when resolving.
MP_LADDER: Tuple[float, ...] = (
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.98,
    1.00,
    1.20,
    1.50,
    1.80,
    2.00,
)

# Owner convenience size (not exact 0.4 MP 16:9 — that is 832×480).
OWNER_PREVIEW_16x9 = (864, 480)


@dataclass(frozen=True)
class CanvasSpec:
    width: int
    height: int
    aspect: str
    megapixels: float
    mode: str  # preview | official | raw
    clamped: bool = False
    note: str = ""

    @property
    def latent_hw(self) -> Tuple[int, int]:
        return self.height // 16, self.width // 16

    def as_dict(self) -> dict:
        lh, lw = self.latent_hw
        return {
            "width": self.width,
            "height": self.height,
            "aspect": self.aspect,
            "megapixels": round(self.megapixels, 4),
            "mode": self.mode,
            "clamped": self.clamped,
            "latent_h": lh,
            "latent_w": lw,
            "note": self.note,
        }


def round_multiple(x: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def parse_aspect(aspect: str) -> float:
    """Parse ``16:9``, ``16/9``, or a float string into w/h > 0."""
    s = aspect.strip().lower()
    if s in ASPECT_RATIOS:
        return ASPECT_RATIOS[s]
    if s in ("auto",):
        raise ValueError("aspect=auto requires an image; resolve via aspect_from_size first")
    if ":" in s:
        a, b = s.split(":", 1)
        return float(a) / float(b)
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def nearest_aspect_name(ratio: float, tol: float = 0.04) -> str:
    best = min(ASPECT_RATIOS.items(), key=lambda kv: abs(math.log(kv[1] / ratio)))
    if abs(math.log(best[1] / ratio)) <= tol:
        return best[0]
    return f"{ratio:.4f}"


def aspect_from_size(width: int, height: int) -> str:
    if height <= 0 or width <= 0:
        raise ValueError("width/height must be positive")
    return nearest_aspect_name(width / height)


def adapt_canvas(
    width: int,
    height: int,
    *,
    mode: str = "preview",
) -> Tuple[int, int]:
    """Map requested W×H to a legal H3 canvas.

    ``preview``: short = min(requested_short, 768) — allows 480p ladders.
    ``official``: always short = 768 (Comfy/PR parity).
    ``raw``: only enforce ×32 and area cap (no short-edge rewrite).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size {width}x{height}")
    mode = mode.lower().strip()
    if mode not in ("preview", "official", "raw"):
        raise ValueError(f"unknown canvas mode {mode!r}")

    max_pixels = float(MAX_PIXELS)
    ratio = width / height

    if mode == "raw":
        nom_w, nom_h = float(width), float(height)
    elif mode == "official":
        short = float(BASE_SHORT_EDGE)
        nom_w, nom_h = (short * ratio, short) if ratio >= 1.0 else (short, short / ratio)
    else:  # preview
        short = min(min(width, height), float(BASE_SHORT_EDGE))
        nom_w, nom_h = (short * ratio, short) if ratio >= 1.0 else (short, short / ratio)

    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w *= s
        nom_h *= s

    cw, ch = round_multiple(nom_w), round_multiple(nom_h)
    # ×32 rounding can re-break the area cap (ultra-wide/tall aspects); shrink
    # both axes one step at a time until legal while keeping aspect roughly.
    while cw * ch > MAX_PIXELS and (cw > CANVAS_MULTIPLE or ch > CANVAS_MULTIPLE):
        if cw >= ch and cw > CANVAS_MULTIPLE:
            cw -= CANVAS_MULTIPLE
        elif ch > CANVAS_MULTIPLE:
            ch -= CANVAS_MULTIPLE
        else:
            break
    return max(CANVAS_MULTIPLE, cw), max(CANVAS_MULTIPLE, ch)


def canvas_from_aspect_short(
    aspect: str,
    short_edge: int = BASE_SHORT_EDGE,
    *,
    mode: str = "official",
) -> CanvasSpec:
    """Build canvas from aspect + short edge (official API style)."""
    r = parse_aspect(aspect)
    if r >= 1.0:
        w, h = short_edge * r, float(short_edge)
    else:
        w, h = float(short_edge), short_edge / r
    # Use official/preview adapt from the nominal size
    cw, ch = adapt_canvas(int(round(w)), int(round(h)), mode=mode if mode != "raw" else "preview")
    mp = (cw * ch) / 1_000_000.0
    clamped = mp > MAX_MP + 1e-9 or (cw * ch) >= MAX_PIXELS
    name = aspect if aspect in ASPECT_RATIOS else nearest_aspect_name(r)
    return CanvasSpec(cw, ch, name, mp, mode, clamped=clamped)


def canvas_from_megapixels(
    aspect: str,
    megapixels: float,
    *,
    mode: str = "preview",
    clamp_base: bool = True,
) -> CanvasSpec:
    """Comfy-style MP × aspect → ×32 canvas.

    If ``clamp_base`` and requested MP exceeds Base max (~1.032), clamp and mark.
    """
    r = parse_aspect(aspect)
    req_mp = float(megapixels)
    clamped = False
    note = ""
    if clamp_base and req_mp > MAX_MP:
        clamped = True
        note = f"requested {req_mp:.2f} MP > Base max {MAX_MP:.3f} MP; clamped (2K needs regenerate/SR)"
        req_mp = MAX_MP

    pixels = req_mp * 1_000_000.0
    # h = sqrt(pixels / r), w = r * h
    h = math.sqrt(pixels / r)
    w = r * h
    cw, ch = round_multiple(w), round_multiple(h)
    # Re-clamp area after rounding (may need stepwise shrink — same as adapt_canvas)
    if cw * ch > MAX_PIXELS and clamp_base:
        s = math.sqrt(MAX_PIXELS / (cw * ch))
        cw, ch = round_multiple(cw * s), round_multiple(ch * s)
        while cw * ch > MAX_PIXELS and (cw > CANVAS_MULTIPLE or ch > CANVAS_MULTIPLE):
            if cw >= ch and cw > CANVAS_MULTIPLE:
                cw -= CANVAS_MULTIPLE
            elif ch > CANVAS_MULTIPLE:
                ch -= CANVAS_MULTIPLE
            else:
                break
        clamped = True
        if not note:
            note = "clamped to Base area cap after ×32 round"

    # Optional preview/official short-edge pass (usually no-op if already under cap)
    if mode == "official":
        cw, ch = adapt_canvas(cw, ch, mode="official")
    elif mode == "preview":
        # Don't force short down if user asked for a specific MP under cap
        pass

    mp = (cw * ch) / 1_000_000.0
    name = aspect if aspect in ASPECT_RATIOS else nearest_aspect_name(r)
    return CanvasSpec(cw, ch, name, mp, mode, clamped=clamped, note=note)


def canvas_from_wh(
    width: int,
    height: int,
    *,
    mode: str = "preview",
) -> CanvasSpec:
    cw, ch = adapt_canvas(width, height, mode=mode)
    mp = (cw * ch) / 1_000_000.0
    return CanvasSpec(
        cw,
        ch,
        aspect_from_size(cw, ch),
        mp,
        mode,
        clamped=(width, height) != (cw, ch) or mp > MAX_MP,
        note="" if (width, height) == (cw, ch) else f"adapted from {width}x{height}",
    )


def resolve_canvas(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect: Optional[str] = None,
    megapixels: Optional[float] = None,
    short_edge: Optional[int] = None,
    mode: str = "preview",
    clamp_base: bool = True,
) -> CanvasSpec:
    """Resolve CLI-style canvas knobs to one CanvasSpec.

    Priority:
      1. aspect + megapixels
      2. aspect (+ optional short_edge, default 768)
      3. width + height (both required together)
      4. defaults official 16:9 1344×768
    """
    mode = (mode or "preview").lower()
    if (width is None) ^ (height is None):
        raise ValueError("pass both --width and --height, or neither (use --aspect / defaults)")

    if aspect and aspect.strip().lower() == "auto":
        if width and height:
            aspect = aspect_from_size(width, height)
        else:
            raise ValueError("--aspect auto needs --width/--height or a source image size")

    if aspect is not None and megapixels is not None:
        return canvas_from_megapixels(aspect, megapixels, mode=mode, clamp_base=clamp_base)

    if aspect is not None:
        se = int(short_edge) if short_edge is not None else BASE_SHORT_EDGE
        return canvas_from_aspect_short(aspect, se, mode=mode if mode != "raw" else "official")

    if width is not None and height is not None:
        return canvas_from_wh(width, height, mode=mode)

    # Default: official native 16:9
    return canvas_from_aspect_short("16:9", BASE_SHORT_EDGE, mode="official")


def list_official_short_edge_table(
    aspects: Optional[Sequence[str]] = None,
    short_edge: int = BASE_SHORT_EDGE,
) -> List[CanvasSpec]:
    keys = list(aspects) if aspects is not None else [
        "21:9",
        "16:9",
        "4:3",
        "1:1",
        "3:4",
        "9:16",
    ]
    return [canvas_from_aspect_short(a, short_edge, mode="official") for a in keys]


def list_mp_ladder(
    aspect: str = "16:9",
    mps: Optional[Sequence[float]] = None,
    *,
    clamp_base: bool = True,
) -> List[CanvasSpec]:
    vals = list(mps) if mps is not None else list(MP_LADDER)
    return [canvas_from_megapixels(aspect, m, mode="preview", clamp_base=clamp_base) for m in vals]


def format_resolution_table(
    rows: Iterable[CanvasSpec],
    *,
    title: str = "",
) -> str:
    lines: List[str] = []
    if title:
        lines.append(title)
    lines.append(f"{'aspect':>8}  {'MP':>6}  {'output':>13}  {'latent':>9}  notes")
    lines.append("-" * 64)
    for r in rows:
        note = r.note
        if r.clamped and "clamp" not in note.lower():
            note = (note + "; clamped").strip("; ")
        lines.append(
            f"{r.aspect:>8}  {r.megapixels:6.3f}  {r.width:5d} x {r.height:<4d}  "
            f"{r.latent_hw[0]:3d}x{r.latent_hw[1]:<4d}  {note}"
        )
    lines.append(f"Base area cap: {MAX_PIXELS} px ({MAX_MP:.3f} MP); multiple={CANVAS_MULTIPLE}")
    return "\n".join(lines)


def add_canvas_args(parser, *, defaults_native: bool = True) -> None:
    """Attach shared canvas flags to an argparse parser / subparser."""
    parser.add_argument(
        "--width",
        type=int,
        default=1344 if defaults_native else None,
        help="canvas width before adapt (default 1344 with height 768 when no preset)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768 if defaults_native else None,
        help="canvas height before adapt (default 768)",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help=(
            "aspect preset e.g. 16:9, 9:16, 1:1, 21:9, 4:3, 3:4 (or auto with W/H). "
            "Combines with --megapixels or --short-edge"
        ),
    )
    parser.add_argument(
        "--megapixels",
        "--mp",
        type=float,
        default=None,
        dest="megapixels",
        help=f"target megapixels with --aspect (clamped to Base max {MAX_MP:.3f} MP)",
    )
    parser.add_argument(
        "--short-edge",
        type=int,
        default=None,
        help=f"short edge with --aspect (official default {BASE_SHORT_EDGE})",
    )
    parser.add_argument(
        "--canvas-mode",
        choices=["preview", "official", "raw"],
        default="preview",
        help=(
            "preview=allow short edge <768 (low-VRAM ladders); "
            "official=force short edge 768 (Comfy parity); "
            "raw=only ×32 + area cap"
        ),
    )


def resolve_from_args(args) -> CanvasSpec:
    """Resolve canvas from an argparse namespace that used add_canvas_args / sample flags."""
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    aspect = getattr(args, "aspect", None)
    mp = getattr(args, "megapixels", None)
    short = getattr(args, "short_edge", None)
    mode = getattr(args, "canvas_mode", "preview") or "preview"

    # If user only left defaults width/height and also passed aspect/mp, prefer preset
    return resolve_canvas(
        width=width,
        height=height,
        aspect=aspect,
        megapixels=mp,
        short_edge=short,
        mode=mode,
    )
