#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Victor Mustar
# Internalized from https://github.com/gary149/h3-game-sprites
"""Decompose a generated video clip into validated, chroma-keyed sprite frames.

 python3 sprite_cut.py clip.mp4 out/ --frames 12
 python3 sprite_cut.py walk.mp4 out/ --frames 8 --loop  # seamless cycle

Writes out/keyed/*.png (RGBA), strip.png, strip_preview.jpg, anim.gif and
report.json (per-frame metrics + summary).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image


def extract(video, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video, os.path.join(tmpdir, "f%04d.png")],
        check=True,
    )
    return sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))


def key_frame(img, color):
    """RGBA with the backdrop removed, plus the background mask.

    Despill clamps the key channels against the off-channel rather than
    blending toward it -- blending visibly tints the character.
    """
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "magenta":
        score, off = r + b - 2 * g, g
        bg = (score > 120) & (r > 90) & (b > 90)
    else:
        score, off = 2 * g - r - b, np.maximum(r, b)
        bg = (score > 120) & (g > 90)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    cap = off * 1.18 + 30
    if color == "magenta":
        a[..., 0], a[..., 2] = np.minimum(r, cap), np.minimum(b, cap)
    else:
        a[..., 1] = np.minimum(g, cap)
    out = np.dstack([a.clip(0, 255).astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA"), bg


def metrics(rgba, bgmask):
    al = np.asarray(rgba)[..., 3] > 0
    ys, xs = np.nonzero(al)
    if len(xs) == 0:
        return None
    return {
        "baseline": int(ys.max()),
        "top": int(ys.min()),
        "cx": round(float(xs.mean()), 1),
        "height": int(ys.max() - ys.min()),
        "coverage": round(float(al.mean()), 4),
        "bg_purity": round(float(bgmask.mean() + al.mean()), 4),
    }


def pick_arclength(energy, n):
    """Equal steps of cumulative motion, so frames land on pose extremes."""
    cum = np.cumsum(energy)
    if cum[-1] <= 0:
        return sorted(set(np.linspace(0, len(energy) - 1, n).round().astype(int).tolist()))
    targets = np.linspace(0, cum[-1], n)
    return sorted({int(np.argmin(np.abs(cum - t))) for t in targets} | {0, len(energy) - 1})


def pick_loop(small, n, lo_frac=0.2, hi_frac=0.85, min_gap=12, max_gap=45):
    """Find the frame pair that matches most closely and sample inside it."""
    total = len(small)
    idxs = range(int(total * lo_frac), int(total * hi_frac))
    best = None
    for i in idxs:
        for j in idxs:
            if not (min_gap <= j - i <= max_gap):
                continue
            d = float(np.abs(small[i] - small[j]).mean())
            if best is None or d < best[0]:
                best = (d, i, j)
    if best is None:
        return None, sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))
    d, i, j = best
    return {"i": i, "j": j, "diff": round(d, 3)}, sorted({i + round(k * (j - i) / n) for k in range(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("outdir")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--loop", action="store_true", help="extract a seamless cycle (walks)")
    ap.add_argument("--key", default="magenta", choices=["magenta", "green"])
    args = ap.parse_args()

    od = args.outdir
    raw, keyed = os.path.join(od, "raw"), os.path.join(od, "keyed")
    os.makedirs(keyed, exist_ok=True)
    names = extract(args.video, raw)

    report, small = [], []
    for i, n in enumerate(names):
        rgba, bgmask = key_frame(Image.open(os.path.join(raw, n)), args.key)
        rgba.save(os.path.join(keyed, n))
        m = metrics(rgba, bgmask)
        if m is None:
            m = {
                "baseline": 0,
                "top": 0,
                "cx": 0.0,
                "height": 0,
                "coverage": 0.0,
                "bg_purity": 0.0,
            }
        m["frame"] = i
        report.append(m)
        s = rgba.resize((max(1, rgba.width // 8), max(1, rgba.height // 8)))
        small.append(np.asarray(s).astype(np.int16))

    energy = [0.0] + [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(small))]
    for m, e in zip(report, energy):
        m["motion"] = round(e, 2)

    loop_info = None
    if args.loop:
        loop_info, picks = pick_loop(small, args.frames)
    else:
        picks = pick_arclength(energy, args.frames)
    picks = picks[: args.frames]

    summary = {
        "frames": len(names),
        "picked": picks,
        "loop": loop_info,
        "baseline_drift_px": max(m["baseline"] for m in report) - min(m["baseline"] for m in report),
        "centroid_drift_px": round(max(m["cx"] for m in report) - min(m["cx"] for m in report), 1),
        "height_var_px": max(m["height"] for m in report) - min(m["height"] for m in report),
        "loop_diff": round(
            loop_info["diff"] if loop_info else float(np.abs(small[picks[0]] - small[picks[-1]]).mean()),
            2,
        ),
        "worst_bg_purity": min(m["bg_purity"] for m in report),
        "first_vs_last_baseline": abs(report[picks[0]]["baseline"] - report[picks[-1]]["baseline"]),
    }

    cells = [Image.open(os.path.join(keyed, names[i])) for i in picks]
    boxes = [c.getbbox() or (0, 0, c.width, c.height) for c in cells]
    cw = max(b[2] - b[0] for b in boxes) + 16
    ch = max(b[3] - b[1] for b in boxes) + 16
    strip = Image.new("RGBA", (cw * len(cells), ch), (0, 0, 0, 0))
    for j, (c, b) in enumerate(zip(cells, boxes)):
        crop = c.crop(b)
        strip.paste(crop, (j * cw + (cw - crop.width) // 2, ch - 8 - crop.height), crop)
    strip.save(os.path.join(od, "strip.png"))
    prev = Image.new("RGB", strip.size, (24, 26, 32))
    prev.paste(strip, (0, 0), strip)
    prev.thumbnail((1800, 1800))
    prev.save(os.path.join(od, "strip_preview.jpg"), quality=88)

    gif, durs = [], []
    for j, i in enumerate(picks):
        f = Image.open(os.path.join(keyed, names[i]))
        bg = Image.new("RGB", f.size, (24, 26, 32))
        bg.paste(f, (0, 0), f)
        bg.thumbnail((420, 560))
        gif.append(bg)
        nxt = picks[j + 1] if j + 1 < len(picks) else picks[0] + len(names)
        durs.append(max(1, (nxt - i)) * 1000 // 24)
    gif[0].save(os.path.join(od, "anim.gif"), save_all=True, append_images=gif[1:], duration=durs, loop=0)

    json.dump({"summary": summary, "frames": report}, open(os.path.join(od, "report.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
