#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Victor Mustar
# Internalized from https://github.com/gary149/h3-game-sprites
"""Pack keyed sprite frames into one atlas at a SINGLE global scale.

 python3 build_atlas.py sprites/ atlas.json --moves idle,walk_fwd,punch \
 --ref idle --height 340
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os

import numpy as np
from PIL import Image


def stats(im):
    """bbox, feet baseline, and a stance anchor x from the bottom ~22% of the mask."""
    a = np.asarray(im)[..., 3] > 0
    ys, xs = np.nonzero(a)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    thresh = ys.max() - int((ys.max() - ys.min()) * 0.22)
    leg = ys >= thresh
    return bbox, int(ys.max()), float(xs[leg].mean() if leg.any() else xs.mean())


def picks_for(d):
    rep = json.load(open(os.path.join(d, "report.json")))["summary"]
    return rep["picked"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sprites_dir")
    ap.add_argument("out_json")
    ap.add_argument("--moves", required=True, help="comma-separated move names")
    ap.add_argument("--ref", default="idle", help="move whose first frame sets the scale")
    ap.add_argument("--height", type=int, default=340, help="on-screen px for the reference pose")
    args = ap.parse_args()

    moves = [m.strip() for m in args.moves.split(",") if m.strip()]

    def frame_path(m, i):
        return os.path.join(args.sprites_dir, m, "keyed", f"f{i + 1:04d}.png")

    ref_i = picks_for(os.path.join(args.sprites_dir, args.ref))[0]
    ref_bbox, ref_base, ref_anchor = stats(Image.open(frame_path(args.ref, ref_i)))
    scale = args.height / (ref_bbox[3] - ref_bbox[1])
    print(f"reference {args.ref}[{ref_i}]: scale {scale:.4f}, ground {ref_base}, anchor_x {ref_anchor:.1f}")

    atlas = {}
    for m in moves:
        frames = []
        for i in picks_for(os.path.join(args.sprites_dir, m)):
            im = Image.open(frame_path(m, i))
            bbox, base, anchor = stats(im)
            crop = im.crop(bbox)
            w, h = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
            buf = io.BytesIO()
            crop.resize((w, h), Image.LANCZOS).save(buf, "PNG", optimize=True)
            frames.append(
                {
                    "b64": base64.b64encode(buf.getvalue()).decode(),
                    "w": w,
                    "h": h,
                    "footOffset": round((base - ref_base) * scale),
                    "anchorOffset": round((anchor - bbox[0]) * scale),
                    "driftX": round((anchor - ref_anchor) * scale),
                }
            )
        atlas[m] = frames
        hs = [f["h"] for f in frames]
        print(f"{m:12s} {len(frames):3d} frames on-screen height {min(hs)}-{max(hs)}px")

    json.dump(atlas, open(args.out_json, "w"))
    print(f"\nwrote {args.out_json} ({os.path.getsize(args.out_json) // 1024} KB)")
    print(
        f"sanity: crouch/tuck poses must be SHORTER than {args.ref} "
        f"({args.height}px). Taller means per-frame scaling crept back in."
    )


if __name__ == "__main__":
    main()
