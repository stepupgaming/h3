#!/usr/bin/env python3
"""Compile open H3-Context-IR *format* prompts from a short brief (free, local).

Official Context-IR is closed API. Open Base tokenizes text as written, so quality
comes from emitting the official field structure. This is the community L1 path
(templates / prompt builders), not a neural reverse-engineer of the hosted IR model.

Examples:
  uv run python scripts/h3_prompt_ir.py "A captain watches the fleet jump away" -o prompt.txt
  uv run python scripts/h3_prompt_ir.py brief.txt --mode fl2va --duration 10
  uv run python scripts/h3_prompt_ir.py "hero walks" --mode ref2va --pictures 2 --videos 1
  uv run python scripts/h3_prompt_ir.py --check already_ir.txt   # validate structure only

Bar B A/B: same Base settings; arm A = raw brief as text tokens; arm B = this output
encoded the same way you encode any prompt (``h3_text_encode.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Package import (uv run from repo root; editable install).
from minimax_h3.prompt_ir import (
    CAMERA_TYPES,
    FRAME_GRID_24FPS,
    MODES,
    STYLES,
    compile_from_brief,
    is_ir_format,
    maybe_compile,
    snap_duration_seconds,
)


def _read_brief(arg: str) -> str:
    p = Path(arg)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return arg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Open IR-format prompt compiler (Context-IR stand-in, local/free)"
    )
    ap.add_argument(
        "brief",
        nargs="?",
        default="",
        help="Short plain brief, or path to a text file",
    )
    ap.add_argument(
        "--mode",
        choices=MODES,
        default="t2va",
        help="Prompt family (default t2va)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Target seconds (snapped to 17k+5 @ 24fps grid)",
    )
    ap.add_argument(
        "--style",
        default=STYLES[0],
        help=f"Opening style phrase (default: {STYLES[0]!r})",
    )
    ap.add_argument("--soundscape", default=None, help="Override overall_soundscape")
    ap.add_argument(
        "--music",
        default=None,
        help="Override non_diegetic_music (use N/A for none)",
    )
    ap.add_argument(
        "--camera",
        default=None,
        help=f"Optional camera move (e.g. one of: {', '.join(CAMERA_TYPES[:6])}, …)",
    )
    ap.add_argument(
        "--pictures",
        type=int,
        default=0,
        help="Ref2VA: number of picture refs (default 0 → 1 if mode=ref2va)",
    )
    ap.add_argument("--videos", type=int, default=0, help="Ref2VA: number of video refs")
    ap.add_argument("--audios", type=int, default=0, help="Ref2VA: number of audio refs")
    ap.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Ref2VA summary task type (repeatable). Default: reference generation",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write prompt to file (default: stdout)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Only check whether brief/file already looks IR-formatted; exit 0/1",
    )
    ap.add_argument(
        "--passthrough-ir",
        action="store_true",
        help="If input already has IR fields, print it unchanged instead of re-wrapping",
    )
    ap.add_argument(
        "--show-grid",
        action="store_true",
        help="Print duration grid and exit",
    )
    args = ap.parse_args(argv)

    if args.show_grid:
        print("frames\tseconds (24 fps, 17k+5)")
        for fr, sec in FRAME_GRID_24FPS:
            print(f"{fr}\t{sec:.2f}")
        return 0

    if not args.brief and not args.check:
        ap.error("brief text or file path required (or --show-grid / --check with path)")

    text = _read_brief(args.brief) if args.brief else ""

    if args.check:
        ok = is_ir_format(text)
        print("ir_format" if ok else "not_ir_format")
        return 0 if ok else 1

    frames, dur = snap_duration_seconds(args.duration)
    n_pic = args.pictures
    if args.mode == "ref2va" and n_pic <= 0:
        n_pic = 1

    if args.passthrough_ir:
        prompt = maybe_compile(
            text,
            mode=args.mode,
            duration_s=args.duration,
            style=args.style,
            soundscape=args.soundscape,
            music=args.music,
            camera=args.camera,
            n_pictures=n_pic,
            n_videos=args.videos,
            n_audios=args.audios,
            task_types=args.tasks,
        )
    else:
        if is_ir_format(text) and not args.passthrough_ir:
            # User fed full IR by mistake into compiler without flag — keep it
            prompt = text if text.endswith("\n") else text + "\n"
            print(
                "# note: input already IR-shaped; passed through (use a short brief to compile)",
                file=sys.stderr,
            )
        else:
            prompt = compile_from_brief(
                text,
                mode=args.mode,
                duration_s=args.duration,
                style=args.style,
                soundscape=args.soundscape,
                music=args.music,
                camera=args.camera,
                n_pictures=n_pic,
                n_videos=args.videos,
                n_audios=args.audios,
                task_types=args.tasks,
            )

    meta = f"# mode={args.mode} duration_snap={dur:.2f}s frames={frames} (not embedded in prompt)\n"
    print(meta, file=sys.stderr)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt, encoding="utf-8")
        print(f"wrote {args.out} ({len(prompt)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(prompt if prompt.endswith("\n") else prompt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
