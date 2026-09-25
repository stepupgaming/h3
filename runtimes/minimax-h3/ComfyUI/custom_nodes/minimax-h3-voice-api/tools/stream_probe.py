#!/usr/bin/env python3
"""Call the streaming route and record true first-audio-byte latency."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8787/v1/audio/speech")
    parser.add_argument("--voice", default="warm_narrator")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    payload = {
        "input": args.text, "voice": args.voice, "stream": True,
        "response_format": "pcm", "steps": args.steps,
        "verify": not args.no_verify, "strict_fidelity": True,
        "min_similarity": 0.90, "max_retries": 2,
        "target_words": 18, "max_words": 24,
        "channels": args.channels,
    }
    started = time.monotonic()
    first_byte = None
    received = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=None) as client, args.output.open("wb") as output:
        with client.stream("POST", args.url, json=payload) as response:
            response.raise_for_status()
            for block in response.iter_bytes():
                if block and first_byte is None:
                    first_byte = time.monotonic()
                output.write(block)
                received += len(block)
    finished = time.monotonic()
    result = {
        "status": "passed", "voice": args.voice,
        "first_audio_byte_seconds": round((first_byte or finished) - started, 3),
        "total_request_seconds": round(finished - started, 3),
        "pcm_bytes": received,
        "audio_seconds": round(received / 2 / 32000 / args.channels, 3),
        "format": "pcm_s16le", "sample_rate": 32000, "channels": args.channels,
        "verified": not args.no_verify,
        "output": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
