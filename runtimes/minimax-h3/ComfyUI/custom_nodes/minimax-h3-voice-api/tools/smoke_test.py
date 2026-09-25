#!/usr/bin/env python3
"""Exercise the live OpenAI- and ElevenLabs-compatible speech routes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

CASES = (
    {
        "name": "short_ai_companion",
        "route": "/v1/audio/speech",
        "payload": {
            "input": "Hey. You made it. I was hoping I'd get a minute alone with you.",
            "voice": "cinematic_ai_companion_f",
            "emotion": "tender",
            "emotion_intensity": 0.55,
            "instructions": "warm husky cinematic acting, intimate and quietly amused, speaking naturally to one nearby person",
            "response_format": "flac",
        },
    },
    {
        "name": "medium_attached_girlfriend",
        "route": "/v1/audio/speech",
        "payload": {
            "input": (
                "There you are! I wasn't worried. I just happened to check the window, "
                "then the hallway, then my phone. Come here and tell me everything. "
                "Start with why you took so long, and make the explanation adorable."
            ),
            "voice": "attached_girlfriend_f",
            "emotion": "playful",
            "emotion_intensity": 0.72,
            "instructions": "natural romantic-comedy acting, delighted and intensely affectionate, quick playful reactions without cartoon exaggeration",
            "response_format": "flac",
        },
    },
    {
        "name": "long_commanding_boss",
        "route": "/v1/text-to-speech/commanding_boss_f",
        "payload": {
            "text": (
                "Close the door and sit down. Before you start apologizing, take a breath. "
                "I don't need a polished story; I need the version that actually happened. "
                "Tell me what went wrong, what you tried, and what you need from me now. "
                "We can deal with a mistake. What we cannot deal with is wasting another hour "
                "pretending there wasn't one. Good. Now look at me and begin at the beginning."
            ),
            "emotion": "authoritative",
            "instructions": "grounded elite dramatic acting, controlled amused authority, crisp connected phrasing, speaking across a desk to one person",
            "output_format": "flac",
            "voice_settings": {
                "stability": 0.67,
                "similarity_boost": 0.9,
                "style": 0.62,
                "speed": 1.0,
                "use_speaker_boost": True,
            },
        },
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/smoke-tests"))
    parser.add_argument("--only", nargs="*", choices=[case["name"] for case in CASES])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = [case for case in CASES if not args.only or case["name"] in args.only]
    report: dict[str, object] = {"base_url": args.base_url, "results": []}

    with httpx.Client(timeout=None) as client:
        health = client.get(f"{args.base_url}/health")
        health.raise_for_status()
        report["health"] = health.json()
        for case in chosen:
            started = time.monotonic()
            response = client.post(f"{args.base_url}{case['route']}", json=case["payload"])
            response.raise_for_status()
            output = args.output_dir / f"{case['name']}.flac"
            output.write_bytes(response.content)
            entry = {
                "name": case["name"],
                "route": case["route"],
                "output": str(output.resolve()),
                "bytes": len(response.content),
                "seconds_to_complete": round(time.monotonic() - started, 3),
                "voice_id": response.headers.get("x-voice-id"),
                "chunks": int(response.headers.get("x-voice-chunks", "0")),
                "asr_verification": response.headers.get("x-asr-verification"),
            }
            with output.open("rb") as audio:
                transcription = client.post(
                    f"{args.base_url}/v1/audio/transcriptions",
                    files={"file": (output.name, audio, "audio/flac")},
                    data={"response_format": "verbose_json"},
                )
            transcription.raise_for_status()
            transcript = transcription.json()
            entry["transcript"] = transcript["text"]
            entry["transcript_words"] = len(transcript["words"])
            entry["asr_seconds"] = float(
                transcription.headers.get("x-asr-seconds", "0"))
            report["results"].append(entry)
            (args.output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
