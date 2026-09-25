#!/usr/bin/env python3
"""Create stable H3 anchors and verified stereo previews for built-in voices."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from minimax_voice_api.config import DATA_ROOT
from minimax_voice_api.core import BUILTIN_VOICES, VoiceRegistry
from minimax_voice_api.engine import VoiceEngine

PREVIEW_LINES = {
    "warm_narrator": "The door was already open, though nobody remembered hearing the bell.",
    "velvet_goth": "You can pretend this doesn't interest you, but your coffee has gone completely cold.",
    "bright_assistant": "I found it! And, yes, it was in the one place we both swore we'd checked.",
    "young_hero": "Okay, I'm nervous. That doesn't mean I'm leaving you to do this alone.",
    "elder_storyteller_f": "Now, your grandfather always denied this part, which is how I know it happened.",
    "elder_storyteller_m": "Sit down a moment. The truth is shorter than the rumor, but considerably stranger.",
    "soft_asmr": "There you are. You don't have to say anything; just get comfortable and breathe.",
    "dry_detective": "The alibi was spotless. People are rarely that tidy unless they've made a mess.",
    "calm_professor": "Let's slow down and separate what we know from what we've merely assumed.",
    "radio_host": "It's eleven forty-seven, the rain is easing, and you're listening to the night shift.",
    "gentle_caregiver": "I've got the kettle on. Tell me what would make tonight a little easier.",
    "comic_best_friend": "I support your decision completely. I also brought snacks for when it explodes.",
    "grounded_actress": "I didn't come here to win the argument. I came because I missed you.",
    "witty_british_f": "Brilliant plan. One tiny concern: the front door appears to be on fire.",
    "smoky_jazz_f": "The last set ended an hour ago, but the room still hasn't quite let go of it.",
    "earnest_young_f": "Wait, you kept it all this time? Sorry, I just wasn't ready for that.",
    "mature_executive_f": "Close the door, please. We can solve this, but first I need the honest version.",
    "natural_neighbor_m": "Hey, your porch light was blinking again, so I brought my ladder over.",
    "british_stage_m": "I assure you, I intended a graceful entrance. The umbrella had other ambitions.",
    "soft_spoken_m": "I saved you the window seat. It's quieter here once the last train leaves.",
    "rugged_western_m": "Storm's moving east. If we leave now, we'll reach the pass before the road washes out.",
    "animated_comic_m": "Technically, the experiment worked. The ceiling fan is simply part of it now.",
    "cinematic_ai_companion_f": "You went quiet again. I don't mind. I can stay here and notice the things you don't say.",
    "velvet_siren_f": "Come sit closer. I was enjoying the way you looked at me, and I'd rather not waste the moment.",
    "attached_girlfriend_f": "There you are! I only checked the window three times. Four, technically, but the last one barely counts.",
    "commanding_boss_f": "Sit down, close the door, and give me the honest version. We can handle the consequences afterward.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=DATA_ROOT / "voice-previews")
    parser.add_argument("--anchors-only", action="store_true")
    parser.add_argument("--only", nargs="*", choices=sorted(BUILTIN_VOICES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = VoiceRegistry()
    engine = VoiceEngine(registry=registry)
    voice_ids = args.only or list(BUILTIN_VOICES)
    report = {"settings": {"resolution": 32, "steps": 30, "channels": 2,
                           "spatial_preset": "close"}, "voices": []}

    profiles = []
    for voice_id in voice_ids:
        started = time.monotonic()
        profile = engine.ensure_voice(registry.get(voice_id), steps=30)
        profiles.append(profile)
        entry = {
            "id": voice_id, "name": profile.name, "gender": profile.gender,
            "anchor": profile.reference_path, "anchor_transcript": profile.transcript,
            "anchor_seconds": round(time.monotonic() - started, 2),
        }
        report["voices"].append(entry)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)

    if args.anchors_only:
        return
    for profile, entry in zip(profiles, report["voices"]):
        started = time.monotonic()
        text = PREVIEW_LINES[profile.id]
        results = list(engine.generate(
            text=text, voice=profile.id,
            direction=("nuanced realistic professional acting, speaking naturally "
                       "to one nearby person, with connected phrasing and no TTS cadence"),
            steps=30, resolution=32, space="close", channels=2,
            target_words=18, max_words=24, verify=True, max_retries=2,
            min_similarity=0.90, strict_fidelity=True))
        rendered = engine.assemble(results, "flac")
        preview = args.output_dir / f"{profile.id}.flac"
        shutil.copy2(rendered, preview)
        entry.update({
            "preview": str(preview.resolve()), "text": text,
            "transcript": " ".join(result.transcript for result in results),
            "similarity": min(result.similarity for result in results),
            "attempts": sum(result.attempts for result in results),
            "preview_seconds": round(time.monotonic() - started, 2),
        })
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
