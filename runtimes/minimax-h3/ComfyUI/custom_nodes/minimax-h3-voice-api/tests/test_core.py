import re
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import minimax_voice_api.engine as engine_module
from minimax_voice_api.core import (
    VoiceRegistry,
    audio_signal_metrics,
    chunk_text,
    onset_orphan_cut,
    safe_id,
    word_similarity,
)
from minimax_voice_api.engine import (
    VoiceEngine,
    continuous_speech_prompt,
    exact_speech_bounds,
    frames_for,
    motion_context_speech_graph,
    performance_text,
    ref2va_graph,
    requested_speech_start,
    speech_prompt,
    speech_seconds,
    t2va_graph,
    transcript_exact_match,
)
from minimax_voice_api.music import (
    instrumental_prompt,
    parse_lyrics,
    recommend_song_duration,
    song_prompt,
)
from minimax_voice_api.sfx import recommend_sfx_duration, sfx_prompt
from minimax_voice_api.sfx_presets import SFX_PRESETS
from minimax_voice_api.transcription import subtitles, timestamped_words, transcription_segments


def test_safe_id():
    assert safe_id("My Velvet Voice!") == "my_velvet_voice"


def test_chunk_text_preserves_words_and_limits():
    text = ("This is the first complete sentence. Here is a much longer sentence, "
            "with enough words to require a careful split, while preserving every single word exactly.")
    chunks = chunk_text(text, target_words=8, max_words=12)
    assert " ".join(chunks).split() == text.split()
    assert all(len(chunk.split()) <= 12 for chunk in chunks)


def test_word_similarity():
    assert word_similarity("Hello there, my friend.", "hello there my friend") == 1.0
    assert word_similarity("one two three four", "one two four") == 0.75
    assert word_similarity("one two", "gibberish") < 0.5
    assert word_similarity("streetlights glow", "street lights glow") == 1.0
    assert word_similarity("streetlights glow", "street signs glow") < 1.0


def test_h3_frame_alignment_and_duration():
    assert frames_for(5) % 17 == 5
    assert frames_for(5.8) == 141
    assert 2.5 <= speech_seconds("Hello world.") <= 13.5


def test_exact_transcript_rejects_script_surrounded_by_gibberish():
    expected = "Only these words."
    assert transcript_exact_match(expected, "Only these words!")
    assert not transcript_exact_match(expected, "noise Only these words")
    assert not transcript_exact_match(expected, "Only these words more noise")


def test_exact_render_rejects_embedded_script_in_raw_take(tmp_path, monkeypatch):
    expected = "Only these words."

    class Result:
        text = "invented noise Only these words more invented noise"
        tokens = [" invented", " noise", " Only", " these", " words",
                  " more", " invented", " noise"]
        timestamps = [0.1, 0.3, 0.6, 0.8, 1.0, 1.3, 1.5, 1.7]

    class Comfy:
        def run(self, graph, _detail):
            assert "/raw_unverified/" in graph["93"]["inputs"]["filename_prefix"]
            return tmp_path / "raw.flac", 1.0

    class Verifier:
        def transcribe_timestamped(self, _path):
            return Result()

    class Registry:
        def comfy_reference(self, _profile):
            return "voice_api/reference.flac"

    def preprocess(_source, destination, _seconds, channels=2):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"candidate")

    monkeypatch.setattr(engine_module, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(engine_module, "preprocess_generated", preprocess)
    monkeypatch.setattr(engine_module, "remove_onset_orphan",
                        lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(engine_module, "audio_duration", lambda _path: 2.0)
    monkeypatch.setattr(engine_module, "audio_signal_metrics", lambda _path: {
        "peak_dbfs": -6.0, "clipped_sample_fraction": 0.0,
        "dc_offset": 0.0, "onset_orphan_cut_seconds": None,
    })
    monkeypatch.setattr(engine_module, "speech_activity_margins",
                        lambda _path: (0.1, 0.2))
    engine = VoiceEngine(registry=Registry(), comfy=Comfy(), verifier=Verifier())
    monkeypatch.setattr(engine, "_ensure_dark_image", lambda: None)

    with pytest.raises(engine_module.SpeechFidelityError) as error:
        engine.render_chunk(
            SimpleNamespace(id="voice", description="A natural adult voice."),
            expected, "a" * 32, 0,
            steps=20, speed=1.0, direction="natural", verify=True,
            max_retries=0, min_similarity=0.85, strict_fidelity=True)
    assert "raw take is not an exact full-script" in str(error.value)
    assert "invented noise" in error.value.transcript


def test_onset_orphan_detector_finds_isolated_h3_blip(tmp_path):
    rate = 32000
    burst = 0.08 * np.sin(2 * np.pi * 700 * np.arange(round(rate * 0.18)) / rate)
    silence = np.zeros(round(rate * 0.72), np.float32)
    speech = 0.12 * np.sin(2 * np.pi * 180 * np.arange(round(rate * 0.70)) / rate)
    path = tmp_path / "artifact.flac"
    sf.write(path, np.concatenate((burst, silence, speech)), rate)
    cut = onset_orphan_cut(path)
    assert cut is not None
    assert 0.80 < cut < 0.90
    assert audio_signal_metrics(path)["onset_orphan_cut_seconds"] == cut


def test_performance_text_adds_pause_without_changing_words():
    text = "I did not come here to win the argument."
    performed = performance_text(text)
    assert "..." in performed
    assert performed.replace(" ...", "") == text


def test_described_voice_has_stable_cache_id(tmp_path, monkeypatch):
    import minimax_voice_api.core as core
    monkeypatch.setattr(core, "VOICE_ROOT", tmp_path / "voices")
    registry = VoiceRegistry()
    first = registry.from_description("A calm adult voice.", "Calm")
    second = registry.from_description("A calm adult voice.", "Ignored")
    assert first.id == second.id
    assert second.description == "A calm adult voice."


def test_timestamp_alignment_removes_reference_leadin_and_tail():
    class Result:
        text = "garbled words Hello there friend extra noise"
        tokens = [" garbled", " words", " Hello", " there", " friend", " extra", " noise"]
        timestamps = [0.1, 0.5, 1.0, 1.3, 1.6, 2.1, 2.4]

    start, end = exact_speech_bounds(Result(), "Hello there friend")
    assert 0.8 < start < 1.0
    assert 1.8 < end < 1.9


def test_timestamp_alignment_accepts_split_compounds():
    class Result:
        text = "vocal noise The street lights glow now"
        tokens = [" vocal", " noise", " The", " street", " lights", " glow", " now"]
        timestamps = [0.1, 0.4, 1.0, 1.2, 1.4, 1.7, 2.0]

    start, end = exact_speech_bounds(Result(), "The streetlights glow now")
    assert 0.8 < start < 1.0
    assert end is None


def test_requested_speech_start_ignores_leading_artifact_and_later_error():
    class Result:
        text = "noise noise Hello there friend today a later mistake"
        tokens = [" noise", " noise", " Hello", " there", " friend", " today",
                  " a", " later", " mistake"]
        timestamps = [0.1, 0.3, 2.0, 2.2, 2.4, 2.6, 3.0, 3.2, 3.4]

    start = requested_speech_start(
        Result(), "Hello there friend today but this ending differs")
    assert start is not None
    assert 1.8 < start < 2.0


def test_transcription_words_segments_and_subtitles():
    class Result:
        text = "Hello there. This works!"
        tokens = [" Hello", " there", ".", " This", " works", "!"]
        timestamps = [0.4, 0.8, 1.1, 1.5, 1.9, 2.2]

    words = timestamped_words(Result(), 2.6)
    assert [word.word for word in words] == ["Hello", "there", "This", "works"]
    assert words[0].start == 0.4
    assert words[1].end < words[2].start
    segments = transcription_segments(Result.text, words)
    assert [segment["text"] for segment in segments] == ["Hello there.", "This works!"]
    assert "00:00:00,400 -->" in subtitles(segments, "srt")
    assert subtitles(segments, "vtt").startswith("WEBVTT\n\n")


def test_voice_direction_cannot_become_dialogue():
    profile = VoiceRegistry().get("cinematic_ai_companion_f")
    target = "These are the only words I asked you to say."
    prompt = speech_prompt(
        profile, target,
        "warm husky acting notes that must never be spoken aloud",
    )
    blocks = re.findall(r"<d>\[English] (.*?)</d>", prompt, re.DOTALL)
    assert blocks == [target]
    assert "warm husky acting notes" in prompt
    assert "warm husky acting notes" not in blocks[0]


def test_continuous_prompt_has_one_exact_dialogue_boundary():
    profile = VoiceRegistry().get("grounded_actress")
    target = "I remembered the coffee, and I brought the ridiculous little mug."
    prompt = continuous_speech_prompt(
        profile, target, "quietly amused natural acting", index=1, total=3)
    assert re.findall(r"<d>\[English] (.*?)</d>", prompt, re.DOTALL) == [target]
    assert "same uninterrupted conversation" in prompt
    assert "quietly amused natural acting" not in re.findall(
        r"<d>.*?</d>", prompt, re.DOTALL)[0]


def test_motion_context_graph_carries_latent_audio_and_trims_overlap():
    graph, outputs = motion_context_speech_graph(
        ["first prompt", "second prompt"], [10.0, 10.0], 30,
        [101, 202], "test/continuous", resolution=32)
    classes = {node["class_type"] for node in graph.values()}
    assert "PathchSageAttentionKJ" not in classes
    assert {"MiniMaxH3MotionContext", "MiniMaxH3MotionContextTrim"} <= classes
    context = next(node for node in graph.values()
                   if node["class_type"] == "MiniMaxH3MotionContext")
    assert context["inputs"]["context_latent"] == ["1003", 0]
    assert context["inputs"]["context_length"] == "22"
    assert context["inputs"]["audio_context_length"] == 240
    assert "audio_mode" not in context["inputs"]
    assert len(outputs) == 2


def test_music_prompts_keep_direction_outside_lyric_blocks():
    lyrics = "[Verse]\nPet rock by the radio\nWaiting for the night to glow\n\n[Chorus]\nRoll on home\nRoll on home"
    prompt, plan = song_prompt(
        "bright handmade power pop", "crunchy guitar and live drums",
        "a playful woman with a low smoky voice", lyrics, 60.0,
    )
    assert re.findall(r"<d>\[English] (.*?)</d>", prompt, re.DOTALL) == [
        "Pet rock by the radio\nWaiting for the night to glow",
        "Roll on home\nRoll on home",
    ]
    assert "playful woman" not in " ".join(re.findall(r"<d>.*?</d>", prompt, re.DOTALL))
    assert "[Shot 2]" not in prompt
    assert "One uninterrupted original musical performance" in prompt
    assert "From about 00:" in prompt
    assert len(plan) == 2
    assert plan[-1]["end"] == 60.0
    assert parse_lyrics("[Intro]\n\n[Verse]\nHello") == [
        ("intro", []), ("verse", ["Hello"]),
    ]


def test_legacy_music_profile_remains_available_for_reproduction():
    prompt, _ = song_prompt(
        "handmade pop", "guitar and drums", "one natural singer",
        "[Verse]\nFirst full line\nSecond full line\n\n[Chorus]\nSing it again",
        30.0, profile="petrock_timed_v1",
    )
    assert "[Shot 2] At 00:" in prompt
    assert prompt.count("<d>[English]") == 3


def test_song_duration_is_style_aware_and_capped():
    lyrics = "[Verse]\n" + "\n".join(["one two three four five six"] * 12)
    slow = recommend_song_duration("a slow ballad at 70 BPM", lyrics)
    rap = recommend_song_duration("fast rap at 150 BPM", lyrics)
    assert slow["recommended_seconds"] >= rap["recommended_seconds"]
    assert slow["recommended_seconds"] <= 60
    assert slow["words"] == rap["words"] == 72


def test_instrumental_prompt_contains_no_dialogue_block():
    prompt = instrumental_prompt("slow nocturnal jazz", "piano and upright bass")
    assert "<d>" not in prompt
    assert "No singing" in prompt


def test_sfx_prompt_forbids_speech_and_score():
    prompt = sfx_prompt(
        "A wooden door latch clicks shut",
        kind="oneshot",
        space="a quiet residential hallway",
        intensity=0.8,
    )
    assert "<d>" not in prompt
    assert "non_diegetic_music: N/A" in prompt
    assert "No singing" in prompt or "no singing" in prompt.casefold()
    assert "door latch" in prompt.casefold()
    assert "dialogue" in prompt.casefold()


def test_sfx_loop_bed_prompt_prefers_continuous_texture():
    prompt = sfx_prompt("Steady gentle rain on a window", kind="loop_bed")
    assert "continuous" in prompt.casefold()
    assert "loop" in prompt.casefold() or "bed" in prompt.casefold()


def test_sfx_duration_heuristics_differ_by_kind():
    click = recommend_sfx_duration("A single soft UI click", kind="oneshot")
    bed = recommend_sfx_duration("Steady forest ambience bed", kind="loop_bed")
    footsteps = recommend_sfx_duration(
        "Several footsteps on gravel then a stop", kind="oneshot")
    assert click["recommended_seconds"] <= 3.0
    assert bed["recommended_seconds"] >= 6.0
    assert footsteps["recommended_seconds"] >= click["recommended_seconds"]
    assert footsteps["multi_event"] is True


def test_sfx_preset_catalog_is_broad():
    assert len(SFX_PRESETS) >= 40
    categories = {item["category"] for item in SFX_PRESETS.values()}
    assert {"Impacts", "Whooshes", "Foley", "Ambience", "UI"} <= categories
    assert all(
        item["kind"] in {"oneshot", "loop_bed"} for item in SFX_PRESETS.values()
    )


def test_populated_intro_is_not_described_as_instrumental():
    prompt, _ = song_prompt(
        "bright pop", "live drums and guitar", "a natural singer",
        "[Intro]\nSing this opening\n\n[Outro]\nSing this ending", 20.0,
    )
    assert "opening vocal hook" in prompt
    assert "sung outro" in prompt
    assert "instrumental introduction with no vocal" not in prompt


def test_production_graphs_need_only_standard_comfy_nodes():
    graphs = [
        t2va_graph("prompt", 5, 20, 1, "test/t2va"),
        ref2va_graph("prompt", "voice.flac", "dark.png", 5, 20, 1,
                     "test/ref2va"),
    ]
    classes = {node["class_type"] for graph in graphs for node in graph.values()}
    assert "PathchSageAttentionKJ" not in classes
    assert "MiniMaxH3ReferenceAudioOnly" not in classes
    assert {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"} <= classes


def test_ref2va_graph_selects_stock_or_eros_unet():
    from minimax_voice_api.config import UNET_REF2VA, UNET_REF2VA_EROS, unet_for_dit

    stock = ref2va_graph("prompt", "voice.flac", "dark.png", 5, 20, 1,
                         "test/stock")
    eros = ref2va_graph("prompt", "voice.flac", "dark.png", 5, 20, 1,
                        "test/eros", unet=UNET_REF2VA_EROS)
    assert stock["6"]["inputs"]["unet_name"] == UNET_REF2VA
    assert eros["6"]["inputs"]["unet_name"] == UNET_REF2VA_EROS
    assert unet_for_dit("stock") == UNET_REF2VA
    assert unet_for_dit("eros") == UNET_REF2VA_EROS
    assert unet_for_dit("EROS") == UNET_REF2VA_EROS


def test_reference_keep_is_opt_in_and_defaults_to_literal_clone():
    strict = ref2va_graph(
        "prompt", "voice.flac", "dark.png", 5, 20, 1, "test/strict")
    freer = ref2va_graph(
        "prompt", "voice.flac", "dark.png", 5, 20, 1, "test/freer",
        audio_ref_keep=0.97)
    assert "105" not in strict
    assert strict["16"]["inputs"]["conditioning"] == ["104", 0]
    assert freer["105"]["class_type"] == "H3RefKeep"
    assert freer["105"]["inputs"]["audio_keep"] == 0.97
