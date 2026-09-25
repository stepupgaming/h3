import json
from pathlib import Path

from fastapi.testclient import TestClient

from minimax_voice_api import app as app_module
from minimax_voice_api.app import ASR_MODEL, app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_health_and_discovery_contracts():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["parakeet"] == "cpu-lazy"
    assert health.json()["warmth"]["enabled"] is True
    voices = client.get("/v1/voices").json()["data"]
    assert len(voices) >= 21
    assert {"warm_narrator", "velvet_goth", "soft_asmr"} <= {v["id"] for v in voices}
    assert sum(v.get("gender") == "female" for v in voices) >= 10
    assert sum(v.get("gender") == "male" for v in voices) >= 10
    assert "playful" in client.get("/v1/audio/emotions").json()["data"]
    model_ids = {item["id"] for item in client.get("/v1/models").json()["data"]}
    assert "minimax-h3-ref2va" in model_ids
    assert "minimax-h3-ref2va-eros" in model_ids
    assert client.get("/v1/warmth").status_code == 200


def test_browser_studio_is_served():
    page = client.get("/")
    assert page.status_code == 200
    assert "Vox Atelier" in page.text
    script = client.get("/studio-assets/app.js")
    assert script.status_code == 200
    assert "v1/audio/speech" in script.text


def test_music_catalog_has_broad_genre_and_subgenre_coverage():
    presets = client.get("/v1/audio/music/presets").json()["data"]
    categories = {preset["category"] for preset in presets}
    assert len(presets) >= 90
    assert {"Pop", "K-pop", "Rock", "Metal", "Hip-hop / Rap", "R&B / Soul",
            "Electronic", "Jazz / Blues", "Country / Folk", "Latin / Global",
            "Cinematic / Media"} <= categories
    assert all({"id", "name", "category", "style", "instrumentation", "vocalist"} <= preset.keys()
               for preset in presets)


def test_sfx_catalog_and_routes_are_first_class():
    presets = client.get("/v1/audio/sfx/presets").json()["data"]
    categories = {preset["category"] for preset in presets}
    assert len(presets) >= 40
    assert {"Impacts", "Whooshes", "Foley", "Ambience", "UI"} <= categories
    assert all(
        {"id", "name", "category", "description", "kind", "default_seconds"}
        <= preset.keys()
        for preset in presets
    )
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/audio/sfx" in paths
    assert "/v1/audio/sfx/presets" in paths
    assert "/v1/audio/sfx/duration" in paths
    assert "/v1/audio/sfx/jobs/{job_id}/prompt" in paths
    body = app.openapi()["components"]["schemas"]["SfxRequest"]["properties"]
    assert body["kind"]["default"] == "oneshot"
    assert body["duration_mode"]["default"] == "auto"
    assert body["prompt_profile"]["default"] == "diegetic_v1"
    models = {item["id"] for item in client.get("/v1/models").json()["data"]}
    assert "minimax-h3-sfx" in models
    page = client.get("/").text
    assert 'data-creation="sfx"' in page
    assert 'id="sfxStudio"' in page
    script = client.get("/studio-assets/app.js").text
    assert "v1/audio/sfx" in script


def test_music_studio_and_api_default_to_full_song():
    body = app.openapi()["components"]["schemas"]["MusicRequest"]["properties"]
    assert body["mode"]["default"] == "song"
    assert body["duration_seconds"]["default"] == 60.0
    assert body["duration_mode"]["default"] == "auto"
    assert body["prompt_profile"]["default"] == "natural_song_sheet_v2"
    page = client.get("/").text
    assert 'class="music-kind-tab active" data-kind="song"' in page
    assert 'id="musicDuration" type="range" min="5" max="60" step="1" value="60"' in page
    assert 'id="viewPromptButton"' in page
    assert 'id="musicAutoDuration" type="checkbox" checked' in page
    assert "/v1/audio/music/duration" in client.get("/openapi.json").json()["paths"]
    assert "/v1/audio/music/jobs/{job_id}/prompt" in client.get("/openapi.json").json()["paths"]
    assert 'id="songFields" class="song-fields"' in page


def test_comfyui_music_workflow_is_packaged_and_ready_to_run():
    workflow_path = REPO_ROOT / "workflows" / "H3 Music Studio - 60 Second Song.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    nodes = {node["type"]: node for node in workflow["nodes"]}
    assert {"MiniMaxH3MusicStudio", "PreviewAudio", "SaveAudio"} <= nodes.keys()
    widgets = nodes["MiniMaxH3MusicStudio"]["widgets_values"]
    assert widgets[0] == "song"
    assert widgets[1] == "Custom"
    assert widgets[2] == 60.0
    assert len(widgets) == 22
    assert widgets[4] == 1234
    assert widgets[5] == "fixed"
    assert "[Bridge]" in widgets[9]
    assert widgets[15] == "simple"
    assert widgets[16] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert widgets[17] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert widgets[18] == "minimax_h3_video_vae_int8_convrot.safetensors"
    assert widgets[19] == "minimax_h3_audio_vae_fp32.safetensors"
    assert widgets[-2:] == ["auto", "natural_song_sheet_v2"]
    assert (REPO_ROOT / "comfyui_nodes.py").is_file()


def test_openai_request_schema():
    schema = app.openapi()
    body = schema["components"]["schemas"]["SpeechRequest"]["properties"]
    for field in ("model", "input", "voice", "response_format", "speed", "instructions",
                  "resolution", "spatial_preset", "strict_fidelity", "speech_mode",
                  "continuity_context_seconds", "fidelity_mode", "candidate_pool_size",
                  "generation_padding_seconds", "target_duration_seconds",
                  "audio_ref_keep", "dit"):
        assert field in body
    assert body["dit"]["default"] == "stock"
    assert body["speech_mode"]["default"] == "reference"
    assert body["fidelity_mode"]["default"] == "exact"
    assert body["generation_padding_seconds"]["default"] == 0.1
    assert body["continuity_context_seconds"]["default"] == 10.0
    assert "/v1/audio/music" in schema["paths"]
    assert "/v1/audio/music/presets" in schema["paths"]
    assert "/v1/audio/sfx" in schema["paths"]
    assert "/v1/audio/transcriptions" in schema["paths"]


def test_continuous_speech_model_and_frontend_control_are_exposed():
    models = {item["id"] for item in client.get("/v1/models").json()["data"]}
    assert "minimax-h3-fl2va-continuous" in models
    page = client.get("/").text
    assert 'id="speechMode"' in page
    assert "Continuous FL2VA" in page


def test_elevenlabs_routes_exist():
    paths = app.openapi()["paths"]
    assert "/v1/voices/add" in paths
    assert "/v1/text-to-speech/{voice_id}" in paths
    assert "/v1/text-to-speech/{voice_id}/stream" in paths


def test_transcription_route_returns_openai_and_timestamp_formats(monkeypatch):
    class Result:
        text = "Hello there. This works."
        tokens = [" Hello", " there", ".", " This", " works", "."]
        timestamps = [0.4, 0.8, 1.1, 1.6, 2.0, 2.3]
        logprobs = [-0.1] * 6

    monkeypatch.setattr(app_module, "audio_duration", lambda _: 2.8)
    monkeypatch.setattr(
        app_module.engine.verifier, "transcribe_timestamped", lambda _: Result())
    upload = {"file": ("sample.mp4", b"fake audio or video", "video/mp4")}

    basic = client.post(
        "/v1/audio/transcriptions",
        files=upload,
        data={"model": "whisper-1", "response_format": "json"},
    )
    assert basic.status_code == 200
    assert basic.json() == {"text": "Hello there. This works."}
    assert basic.headers["x-asr-model"] == ASR_MODEL
    assert basic.headers["x-asr-device"] == "cpu-int8"

    verbose = client.post(
        "/v1/audio/transcriptions",
        files=upload,
        data={"response_format": "verbose_json"},
    )
    assert verbose.status_code == 200
    assert verbose.json()["language"] == "english"
    assert [word["word"] for word in verbose.json()["words"]] == [
        "Hello", "there", "This", "works"]
    assert len(verbose.json()["segments"]) == 2

    subtitles = client.post(
        "/v1/audio/transcriptions",
        files=upload,
        data={"response_format": "vtt"},
    )
    assert subtitles.status_code == 200
    assert subtitles.text.startswith("WEBVTT\n\n")
    assert "Hello there." in subtitles.text


def test_transcription_rejects_unsupported_language(monkeypatch):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", b"not reached", "audio/wav")},
        data={"language": "French"},
    )
    assert response.status_code == 422
    assert "English only" in response.json()["detail"]
