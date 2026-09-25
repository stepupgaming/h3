# Purpose

Internalized MiniMax-H3 voice-api pack for Gemmy product audio:

- Speech (Ref2VA TTS + Parakeet verify)
- Music (song / instrumental)
- SFX (diegetic oneshot / loop_bed)

Serves OpenAI/ElevenLabs-shaped HTTP and native Comfy studio nodes.

# Ownership

- This tree owns FastAPI app, presets, prompt builders, Comfy node expansions, studio web UI, tests.
- Product CLI is `gemmy audio-h3` in the Gemmy repo (`src/commands/modalities/audio_h3/`). Speech/music/SFX go through this FastAPI pack. **Convert** (`gemmy audio-h3 convert` / `gemmy audio-h3-convert`) does **not** use this sidecar — it is the headless video-h3 Comfy worker with `init_audio` + `denoise`.
- Shared product Comfy pack: sibling `../minimax-h3/ComfyUI` (junction at `custom_nodes/minimax-h3-voice-api`).
- Continuous FL2VA speech needs the optional local GPL clone `../minimax-h3/ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context` @ **v0.3.1** (`725a731`). Default Ref2VA speech does not. Do not replace video continue/loop with this pack.
- Shared external weights: `GEMMY_H3_CHECKPOINTS` / video-h3 checkpoint root — never vendored here.
- Speech DiT: `--dit stock` (default) = `minimax_h3_ref2va_pruned_int8_convrot.safetensors` on `G:\Models\minimax-h3-backup`. `--dit eros` = Eros INT8 on `F:\Models\minimax-h3-eros`. Both roots must be on Comfy `extra_model_paths.yaml` (written by `gemmy audio-h3 install`). Restart `:8188` after that file changes.
- Job/voice data: `H3_DATA_DIR` (default Gemmy `outputs/h3_voice_api_data`).

# Local Contracts

- Internal-only: Gemmy must not default to research `C:\Projects\minimax-h3\minimax-h3-voice-api`.
- Python via **uv** (`pyproject.toml` + `uv.lock` + `.python-version` 3.12).
- No disk/NVMe weight offload; Comfy must stay on RAM↔VRAM policy used by video-h3.
- SFX is first-class (`diegetic_v1`), not a music-instrumental hack.
- Serialized H3 jobs inside the API; Gemmy still GPU-handoffs llama before long jobs.

# Work Guidance

- Refresh from upstream/research pack by copying source only; keep product `.env` pointing at Gemmy Comfy + data dir.
- After dep changes: `uv lock --python 3.12 && uv sync --python 3.12`.
- Verify with `gemmy audio-h3 install|doctor` and package pytest when touching Python.

# Verification

- `uv run pytest` (package unit/API contract tests; no GPU)
- `gemmy audio-h3 doctor` (stock Ref2VA on G: and Eros INT8 required; extra_model_paths includes both)
- Live: Comfy :8188 + `gemmy audio-h3 serve` + speech/music/sfx smoke

# Child DOX Index

(none)
