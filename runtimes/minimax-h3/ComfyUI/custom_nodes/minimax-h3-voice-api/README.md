# MiniMax H3 Voice API

A local FastAPI service that turns MiniMax H3 in ComfyUI into an OpenAI- and
ElevenLabs-style voice platform. It includes long-form speech, streaming,
described and cloned voices, CPU-only transcript verification, stereo spatial
delivery, instrumentals, structured songs, 99 music presets, **first-class SFX /
Foley** (one-shots and ambience beds), and the responsive **Vox Atelier**
browser studio.

This repository contains application code only. It does not redistribute
MiniMax, Qwen, or codec checkpoints.

## Features

- OpenAI-compatible `POST /v1/audio/speech`
- OpenAI-compatible CPU transcription at `POST /v1/audio/transcriptions`
- ElevenLabs-compatible `/v1/text-to-speech/{voice_id}` routes
- Arbitrary-length text with punctuation-aware chunking and smooth joins
- Verified Ref2VA speech by default, with experimental continuous FL2VA and up
  to 10 seconds of direct latent voice memory between chunks
- PCM streaming: the first accepted chunk plays while the next one renders
- 26 built-in voice personas plus described voices
- Voice cloning from audio or video; FFmpeg extracts and normalizes it
- Acting direction, emotion, speed, room perspective, and stereo output
- Exact full-transcript Parakeet verification, automatic retries, candidate
  pooling, final encoded-file validation, and machine-readable quality reports
- Startup-blip detection, clipping checks, natural silence-aware joins, exact
  target-duration fitting, and correctly encoded FLAC/WAV/MP3/Opus/AAC output
- Optional `audio_ref_keep` conditioning control from literal cloning (`1.0`)
  through conservatively freer delivery (`0.95-0.99`)
- Instrumentals and songs with exact `[Verse]`, `[Chorus]`, `[Bridge]`, and
  other labeled lyric sections, translated into timestamped H3 shot anchors
- 99 searchable music presets in 11 genre families
- First-class SFX / Foley at `POST /v1/audio/sfx` with one-shot and loop-bed
  kinds, diegetic prompts that forbid speech and score, and a curated preset
  library (impacts, whooshes, UI, Foley, ambience, vehicles, sci-fi, nature)
- Native ComfyUI **H3 Music Studio** and **H3 SFX Studio** nodes
- Serialized H3 jobs to protect 24 GB GPUs from overlapping generations
- Model-hot status, queue state, and interactive OpenAPI documentation
- No LLM and no GPU ASR process

## Requirements

- Linux and Python 3.11 or 3.12
- FFmpeg and FFprobe on `PATH`
- A working recent ComfyUI installation with MiniMax H3 nodes
- [ComfyUI H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
  **v0.3.1** (`725a731`) as a local GPL clone under product
  `runtimes/minimax-h3/ComfyUI/custom_nodes/ComfyUI-H3-Motion-Context` (gitignored)
  for continuous FL2VA speech. Product Comfy is 0.36.0; v0.6.2 is possible later
  (not this pass). Default Ref2VA speech does not need it. This is not
  `gemmy-h3-context` (video continue/loop).
- An NVIDIA GPU with enough memory for the selected H3 checkpoints; this stack
  was developed on a 24 GB RTX 4090
- These configured filenames in the corresponding ComfyUI model folders:

| ComfyUI folder | Default filename |
|---|---|
| `models/diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `models/diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` (default) |
| `models/vae` | `minimax_h3_video_vae_fp16.safetensors` (`--vae fp16`) |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |

Model filenames are configurable. The production graph uses standard ComfyUI
H3 and sampling nodes plus the two Motion Context continuation nodes. This
repository supplies the small optional `H3RefKeep` conditioning node itself.

## Install

```bash
git clone <your-repository-url> minimax-h3-voice-api
cd minimax-h3-voice-api

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

cp .env.example .env
```

Edit `.env` and set `H3_COMFY_ROOT` to the directory containing
`ComfyUI/main.py`. If the API and ComfyUI repositories are siblings, the default
already resolves `../ComfyUI`.

Start ComfyUI on its configured URL, normally `http://127.0.0.1:8188`, then
validate the complete integration:

```bash
h3-check
```

The command checks FFmpeg, ComfyUI, required nodes, and all five model files
without running the GPU.

## Run

```bash
./run.sh
```

Or use the installed command:

```bash
h3-voice-api --host 0.0.0.0 --port 8787
```

Open:

- Browser studio: `http://localhost:8787/`
- Interactive API docs: `http://localhost:8787/docs`
- Health and model residency: `http://localhost:8787/health`

To use the studio from another device on the home network, open port 8787 using
the server computer's LAN address. Do not expose the unauthenticated service to
the public internet.

## Generate speech

```bash
curl -X POST http://127.0.0.1:8787/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hey. You made it. I was hoping I would get a minute with you.",
    "voice": "cinematic_ai_companion_f",
    "speech_mode": "reference",
    "fidelity_mode": "exact",
    "candidate_pool_size": 2,
    "generation_padding_seconds": 0.1,
    "audio_ref_keep": 1.0,
    "emotion": "tender",
    "instructions": "natural intimate cinematic acting with clear diction",
    "response_format": "flac"
  }' --output speech.flac
```

`speech_mode: "reference"` is the production default. It reuses the selected
or uploaded voice anchor on every chunk. Exact mode checks the complete raw
transcript before alignment: a take containing the requested sentence plus a
nonsense prefix or suffix is rejected and retried, never cropped into a false
success. Model-native renders are labeled under
`ComfyUI/output/voice_api/raw_unverified/`; only gated job files are delivered.

The duration estimator gives H3 a tight speech window because unused Ref2VA
timeline is frequently filled with invented words. `generation_padding_seconds`
defaults to `0.10`, and frame counts snap to the nearest valid H3 duration.
Increase it only when a deliberately slow performance is being truncated.

`audio_ref_keep` defaults to `1.0`, the literal reference conditioning used in
training. Values around `0.97-0.99` can produce a fresher take while preserving
identity; lower values increasingly trade exact timbre for variation.

`speech_mode: "continuous"` is experimental. It uses FL2VA to establish an expressive performance,
then carries the actual prior audio latent into later chunks. Ten seconds of
voice memory produced the strongest measured cross-chunk identity in local
tests. A catalog persona is interpreted from its description in this mode; it
is not an exact clone of that persona's stored anchor.

The first continuous chunk is timestamp-aligned against the requested opening
words on CPU. If H3 emits a brief false start or an overlong introductory pause,
the API removes that pre-roll, retains a short natural breath before the first
word, and applies a tiny click-safe fade. Later chunks are left untouched so the
latent continuation seams remain exact. The amount removed is returned in the
`X-Voice-Leading-Trim-Seconds` response header and shown in the studio.

The continuous path remains opt-in because latent continuation can produce
garbled words near a handoff. Its final assembled file is still transcript- and
signal-checked before delivery.

Only requested speech is placed inside H3's dialogue tag. Voice descriptions,
acting notes, room directions, and metadata remain outside it and are explicitly
marked inaudible.

## Clone a permitted voice

```bash
curl -X POST http://127.0.0.1:8787/v1/voices \
  -F 'name=my_voice' \
  -F 'description=A warm natural adult voice with relaxed delivery.' \
  -F 'file=@/path/to/permitted-reference.mp4'
```

Audio and video are accepted. FFmpeg extracts up to 14 seconds, converts it to
32 kHz stereo FLAC, applies a high-pass filter, and normalizes loudness. Use only
voices you have permission to clone.

## Stream long speech

Ref2VA streaming returns signed 16-bit little-endian PCM at 32 kHz. Accepted
chunks are joined with 60 ms blends while later chunks continue rendering.
Continuous FL2VA currently returns after the complete dependent latent graph
finishes, because every later chunk depends on the previous sampler output.

Non-streaming speech is re-transcribed after joining and encoding. The API
rejects invented seam speech, remaining onset artifacts, clipping, and missed
target durations. Use the `X-H3-Job-Id` response header with
`GET /v1/audio/speech/jobs/{job_id}/report` to inspect every attempt and the
final quality receipt.

```bash
curl -N -X POST http://127.0.0.1:8787/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "A long passage goes here.",
    "voice": "warm_narrator",
    "stream": true,
    "channels": 2
  }' | ffplay -nodisp -autoexit -f s16le -ar 32000 -ac 2 -i pipe:0
```

## Generate music

List the complete catalog:

```bash
curl http://127.0.0.1:8787/v1/audio/music/presets
```

Generate a structured song:

```bash
curl -X POST http://127.0.0.1:8787/v1/audio/music \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "song",
    "title": "County Line",
    "preset": "rock_hard",
    "lyrics": "[Intro]\n\n[Verse]\nThe diner sign is buzzing red above the county line\nYou draw a little orbit in the sugar by your side\n\n[Chorus]\nMeet me where the midnight coffee shines\nI will tell you every bad idea of mine\n\n[Outro]",
    "duration_mode": "auto",
    "prompt_profile": "natural_song_sheet_v2",
    "response_format": "flac"
  }' --output county-line.flac
```

With `duration_mode: "auto"`, the API estimates a conservative length from
lyric words, complete phrases, section count, explicit BPM, and fast-flow cues
such as rap or punk. Set `duration_mode: "manual"` and `duration_seconds` to
force any supported 5–60 second length. The same planner is available without
running H3 at `POST /v1/audio/music/duration`.

Style, instrumentation, vocalist direction, and section labels cannot become
lyrics; only the plain text inside each populated section is vocalized.

## Generate SFX / Foley

List the SFX library:

```bash
curl http://127.0.0.1:8787/v1/audio/sfx/presets
```

Render a one-shot from a preset:

```bash
curl -X POST http://127.0.0.1:8787/v1/audio/sfx \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Door latch",
    "preset": "foley_door_open_close",
    "duration_mode": "auto",
    "intensity": 0.75,
    "response_format": "flac"
  }' --output door.flac
```

Or describe a custom effect:

```bash
curl -X POST http://127.0.0.1:8787/v1/audio/sfx \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Whoosh",
    "kind": "oneshot",
    "description": "A fast stereo fabric whoosh with a sharp center peak and quick decay",
    "space": "a dry Foley stage",
    "duration_seconds": 2.0,
    "duration_mode": "manual",
    "response_format": "wav"
  }' --output whoosh.wav
```

`kind` is `oneshot` (default) or `loop_bed` for continuous ambience. Duration
spans **1.5–30 seconds** and snaps to H3’s frame grid. `duration_mode: "auto"`
uses the preset default when a stock preset description is selected, otherwise
a kind-aware heuristic (`POST /v1/audio/sfx/duration` plans without running H3).
SFX prompts place the event in `overall_soundscape` and force
`non_diegetic_music: N/A`—they are not music instrumentals with the vocals
stripped.

Voice, music, and SFX are all available from the studio at `/` (`/#sfx` opens
the effects bay) as well as from the API. Short voice requests render as one
chunk; long text is split at natural punctuation and assembled automatically.
Structured music can run from 5 to 60 seconds as one continuous latent
performance rather than separately spliced clips. The default
`natural_song_sheet_v2` prompt keeps one stable shot and recording perspective,
allocates section timing from both word and phrase counts, and places each
musical section in one connected exact lyric block. `petrock_timed_v1` remains
available for reproducing the original successful Pet Rock shot-timing grammar.
For the strongest one-minute phrasing, use roughly 3–4 vocal sections and 8–14
complete lyric lines; the studio warns when a draft is substantially denser.

## ComfyUI music and SFX nodes

The repository is also a directly installable ComfyUI custom node. Clone it or
symlink it into `ComfyUI/custom_nodes`, then restart ComfyUI:

```bash
ln -s /absolute/path/to/minimax-h3-voice-api \
  /absolute/path/to/ComfyUI/custom_nodes/minimax-h3-voice-api
```

Add **MiniMax H3 → Music → H3 Music Studio** or **MiniMax H3 → SFX → H3 SFX
Studio** and connect the `audio` output to Preview Audio or Save Audio. Music
includes all 99 presets, song and instrumental modes, structured lyrics, singer
and arrangement direction, 5–60-second duration, 10–50 steps, 32/64/128
audio-first canvas sizes, sampler, scheduler, seed, and model controls. SFX
includes the Foley/SFX preset library, one-shot vs loop-bed kinds, intensity,
1.5–30-second duration, and the same audio-first graph. Both expand into native
H3 loader, conditioning, sampler, and audio decode nodes; they never call back
into the API. Auto duration uses the same planners as the HTTP service; Manual
leaves length under the workflow author's control.

A complete example is included at
`workflows/H3 Music Studio - 60 Second Song.json`.

## Transcribe audio or video

The transcription route accepts common audio and video containers through the
same multipart contract used by OpenAI clients. `whisper-1` is accepted as a
compatibility alias, but the response identifies the actual local Parakeet
model. The installed model is English-only and deterministic.

```bash
curl -X POST http://127.0.0.1:8787/v1/audio/transcriptions \
  -F 'file=@speech.flac' \
  -F 'model=nemo-parakeet-tdt-0.6b-v2-int8' \
  -F 'response_format=verbose_json'
```

Supported response formats are `json`, `text`, `verbose_json`, `srt`, and
`vtt`. Verbose JSON includes word timestamps and sentence-sized subtitle
segments. FFmpeg extracts mono 16 kHz audio from either audio or video before
Parakeet runs. Response headers report the model, CPU device, audio duration,
ASR time, and real-time factor.

## CPU transcription and speech verification

`onnx-asr` loads `nemo-parakeet-tdt-0.6b-v2` with
`CPUExecutionProvider`. It never claims CUDA memory. When
`H3_PARAKEET_MODEL` is unset, the library downloads and caches the int8 model on
first use. To reuse an existing model directory, set:

```bash
H3_PARAKEET_MODEL=/absolute/path/to/parakeet-int8
```

Set `"verify": false` for requests where transcript scoring is unwanted. Voice
anchor creation still uses verification to avoid permanently caching a garbled
persona.

## Configuration

All settings can be supplied through `.env` or real environment variables.

| Variable | Default |
|---|---|
| `H3_COMFY_ROOT` | sibling `../ComfyUI` when present |
| `H3_COMFY_URL` | `http://127.0.0.1:8188` |
| `H3_DATA_DIR` | `~/.local/share/minimax-h3-voice-api` |
| `H3_PARAKEET_MODEL` | onnx-asr managed cache |
| `H3_MAX_TRANSCRIPTION_MB` | `250` |
| `H3_MAX_TRANSCRIPTION_SECONDS` | `3600` |
| `H3_API_HOST` | `0.0.0.0` |
| `H3_API_PORT` | `8787` |
| `H3_MODEL_*` | filenames listed under Requirements |

Generated references, jobs, reports, and final media live under `H3_DATA_DIR`.
They are never part of the Python package or Git repository.

## Tests

Unit and API contract tests do not require a GPU or running ComfyUI:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check minimax_voice_api tests
```

With the live API running, the smoke tests generate short, medium, and long-form
examples, then send every result back through the public transcription route:

```bash
python tools/smoke_test.py --base-url http://127.0.0.1:8787
```

## Scope and limitations

H3 is a generative audiovisual model rather than a deterministic phoneme TTS
engine. CPU verification and retries substantially improve fidelity, but cannot
mathematically guarantee every word. H3 work is serialized for VRAM safety; the
model remains managed by ComfyUI and may move between RAM and VRAM under memory
pressure.

This project is an independent integration and is not affiliated with MiniMax,
OpenAI, ElevenLabs, or ComfyUI.
