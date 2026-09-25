# H3 First-Frame to Video-Audio

MiniMax-H3 image-to-video (first frame). Gemmy `video h3 generate --mode i2v`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Animate a first-frame still into video+audio with MiniMax-H3 FL2VA.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-fl2va
```

## Required parameters

- `first_image`
- `prompt`

## Optional parameters

- `unet` (default "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
- `clip` (default "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
- `video_vae` (default "minimax_h3_video_vae_fp16.safetensors")
- `audio_vae` (default "minimax_h3_audio_vae_fp32.safetensors")
- `width` (default 864)
- `height` (default 480)
- `length` (default 124)
- `seed` (default 42)
- `steps` (default 20)
- `denoise` (default 1)
- `shift_video` (default 12)
- `shift_audio` (default 3)
- `latent_prefix` (default "h3/fl2va_av")
- `output_prefix` (default "h3/fl2va")
- `fingerprint` (default "{}")

## Custom node packs

- `gemmy-h3-context`

## Node classes

- `CLIPLoader`
- `ConditioningZeroOut`
- `CreateVideo`
- `GemmyH3SaveAVLatent`
- `KSampler`
- `LoadImage`
- `MiniMaxH3ImageToVideo`
- `MiniMaxH3SigmaShift`
- `SaveVideo`
- `UNETLoader`
- `VAEDecode`
- `VAEDecodeAudio`
- `VAELoader`

## Models

- model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- model: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- model: `minimax_h3_video_vae_fp16.safetensors`
- model: `minimax_h3_audio_vae_fp32.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-fl2va --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-fl2va --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
