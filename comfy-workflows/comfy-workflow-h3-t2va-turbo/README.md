# H3 Text to Video-Audio (Turbo)

MiniMax-H3 T2VA with Turbo LoRA + TurboSampler. Distinct topology from the 20-step euler path.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Faster H3 T2VA using the turbo sampler stack.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-t2va-turbo
```

## Required parameters

- `prompt`

## Optional parameters

- `unet` (default "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
- `clip` (default "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
- `video_vae` (default "minimax_h3_video_vae_fp16.safetensors")
- `audio_vae` (default "minimax_h3_audio_vae_fp32.safetensors")
- `turbo_lora` (default "minimax_h3_turbo_v4_step600_ema.safetensors")
- `width` (default 864)
- `height` (default 480)
- `length` (default 124)
- `seed` (default 42)
- `steps` (default 4)
- `denoise` (default 1)
- `shift_video` (default 12)
- `shift_audio` (default 3)
- `latent_prefix` (default "h3/t2va_turbo_av")
- `output_prefix` (default "h3/t2va_turbo")
- `fingerprint` (default "{}")

## Custom node packs

- `gemmy-h3-context`
- `ComfyUI-MiniMax-H3-Turbo`

## Node classes

- `BasicGuider`
- `BasicScheduler`
- `CLIPLoader`
- `ConditioningZeroOut`
- `CreateVideo`
- `GemmyH3SaveAVLatent`
- `MiniMaxH3ImageToVideo`
- `MiniMaxH3SigmaShift`
- `MiniMaxH3TurboLoRA`
- `MiniMaxH3TurboSampler`
- `RandomNoise`
- `SamplerCustomAdvanced`
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
- model: `minimax_h3_turbo_v4_step600_ema.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-t2va-turbo --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-t2va-turbo --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
