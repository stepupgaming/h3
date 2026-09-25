# H3 Latent Upscale + Refine

3D latent upscale then a short H3 refine sample. Gemmy `video h3 upscale --backend latent`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Enlarge a persisted H3 AV latent and refine at the new size.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-latent-refine
```

## Required parameters

- `prompt`
- `context_latent`

## Optional parameters

- `unet` (default "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
- `clip` (default "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
- `video_vae` (default "minimax_h3_video_vae_fp16.safetensors")
- `audio_vae` (default "minimax_h3_audio_vae_fp32.safetensors")
- `width` (default 1280)
- `height` (default 720)
- `length` (default 124)
- `upscale_model` (default "minimax_h3_latent_upscaler_3d_fp16.safetensors")
- `seed` (default 42)
- `shift_video` (default 12)
- `shift_audio` (default 3)
- `latent_prefix` (default "h3/latent_refine_av")
- `output_prefix` (default "h3/latent_refine")
- `fingerprint` (default "{}")

## Custom node packs

- `gemmy-h3-context`
- `Comfyui_Minimax_h3_latent_Upscaler`

## Node classes

- `BasicGuider`
- `CLIPLoader`
- `ConditioningZeroOut`
- `CreateVideo`
- `GemmyH3JoinAV`
- `GemmyH3LoadAVLatent`
- `GemmyH3SaveAVLatent`
- `GemmyH3SplitAV`
- `KSamplerSelect`
- `ManualSigmas`
- `MiniMaxH3ImageToVideo`
- `MiniMaxH3SigmaShift`
- `MinimaxH3LatentUpscaler3D`
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
- model: `minimax_h3_latent_upscaler_3d_fp16.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-latent-refine --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-latent-refine --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
