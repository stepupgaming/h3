# H3 Latent Upscale Preview

3D latent upscale then decode only (no second H3 sample).

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Preview a latent upscale without a refine pass.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-latent-preview
```

## Required parameters

- `context_latent`

## Optional parameters

- `upscale_model` (default "minimax_h3_latent_upscaler_3d_fp16.safetensors")
- `scale` (default 2)
- `video_vae` (default "minimax_h3_video_vae_fp16.safetensors")
- `audio_vae` (default "minimax_h3_audio_vae_fp32.safetensors")
- `output_prefix` (default "h3/latent_up")
- `latent_prefix` (default "h3/latent_up")
- `fingerprint` (default "{}")

## Custom node packs

- `gemmy-h3-context`
- `Comfyui_Minimax_h3_latent_Upscaler`

## Node classes

- `CreateVideo`
- `GemmyH3JoinAV`
- `GemmyH3LoadAVLatent`
- `GemmyH3SaveAVLatent`
- `GemmyH3SplitAV`
- `MinimaxH3LatentUpscaler3D`
- `SaveVideo`
- `VAEDecode`
- `VAEDecodeAudio`
- `VAELoader`

## Models

- model: `minimax_h3_latent_upscaler_3d_fp16.safetensors`
- model: `minimax_h3_video_vae_fp16.safetensors`
- model: `minimax_h3_audio_vae_fp32.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-latent-preview --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-latent-preview --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
