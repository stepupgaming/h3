# H3 Masked Inpaint

H3 mask_edit continuation: encode crop video, apply mask, sample. Used after SAM3 track.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Inpaint a masked region of an H3 video.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-mask-edit
```

## Required parameters

- `crop_video`
- `crop_mask_video`
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
- `denoise` (default 0.65)
- `shift_video` (default 12)
- `shift_audio` (default 3)
- `latent_prefix` (default "h3/mask_edit_av")
- `output_prefix` (default "h3/mask_edit")
- `fingerprint` (default "{}")

## Custom node packs

- `gemmy-h3-context`

## Node classes

- `CLIPLoader`
- `ConditioningZeroOut`
- `CreateVideo`
- `GemmyH3EncodeVideoFrames`
- `GemmyH3JoinAV`
- `GemmyH3SaveAVLatent`
- `GemmyH3SetAVNoiseMask`
- `GemmyH3SplitAV`
- `GetVideoComponents`
- `ImageToMask`
- `KSampler`
- `LoadVideo`
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
cwf inspect @stepupgaming/comfy-workflow-h3-mask-edit --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-mask-edit --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
