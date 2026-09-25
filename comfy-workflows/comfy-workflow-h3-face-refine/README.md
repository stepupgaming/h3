# H3 Face Refine

Track/crop faces, H3 denoise 0.35, stitch back. Gemmy `video h3 face-refine`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Sharpen faces in an H3 video without rewriting the whole clip.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-face-refine
```

## Required parameters

- `video`
- `prompt`

## Optional parameters

- `detector` (default "face_yolov8m.pt")
- `unet` (default "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
- `clip` (default "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
- `video_vae` (default "minimax_h3_video_vae_fp16.safetensors")
- `length` (default 124)
- `seed` (default 42)
- `steps` (default 8)
- `denoise` (default 0.35)
- `output_prefix` (default "h3/face")

## Custom node packs

- `ComfyUI-H3-FaceRefine`

## Node classes

- `CLIPLoader`
- `ConditioningZeroOut`
- `CreateVideo`
- `GetVideoComponents`
- `H3FaceStitch`
- `H3FaceTrackCrop`
- `H3InjectVideoLatent`
- `KSampler`
- `LoadVideo`
- `MiniMaxH3ImageToVideo`
- `MiniMaxH3SigmaShift`
- `SaveVideo`
- `UNETLoader`
- `VAEDecode`
- `VAELoader`

## Models

- model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- model: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- model: `minimax_h3_video_vae_fp16.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-face-refine --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-face-refine --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
