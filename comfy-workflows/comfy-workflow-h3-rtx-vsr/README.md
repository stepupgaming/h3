# NVIDIA RTX Video Super-Resolution

LoadVideo → RTXVideoSuperResolution → SaveVideo. Gemmy `video h3 upscale --backend rtx`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Pixel-space video super-resolution via NVIDIA RTX VSR Pro.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-rtx-vsr
```

## Required parameters

- `video`

## Optional parameters

- `quality` (default "HIGH")
- `scale` (default 2)
- `output_prefix` (default "h3/rtx_vsr")

## Custom node packs

- `ComfyUI-NVIDIA-RTX-VSR-Pro`

## Node classes

- `CreateVideo`
- `GetVideoComponents`
- `LoadVideo`
- `RTXVideoSuperResolution`
- `SaveVideo`

## Models

- (parameterized)

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-rtx-vsr --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-rtx-vsr --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
