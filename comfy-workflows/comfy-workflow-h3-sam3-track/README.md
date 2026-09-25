# SAM3 Video Track to Mask

SAM3.1 video track → mask MP4. First stage of Gemmy `video h3 edit`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Track an object in video and emit a mask clip.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-sam3-track
```

## Required parameters

- `video`
- `prompt`

## Optional parameters

- `checkpoint` (default "sam3.1_multiplex_fp16.safetensors")
- `output_prefix` (default "h3/sam3")

## Custom node packs

- (core ComfyUI classes only, or mapping unknown)

## Node classes

- `CLIPTextEncode`
- `CheckpointLoaderSimple`
- `CreateVideo`
- `GetVideoComponents`
- `LoadVideo`
- `MaskToImage`
- `SAM3_TrackToMask`
- `SAM3_VideoTrack`
- `SaveVideo`

## Models

- checkpoint: `sam3.1_multiplex_fp16.safetensors`

Do not publish model binaries. Filenames are defaults from Gemmy's H3/Krea layout; they are not universally available.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-sam3-track --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-sam3-track --url http://127.0.0.1:8188 --param prompt=...
```

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Models and third-party custom nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
