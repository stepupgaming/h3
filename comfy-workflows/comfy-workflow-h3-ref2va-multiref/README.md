# Comfy Workflow H3 Ref2va Multiref

Imported ComfyUI workflow packaged as @stepupgaming/comfy-workflow-h3-ref2va-multiref. Verify license and redistribution rights before publishing.

## Source

Imported from a ComfyUI workflow JSON. Edit `workflow.ts` if you want a typed authoring surface; `workflow.ir.json` remains the canonical payload.

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-ref2va-multiref
```

## Required parameters

- (none — this package is a concrete graph)

## Optional parameters

- (none)

## Node requirements

- CLIPLoader
- ConditioningZeroOut
- CreateVideo
- GemmyH3SaveAVLatent
- KSampler
- LoadImage
- MiniMaxH3ReferenceToVideo
- MiniMaxH3SigmaShift
- SaveVideo
- UNETLoader
- VAEDecode
- VAEDecodeAudio
- VAELoader

## Model requirements

- model: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- model: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- model: `minimax_h3_video_vae_fp16.safetensors`
- model: `minimax_h3_audio_vae_fp32.safetensors`

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-ref2va-multiref --url http://127.0.0.1:8188

cwf run @stepupgaming/comfy-workflow-h3-ref2va-multiref --url http://127.0.0.1:8188
```

Or from this directory after `cwf pack`:

```sh
cwf inspect .
cwf run . --url http://127.0.0.1:8188
```

## License / redistribution

This package was generated from an existing ComfyUI workflow. Importing a workflow does **not** grant redistribution rights to any models, custom nodes, images, or the workflow itself. Verify the license of every asset and of the source workflow before you `npm publish`.

The generated `package.json` defaults to MIT for the *package scaffolding only* — change it if that does not match the source.

## Docs

- [Convert a ComfyUI workflow into a package](https://stepupgaming.github.io/comfy-workflows/guide/convert-workflow)
- [Workflow packages](https://stepupgaming.github.io/comfy-workflows/guide/packages)

## Git (optional)

```sh
git init
git add .
git commit -m "Initial workflow package"
```

GitHub hosting is optional. npm is the package transport.

Never hand-edit generated workflow JSON.
