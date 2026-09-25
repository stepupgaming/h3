# H3 PixelForge sprites

Post-process a MiniMax-H3 clip into a keyed sprite loop at the drawing's own size.

Gemmy command: `gemmy video h3 sprites --input clip.mp4 --loop --atlas`.

Unofficial project. Not affiliated with or endorsed by Comfy Org.

## Purpose

Turn an H3 idle-pin / action clip into keyed PNG frames, a sheet, a GIF, and
optional `atlas.json`. Generation stays on `gemmy image` + `gemmy video h3`.
This package is the forge graph only.

Product look is the **cel drawing**, not 8-bit crush. Quantize on the full
864-wide chroma plate is not in this graph: it treats empty backdrop as canvas
and turns a ~100px character into ~15px mush. Optional 8-bit is a later,
subject-sized pass — not the default.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

PixelForge classes are `rawNode` until `environments/h3` is recaptured + codegen.

## Order (binding)

load video → decimate → chroma key (full res) → union crop / bottom-center →
loop trim → frame dedup → sheet + PNG sequence + GIF.

Do not quantize the full plate before crop.

## Install

```sh
pnpm add @stepupgaming/comfy-workflow-h3-pixelforge
```

Runtime pin (not auto-installed by this package):

`runtimes/minimax-h3/ComfyUI/custom_nodes/ComfyUI-PixelForge-H3`
(`japaneserunic/blubs-pixel-nodepack` @ `93a05eb`, MIT).

## Required parameters

- `video` — staged filename under Comfy `input/`

## Optional parameters

- `every_nth` (default 2)
- `start_offset` (default 0)
- `max_frames` (default 0 = no cap)
- `key_color` (default `"auto"`; `#00FF00` / `#FF00FF` when auto misses)
- `loop_mode` (`off` | `auto` | `pingpong`, default `off`)
- `filename_prefix` (default `gemmy/pixelforge/sprite`)
- `sheet_prefix` (default `gemmy/pixelforge/sheet`)
- `gif_prefix` (default `gemmy/pixelforge/gif`)
- `webp_prefix` (default `gemmy/pixelforge/preview`)
- `tag_name` (default `run`)
- `fps` (default 12)

## Custom node packs

- `ComfyUI-PixelForge-H3` (`source: "manual"`)

## Node classes

- `LoadVideo`
- `PixelForgeVideoToFrames`
- `PixelForgeChromaKey`
- `PixelForgeAutoCrop`
- `PixelForgeLoopTrim`
- `PixelForgeFrameDedup`
- `PixelForgeSheetPack`
- `PixelForgeAsepriteExport`
- `PixelForgeSaveGIF`
- `SaveImage`
- `SaveAnimatedWEBP`

## Models

None. This graph does not load a DiT.

## Inspect & run

```sh
cwf inspect @stepupgaming/comfy-workflow-h3-pixelforge --url http://127.0.0.1:8188
cwf run @stepupgaming/comfy-workflow-h3-pixelforge --url http://127.0.0.1:8188 --param video=clip.mp4
```

Product path is `gemmy video h3 sprites`, which materializes this package and
copies keyed frames / sheet / GIF out of Comfy `output/`.

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. PixelForge
(`japaneserunic/blubs-pixel-nodepack`) is MIT. Models and third-party custom
nodes remain under their own licenses.

Never hand-edit generated workflow JSON.
