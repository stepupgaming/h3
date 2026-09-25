# H3 DLSS frame interpolate

Post-process a finished MiniMax-H3 MP4 with NVIDIA DLSS Frame Generation.

Gemmy command: `gemmy video h3 interpolate --input clip.mp4 --fps 48`.

Unofficial project. Not affiliated with or endorsed by Comfy Org or NVIDIA.

## Purpose

Raise a 24 fps H3 master to 48/60/120 without resampling the DiT. This is a
deliverable post. Do not feed the result back into `continue` / `loop` encode.
Pixel upscale stays `gemmy video h3 upscale --backend rtx` (RTX-VSR-Pro).
This graph does **not** call the pack's video/image Super Resolution nodes.

AUTHORING SOURCE: `ir.build.ts`.
GENERATED: `workflow.ir.json`, `comfy.workflow.json`, `prompt.template.json` — do not hand-edit.
Rebuild with `pnpm comfy:build` from `comfy-workflows/`.
Gemmy runtime binds `prompt.template.json` (Python, no Node).

`NvidiaDLSSFrameInterpolation` is `rawNode` until `environments/h3` is recaptured + codegen.

## Order

load video → NVIDIA DLSS Frame Interpolation → Save Video.

Target FPS may not exceed 6× the source rate.

## Runtime pin

`runtimes/minimax-h3/ComfyUI/custom_nodes/ComfyUI-NVIDIA-DLSS-Frame-Interpolation`
(`Konohamaru04/ComfyUI-NVIDIA-DLSS-Frame-Interpolation` @ `c755e27`, MIT node
code). NVIDIA DLSS SDK DLLs are **local Git LFS artifacts**, not committed.
Clone/pull with Git LFS, or doctor will report stub/missing binaries.

Windows + current NVIDIA driver. Hardware-accelerated GPU scheduling (HAGS) is
recommended for Frame Generation. FFmpeg/FFprobe via Gemmy (`GEMMY_FFMPEG` /
`DLSS_FFMPEG_PATH`).

## Required parameters

- `video` — staged filename under Comfy `input/`

## Optional parameters

- `output_fps` (default `"48"`; 23.976 / 25 / 29.97 / 30 / 48 / 50 / 59.94 / 60 / 90 / 120)
- `dlss_engine` (default `"Auto"`; `Native DLSSG` / `Cascade`)
- `encoding_quality` (default `"Max"`)
- `video_codec` (default `"H.264"`)
- `container` (default `"MP4"`)
- `filename_prefix` (default `gemmy/dlss/interpolate`)

## Custom node packs

- `ComfyUI-NVIDIA-DLSS-Frame-Interpolation` (`source: "manual"`)

## Node classes

- `LoadVideo`
- `NvidiaDLSSFrameInterpolation`
- `SaveVideo`

## Models

None. This graph does not load a DiT. NVIDIA runtimes ship beside the pin.

## License

MIT for this package's Graph IR / scaffolding, owned by Gemmy. Node Python is
MIT (Konohamaru04 / Merserk DLSS 5 Visual Enhancer). NVIDIA DLSS SDK, RenoDX,
and ReShade notices sit beside the binaries and are **not** redistributed by
this repo. NVIDIA's SDK includes a commercial-release notification requirement.

Never hand-edit generated workflow JSON.
