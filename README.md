# h3

Windows CLI for MiniMax-H3 video.

Clone this repo and build `h3` from that folder. The binary looks for the engine inside the clone, so leave the folder where you compiled it. If you move the clone, set `GEMMY_REPO_ROOT` to the new path.

## What you install first

- Rust (`cargo`)
- `uv` on `PATH`
- `ffmpeg` on `PATH`
- an NVIDIA driver

The Comfy pack is already in `runtimes\minimax-h3\ComfyUI`. Setup does not download Comfy. Download does not install Comfy.

## First clip

```
cargo install --path .
h3 setup --checkpoints D:\h3-models
h3 download --list
h3 download base
h3 download eros
h3 doctor
h3 generate --first-frame still.png --prompt "..." --duration 5 -o clip.mp4
```

`h3 setup` writes `%APPDATA%\h3\config.json` and syncs the Python env with `uv` when it is missing.

`base` is the image-to-video and text-to-video set. `eros` is the default reference-to-video model. Other sets are optional: `vae-fp16`, `eros-bf16`, `ref2va-stock`, `singularity`, `turbo`, `realism`, `vsa`. `h3 download --dry-run base` prints URLs and destinations. A gated Hugging Face repo needs `HF_TOKEN`.

Image-to-video takes `--first-frame`. Text-to-video is `h3 generate --mode t2va`.

## Comfy

Use the pack in this repo. `h3 setup --comfy` accepts another folder only when that folder already has `run_h3_workflow.py` and `custom_nodes\gemmy-h3-context`. A stock Comfy Portable install does not run these graphs, and this CLI does not attach to a Comfy process you already have open.

## Commands

```
h3 setup | download | install | verify | doctor
h3 generate | continue | loop | shortfilm | edit
h3 upscale | interpolate | face-refine | sprites | refmod
```
