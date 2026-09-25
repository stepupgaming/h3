# h3

Windows CLI for MiniMax-H3 video.

## Windows installer

GitHub Releases publishes `h3-<version>-windows-x64-setup.exe`. It installs `h3` for the current user, adds it to that user's PATH, and then runs `h3 install`. That step installs `uv` and `ffmpeg` if they are missing, creates the Python environment, downloads the product Comfy node packs, and downloads the `base`, `eros`, `latent`, and `face` weight sets. The console stays open while those files download. Open a new terminal, then:

```
h3 doctor
h3 generate --first-frame still.png --prompt "..." --duration 5 -o clip.mp4
```

If you already have ComfyUI, point h3 at that folder. Setup copies the H3 runner and shipped nodes into it and downloads any product pack that is missing:

```
h3 setup --comfy C:\path\to\ComfyUI
```

## Build from the clone

Clone this repo and build `h3` from that folder. The binary looks for the engine inside the clone, so leave the folder where you compiled it. If you move the clone, set `GEMMY_REPO_ROOT` to the new path.

## What you install first

- Rust (`cargo`), if you are building from the clone
- an NVIDIA driver

`h3 install` installs `uv` and `ffmpeg` when they are not already on the machine. The Comfy pack is in `runtimes\minimax-h3\ComfyUI`. `h3 install` fills in the Python environment, the latent-upscaler and face-refine node packs, and the weight files.

## First clip

```
cargo install --path .
h3 install
h3 doctor
h3 generate --first-frame still.png --prompt "..." --duration 5 -o clip.mp4
```

`h3 setup` writes `%APPDATA%\h3\config.json` when you choose a checkpoints folder or point at an existing ComfyUI.

`h3 install` downloads `base` (image-to-video and text-to-video), `eros` (reference-to-video), `latent` (`h3 upscale --backend latent`), and `face` (`h3 face-refine`). Other sets stay opt-in: `vae-fp16`, `eros-bf16`, `ref2va-stock`, `singularity`, `turbo`, `realism`, `vsa`. `h3 download --list` prints them. `h3 download --dry-run base` prints URLs and destinations. A gated Hugging Face repo needs `HF_TOKEN`. Running `h3 install` again skips files that are already the right size.

Image-to-video takes `--first-frame`. Text-to-video is `h3 generate --mode t2va`.

## Comfy

Use the pack in this repo, or point at a ComfyUI folder you already have:

```
h3 setup --comfy C:\path\to\ComfyUI
```

That copies the H3 runner and the shipped node packs into that folder and downloads the latent-upscaler and face-refine packs if they are not there. This CLI does not attach to a Comfy process you already have open.

## Commands

```
h3 setup | download | install | verify | doctor
h3 generate | continue | loop | shortfilm | edit
h3 upscale | interpolate | face-refine | sprites | refmod
```
