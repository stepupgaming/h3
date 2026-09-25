# h3

Windows CLI for MiniMax-H3 video.

## Install

Download the Windows setup exe from [GitHub Releases](https://github.com/stepupgaming/h3/releases). The file is `h3-<version>-windows-x64-setup.exe`.

The installer puts `h3` on the current user's PATH and runs `h3 install`. That step creates the Python environment, installs `uv` and `ffmpeg` when they are missing, downloads the Comfy node packs, and downloads the model weights. The weight download is large. A console window stays open until it finishes.

Open a new terminal after the installer closes.

```
h3 doctor
h3 generate --first-frame still.png --prompt "camera slowly pushes in" --duration 5 -o clip.mp4
```

`h3 doctor` prints the checkpoints folder. On a new machine the weights are in `%USERPROFILE%\h3-models`. To put them on another drive:

```
h3 setup --checkpoints D:\h3-models
h3 install
```

The second `h3 install` skips files that are already the right size.

Install an NVIDIA driver yourself. The setup exe does not include one.

## Make a clip

Image-to-video takes a still you already have. Pass it with `--first-frame`. Default length is 5 seconds. Default size is 480p, 864×480.

```
h3 generate --first-frame still.png --prompt "camera slowly pushes in" --duration 5 -o clip.mp4
h3 generate --canvas 720p --first-frame still.png --prompt "camera slowly pushes in" --duration 5 -o clip.mp4
```

Text only:

```
h3 generate --mode t2va --prompt "abstract aurora over a lake" --duration 5 -o clip.mp4
```

First frame and last frame:

```
h3 generate --mode fl2va --first-frame start.png --last-frame end.png --prompt "she turns toward the window" --duration 5 -o clip.mp4
```

A reference image uses the Eros weights from `h3 install`:

```
h3 generate --mode ref2va --ref-image subject.png --prompt "walks through a neon market" --duration 5 -o clip.mp4
```

`h3 generate --help` lists the rest of the flags.

## After the clip

```
h3 continue --input clip.mp4 --prompt "the camera keeps pushing in" --num-windows 2 -o longer.mp4
h3 upscale --input clip.mp4 --canvas 1080p -o clip_1080p.mp4
h3 upscale --input clip.mp4 --backend latent --prompt "the same shot, sharper" --canvas 720p -o clip_720p.mp4
h3 face-refine --input clip.mp4 -o clip_face.mp4
h3 interpolate --input clip.mp4 --fps 48 -o clip_48.mp4
```

`continue` extends the same shot. `upscale --canvas 1080p` enlarges a finished video. `--backend latent` resamples at a larger size and needs the `.h3av.safetensors` file next to the clip. `face-refine` is a pass on a finished mp4. `interpolate` raises the frame rate, 24 fps to 48 fps by default. `h3 interpolate --check` reports whether the NVIDIA frame-generation runtime is on the machine.

## Commands

```
h3 install       Python env, ffmpeg, node packs, and the base, eros, latent, and face weights
h3 doctor        CUDA, ffmpeg, Comfy, and the weight paths
h3 setup         checkpoints folder, or an existing ComfyUI folder
h3 download      an optional weight set
h3 verify        required files are present
h3 generate      one clip
h3 continue      longer version of the same shot
h3 loop          several scenes from a plan file
h3 shortfilm     storyboard, or one still per scene
h3 edit          replace one region in an existing clip
h3 upscale       larger frame
h3 interpolate   higher frame rate
h3 face-refine   face pass on a finished clip
h3 sprites       sprite sheet or GIF from a clip
h3 refmod        save, list, or inspect a reference latent
```

`h3 <command> --help` is the flag list for that command.

`h3 install` downloads four weight sets.

- `base` for image-to-video, text-to-video, and first-plus-last
- `eros` for reference-to-video and shortfilm
- `latent` for `h3 upscale --backend latent`
- `face` for `h3 face-refine`

Other sets are separate downloads. `h3 download --list` prints them: `vae-fp16`, `eros-bf16`, `ref2va-stock`, `singularity`, `turbo`, `realism`, `vsa`. `h3 download --dry-run singularity` prints URLs and destinations and does not download.

## Use a ComfyUI folder you already have

The installer uses the Comfy pack shipped with h3. Another folder works when it contains `main.py`, or `ComfyUI\main.py` one level down.

```
h3 setup --comfy C:\path\to\ComfyUI
```

That copies the H3 runner and the shipped nodes into that folder, then downloads any product node pack that is missing. Each job starts its own Comfy process.

## Build from a clone

The setup exe is the normal install. A clone needs Rust and an NVIDIA driver.

```
git clone https://github.com/stepupgaming/h3.git
cd h3
cargo install --path .
h3 install
h3 doctor
```

`cargo install` puts `h3.exe` on PATH. The engine stays in the clone, at the path recorded when you compiled. Leave that folder where it is.
