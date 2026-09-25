# Purpose

Native Windows CLI for MiniMax-H3 video. Binary name `h3`.

# Ownership

- `src/main.rs` parses `h3 <subcommand>` and holds the GPU queue lock.
- `src/host/` owns config, process launch, worker checks, ffmpeg, path roots, and the queue lock.
- `src/video/h3/` owns install, setup, download, doctor, generate, continue, loop, shortfilm, edit, upscale, interpolate, face-refine, sprites, and refmod.
- `runtimes/minimax-h3/` is the engine (workers, ComfyUI, scripts). Python is `uv` in that tree. `.venv` is local and not committed.
- `comfy-workflows/` holds compiled H3 video packages plus `materialize.py`.
- `installer/h3.iss` is the Inno Setup script for the Windows release installer.
- `.github/workflows/release-windows.yml` builds that installer and publishes it.

# Local Contracts

- Weights stay external. Order: `GEMMY_H3_CHECKPOINTS`, then `%APPDATA%\h3\config.json` (`h3 setup`), then an existing sibling or models root, then `%USERPROFILE%\h3-models`.
- `h3 install` is the machine setup: install `uv` if missing, `uv sync` the runtime and `jev-sdk` when that project exists, find or install `ffmpeg`, download product node packs that are not in git (latent upscaler `40316cf`, FaceRefine `d8521d1`) into the active Comfy `custom_nodes`, and download weight sets `base`, `eros`, `latent`, and `face`. ffmpeg is taken from `GEMMY_FFMPEG`, `where`, the process PATH, the user and machine Path registry values, WinGet Links, and `WinGet\Packages\*\*\bin\ffmpeg.exe`. `winget install Gyan.FFmpeg` runs only when none of those exist. An already-installed ffmpeg is success even when winget returns "no applicable update". Files already at the listed size are skipped. `--dry-run` prints the plan and does not download. Missing required weights fail the command. A failed `h3 install` writes `install.log` next to `h3.exe`.
- Worker launches attach a Windows job so closing the parent kills the child. If that attach is refused because the parent is already in a job (the setup exe), the child keeps running.
- `h3 setup` saves checkpoints / Comfy / Eros / stock Ref2VA / Singularity roots, `uv sync` when Python is missing, and downloads those same product node packs into the active Comfy. `--comfy PATH` accepts a ComfyUI folder (`main.py`, or `ComfyUI\main.py` one level down), copies `run_h3_workflow.py`, `h3_workflow_build.py`, and shipped `custom_nodes` from the bundled pack, then downloads any missing product pack.
- `h3 download` fetches one or more sets: `base`, `vae-fp16`, `eros`, `eros-bf16`, `ref2va-stock`, `singularity`, `turbo`, `realism`, `vsa`, `latent`, `face`. `--list` and `--dry-run` do not download.
- Image-to-video requires `--first-frame`. Shortfilm `--phase video` takes a board already on disk. `h3 download` does not fetch Comfy.
- NVIDIA DLSS SDK binaries under the interpolate node stay on the machine. They are gitignored (one DLL is over GitHub's 100 MB limit).
- Do not commit `.env`, `ComfyUI/extra_model_paths.yaml`, or `ComfyUI/models`.
- Use `uv`, never `pip`. Native Windows first.
- `repo_root()` is `GEMMY_REPO_ROOT` when that variable is set, otherwise the directory that contains `h3.exe` when `runtimes/minimax-h3` is beside it, otherwise the compile-time crate path. `cargo install --path .` keeps the compile-time path. The Windows installer uses the install directory.
- `.github/workflows/release-windows.yml` builds `installer/h3.iss` on `windows-latest`. A `v*` tag publishes `h3-<version>-windows-x64-setup.exe` to the GitHub Release. `workflow_dispatch` uploads that same installer as an Actions artifact. The installer is per-user, adds `h3` to the user PATH, and runs `h3 install` before it finishes. It does not bundle `.venv`, weights, or the NVIDIA interpolate DLLs inside the exe; `h3 install` creates the venv and downloads weights onto the machine. The release job runs the install-layout test only. The full `cargo test` suite still expects local weight files.

# Work Guidance

- Do not vendor multi-GB checkpoints into this tree.

# Verification

- `cargo test`
- `h3 doctor` reports `comfy_root` under this repo's `runtimes\minimax-h3\ComfyUI`
- `h3 generate --mode ref2va --ref-image <still> --prompt "..." --dry-run-plan` shows Eros two-stage
- `h3 generate --prompt "..." --dry-run-plan` without `--first-frame` fails closed

# Child DOX Index

- `runtimes/minimax-h3/AGENTS.md` — engine, Comfy pack, and workers.
- `comfy-workflows/AGENTS.md` — compiled H3 workflow packages.
