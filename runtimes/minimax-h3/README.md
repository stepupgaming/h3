# MiniMax-H3 (Gemmy runtime)

Internalized production code for `gemmy video h3`.

- Package: `minimax_h3/`
- Worker: `gemmy_h3_generate.py`
- Scripts: `scripts/h3_*.py`
- Official VAE modules (code only): `MiniMax-H3-Official/FL2VA/`
- Weights: external — see `docs/MODEL_LOCATIONS.md` and `GEMMY_H3_CHECKPOINTS`

Product CLI:
- `gemmy video h3 generate` — `i2v` (default, auto Krea), `t2va`, `fl2va`, `ref2va`, optional `--allow-keyframe-refs`
- `gemmy video h3 edit` — SAM3.1 track + Eros crop sample + uncrop (`gemmy_h3_comfy_edit.py`)
- `gemmy video h3 continue` — multi-window FL2VA stitch (`scripts/h3_fl2va_continue.py`)
- `gemmy video h3 upscale` — post-decode SR (`scripts/h3_upscale.py`; RTX VSR when `nvidia-vfx` installed)

Setup: `gemmy video h3 install` (uv sync). Do not commit `.venv` or multi-GB checkpoints.
