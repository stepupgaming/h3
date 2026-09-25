# MiniMax H3 Comfy setup (Gemmy internalized product pack)

This directory is `runtimes/minimax-h3/ComfyUI`. Deleting external research checkouts must not break Gemmy.

Weights: external via `GEMMY_H3_CHECKPOINTS`. Code: this tree.
Dev override only: `GEMMY_H3_COMFY`.

## Comfy core pin
- Product pack: **ComfyUI v0.36.0** (tag; native H3 VAE + AddGuide + chunked I/O). Do not full-pull master (SaveVideo DynamicCombo still needs the Gemmy wrapper).
- Windows: `comfy/utils.py` `--disable-mmap` uses safetensors `backend=pread`. Re-apply after overlays.
- Pins stay in `h3_custom_nodes_pins.json`; never overwrite `custom_nodes/`.
- Kitchen: `comfy-kitchen==0.2.34`. AIMDO pin `0.5.3` stays disabled.

## LoRAs (under checkpoints/loras/)
| File | Flag | Notes |
|---|---|---|
| `minimax_h3_turbo_v4_step600_ema.safetensors` | `--turbo v4` | Larryvrh node + Turbo Sampler; strength 1.0 |
| `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | `--turbo ema_wan` / v1 | lab fallback |
| `h3-realism-people-t2v-i2v-r2v.safetensors` | `--realism` | fal People style; `LoraLoaderModelOnly` after turbo; trigger `r34l1sm`; strength 1.0 (0.6–0.8 lighter) |
