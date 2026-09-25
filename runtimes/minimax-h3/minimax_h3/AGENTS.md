# minimax_h3/ — PyTorch reference

## Purpose

Standalone PyTorch 1:1 mirror of `src/`, independent of ComfyUI. Readable reference runtime for parity, audits, debugging, and the **owner-box production sample path** (streamed pruned int8 + SageAttention).

## Ownership

Owns the `minimax_h3` Python package: model modules, CLI (`cli.py` / `__main__.py`), and `tests/`.

Does not own: Candle runtime, repo-level parity driver scripts under `scripts/`, or checkpoint blobs.

## Local Contracts

| Module | Mirrors |
|---|---|
| `config.py` | `src/config.rs` |
| `layout.py` | `src/layout.rs` |
| `rotary.py` | `src/rotary.rs` |
| `timestep.py` | `src/timestep.rs` |
| `attention.py` | `src/attention.rs` |
| `blocks.py` | `src/blocks.rs` |
| `weights.py` | `src/weights.rs` |
| `model.py` | `src/model.rs` (+ Python streaming seams) |
| `cli.py` | `src/main.rs` (shared commands + Python `sample`) |
| `canvas.py` | Python-only canvas / resolution presets (Base ×32, short-edge 768, MP ladders) |
| `frame_grid.py` | Python-only 17k+5 snap, latent_t/audio_t, `flow_sigmas` (shared by sample/continue/encode) |
| `conditioning.py` | Python-only FL2VA/Ref2VA loaders + `build_conditioning` → `Payload` |
| `sample.py` | Python-only production sample pipeline (`SampleRequest` / `run_sample`) |
| `prompt_ir.py` | Python-only open IR-format prompt compiler (Context-IR stand-in L1; no Rust twin) |
| `profile.py` | Python-only phase profiler (no Rust twin required) |
| `int8_gemm.py` | Python-only Triton bf16×int8 experiment (opt-in; slower than cast on sm120) |

- Public exports live in `__init__.py`: math types **and** production seam (`run_sample`, `SampleRequest`, `build_conditioning`, `flow_sigmas`, `snap_frames`, …).
- **Gemmy / external product integration:** call `minimax_h3.run_sample(SampleRequest(...))` or CLI `python -m minimax_h3 sample …`. TE/VAE/continue stay under `scripts/` (subprocess VRAM sequencing) until package adapters exist.
- Load-time chain matches Rust: safetensors/random → optional `QuantizingSource` → optional `AdalnSvdSource`.
- **Streaming is real on the sample path:** `stream_gpu` moves one DiT/refiner block at a time (RAM↔VRAM only; no disk weight I/O). Deferred ConvRot int8 stays packed until device forward.
- **Deferred weight flavor (auto from checkpoint):**
  - **int8_convrot (default production):** `DeferredInt8` — packed I8 + per-row f32 scale; forward folds ConvRot Hadamard into the activation (`cast` backend).
  - DiT NVFP4 experiment removed (slower than int8 with eager dequant). TE NVFP4 lives only in `scripts/h3_text_encode.py`.
- **Lossless speed defaults (sample / CUDA):**
  - `--prefetch 1` — async H2D of block `i+1` overlapped with compute of block `i` (pinned host storages via `pin_stream_hosts`)
  - Stream free path: finished blocks leave via CUDA `event.query()` when possible (no host sync on the hot path); only block-wait if free backlog grows. Do not host-synchronize every block before compute.
  - `--precompute-adaln` — per-step AdaLN mod table; adaln weights not re-streamed per block
  - `--preproject-text` — `MiniMaxH3Model.prepare_text_states` runs condition_proj + token refiner once; denoise loop passes `[L, hidden]`
  - In-place / single-buffer `mod_scale_shift` / `mod_gate`; SwiGLU silu inplace on gate half
  - AdaLN precompute casts mods to stream dtype once/step; AdaLN linears stay on compute device (no per-step CPU bounce); `mod_gate` mutates residual in place
- Disable with `--prefetch 0` / `--no-precompute-adaln` / `--no-preproject-text` for A/B or legacy behavior.
- **Owner stream A/B (864×480, 39f, sage auto, steady s/step):** after free-path fix, `--prefetch` 1/2/3/5 are ~tied (~5.5 s/step); depth>1 only raises peak VRAM (~1.8→3.2 GB). Pin 0→12 GB: ~5.57→5.39 s/step (~3%) — not worth raising default pin. Step is compute-bound (attn+mlp). Keep **`--prefetch 1`**, **`--max-pin-gb 4`**. Microbench: `scripts/bench_stream_prefetch.py`.
- Deferred int8 default remains **`cast`**. Owner sm120 (5060 Ti) microbenches: `int_mm` slower+drift; `torch._weight_int8pack_mm` ~100× slower; Triton bf16×int8 ~0.55× cast; hand custom W8A16 CUDA tried and removed (~5–10× slower). Hadamard already tiny (~0.3–0.7 ms vs 5–14 ms GEMM). Cast time is ~90% cuBLAS GEMM / ~10% int8→bf16 inflate — naive custom kernels cannot beat cuBLAS. Do not flip default. Sage `auto`≡`fp8pp`.
- Host-pin budget (sample / CUDA prefetch): `--max-pin-gb` default **4** (only lock a small runway of weight bytes) + `--ram-reserve-gb` default **8** (stop if free RAM would drop below this). Pinning the full ~20 GB model is what drives Windows "Memory in use" to ~99% on a 64 GB box — do not remove the cap for normal use. `--max-pin-gb 0` = no pin; negative = reserve-only cap. Remaining weights stay pageable; quality unchanged.
- Deferred int8 Linear default backend is **`cast`** (ephemeral int8→bf16 inflate). Opt-in backends via `Linear.deferred_int8_backend`: `int_mm` / `triton` / `pack` — measure with `scripts/bench_int8_linear.py` and `minimax_h3/int8_gemm.py`.
- `--profile` / `--profile-json` emit phase timers (pack_embed, h2d*, adaln, attn, mlp, d2h, final) + peak VRAM. `h2d_enqueue` / free bookkeeping use host timing (`sync=False`) so measuring does not destroy transfer/compute overlap; attn/mlp stay device-sync timed.
- Attention CLI (`--attn`, default **`sage`** = production fast/approx):
  - **`sage`** — SageAttention SA2 (default). Approximate on fp8 backends.
  - **`fa2`** / `flash-attn` / `flash_attn` — real FlashAttention-2 HQ (`flash_attn` package). Preferred lossless-class path when the wheel is present.
  - **`flash`** — portable in-repo chunked online-softmax (Candle parity / fallback). **Not** Dao FA.
  - `sdpa` / `eager` / `auto` — debug / short-seq (SDPA may OOM at long packed seq on 16 GB).
- Windows pins: SA2 sageattention wheel + community **flash-attn 2.8.3 cu130 cp312** wheel (same family as gemmy `locate-anything-la-flash`). Official Dao/PyPI has no Windows FA wheels; torch 2.13+cu130 build has no built-in `FLASH_ATTENTION` kernel. Owner A/B: `outputs/sage_onoff_ab/` (sage vs portable flash) and `outputs/sage_fa2_ab/` (sage vs fa2).
- Dense Sage sub-backends via `--sage-backend` (default `auto`): `auto` / `fp8` / `fp8pp` / `fp16_cuda` / `fp16_triton` / `sa1`. Process-wide `set_sage_backend`. On sm120 `auto` ≡ fp8++ path. `fp16_cuda` is **refused on sm120+** (owner A/B: huge drift). Microbench: `scripts/bench_sage_backends.py`.
- Sample default remains **T2VA** (empty payload). Cond surfaces on `sample` (mutually exclusive modes by default):
  - **FL2VA:** `--first-frame PATH` / `--last-frame PATH` — DiT-normalized keyframe latents `[1,24,1,lat_h,lat_w]` (keys `video_latent` / `first_frame_latent` / `last_frame_latent`). Sets `Payload.keyframes` + `cond_video_latents` + `frame_count`; COND rows at `visual_cond_noise_aug=0.999`.
  - **Ref2VA:** `--ref-image` / `--ref-video` / `--ref-audio` / `--ref-av VIDEO,AUDIO` (repeatable; order preserved). Caps match official: ≤9 images, ≤3 videos, ≤3 audios, ≤12 files; audio cannot be sole input. Latents pre-encoded; sample never loads VAE. Packs `Payload.refs` + `cond_video_latents` / `cond_audio_latents` (VIDEO_AUDIO emits audio rows then video rows — push audio latent before video latent). Use **Ref2VA** checkpoint (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`).
  - **Experimental Multishot DiT memory stand-in:** `--allow-keyframe-refs` packs FL2VA keyframe(s) + `--ref-image` together. Prefer Ref2VA weights. Driven by `h3_fl2va_continue.py --dit-memory-refs`. Primary Multishot path is TE vision memory (see scripts AGENTS), not DiT refs.
  - Encode all media with `scripts/h3_vae_encode.py` (`--mode image|video|audio|auto`) at the sample canvas before sample.
- **`PackedLayout.signature`** includes keyframe count, `frame_count`, and a refs fingerprint (not only text/latent/audio dims). Rope cache keys on this — T2VA vs FL2VA/Ref2VA must not collide (regression tests in `tests/test_model.py`).
- **`MiniMaxH3Model.layout` cache** uses the same refs fingerprint as `PackedLayout.signature` (not `len(refs)` alone).
- **T2VA text path:** subject steering depends on correct `scripts/h3_text_encode.py` NVFP4 dequant (`from_blocked` on swizzled `weight_scale`). Pre-fix embeds made pure T2VA seed-dominated (duck/apple → same golfer); post-fix duck/apple hit subjects @ 20 steps. Re-encode stale `text.safetensors` before sample. Short-vs-IR Bar B unblocked. Notes: `outputs/t2va_debug/REPORT.md` (gitignored).
- **Canvas presets** (`canvas.py` + `sample` / `canvas-table`):
  - `--aspect` + `--megapixels`/`--mp` or `--short-edge`; or `--width`/`--height`; or default official 16:9 **1344×768**.
  - `--canvas-mode preview|official|raw` (default `preview` keeps short edges <768 for owner ladders; `official` forces short=768).
  - Axes ×32; Base area cap **768×1344 (~1.032 MP)** — above that needs closed H3-Regenerate-2K or post-decode SR (`scripts/h3_upscale.py`).
  - `uv run python -m minimax_h3 canvas-table` prints official short-edge + MP ladder for any aspect.
- **Prompt / IR format (open):** `prompt_ir.py` + `scripts/h3_prompt_ir.py` emit official structured prompts (T2VA family 3 fields; Ref2VA 6 sections). Base does not rewrite — paste-ready IR text is the quality lever. Not hosted Context-IR; not 2K regen.
- Multi-window continue lives in `scripts/h3_fl2va_continue.py` (WanGP FL2VA protocol), not in the sample CLI loop.
- `save_safetensors` key names are the cross-runtime weight contract.
- Approximate long-gen shortcuts that failed owner-box A/B stay **removed**: portable multi-Sage window sparse; Triton fused sparse; first-block residual step cache; naive short-seg latent last-frame / Ref2VA tail-ref chain. Product default stays dense Sage + lossless stack. FL2VA continue is a separate, gated packaging path (see root AGENTS.md).

## Work Guidance

- Run this package through **uv**: `uv run python -m minimax_h3 …` / `uv run python minimax_h3/tests/…`. Root owns `pyproject.toml` / `uv.lock`.
- Change shared numerics in lockstep with the matching `src/*.rs` file.
- Streaming / prefetch / profiler / sample ergonomics may stay Python-first; do not weaken golden math.
- Keep CLI command names and flag meanings aligned with Rust for `audit`, `check-shapes`, `infer`, `list-params`, `layout-test`, `bench` where both implement them.
- Tests stay on CPU + tiny config + random weights unless a test explicitly needs otherwise.
- After lossless opts: gate with fixed-seed latent compare (`exact` preferred) on short-clip FL2VA/Ref2VA before claiming a win.
- Owner target: **16 GB VRAM + 64 GB RAM**; never introduce disk/NVMe weight offload.

## Verification

Python is uv-managed at repo root (`uv sync` once). Prefer `uv run` (no `PYTHONPATH=` needed — package is installed editable).

- `uv run python -m minimax_h3 --tiny check-shapes`
- `uv run python -m minimax_h3 layout-test`
- `uv run python -m minimax_h3 canvas-table`
- `uv run python minimax_h3/tests/test_model.py`
- `uv run python minimax_h3/tests/test_canvas.py`
- `uv run python minimax_h3/tests/test_prompt_ir.py`
- `uv run python minimax_h3/tests/test_opt_lossless.py`
- Cross-runtime: `uv run python scripts/parity_check.py` from repo root after `cargo build --release`
- Owner-box sample profile (short clip); write under gitignored `outputs/`:
  ```
  uv run python -m minimax_h3 --device cuda --attn sage sample \
    checkpoints/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    parity_real/short/inputs/text.safetensors \
    --steps 2 --width 864 --height 480 --length 39 --seed 42 \
    --profile --out outputs/sample_prof
  ```

## Child DOX Index

- None. `tests/` stays under this package contract.
