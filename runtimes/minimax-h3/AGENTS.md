# Purpose

Internalized MiniMax-H3 production runtime for `gemmy video h3`.

# Ownership

- **Full executable runtime lives here** — no external code checkout is required for production:
  - `minimax_h3/` package, production `scripts/`, Official FL2VA VAE modules (code only)
  - `ComfyUI/` — product Comfy engine: `run_h3_workflow.py`, `h3_workflow_build.py`, pins, pinned custom_nodes (Turbo / Sol / Spectrum / KJ / FBC / **RTX VSR Pro** / **PixelForge** / **DLSS Frame Interpolation**)
  - Workers: `gemmy_h3_comfy_generate.py` (**default** `--engine comfy`), `gemmy_h3_comfy_continue.py`, `gemmy_h3_comfy_edit.py` (SAM3 then Eros crop), `gemmy_h3_generate.py` (`--engine python`), optional `gemmy_h3_latent_upscale.py` / `gemmy_h3_face_refine.py` / `gemmy_h3_sprites.py` (PixelForge) / `gemmy_h3_interpolate.py` (DLSS Frame Generation) / `gemmy_h3_refmod.py` (RefMods)
  - `pyproject.toml` + `uv.lock`
- Multi-GB weights stay **external** (see `docs/MODEL_LOCATIONS.md`). Never vendor pruned DiT/TE/VAE shards into this tree.
- External `C:\Projects\minimax-h3` is research/reference only. **Deleting it must not break Gemmy** for either engine.

# Local Contracts

- Eros still-only generate/refine binds `comfy-workflow-h3-ganloss-stage1-still` and `stage2-still`. The unsuffixed packages retain crop-video conditioning for masked editing. Both variants share Gemmy's SDK-authored stage definitions; Python selects and binds compiled packages, never edits topology. Stage-2 width/height drive conditioning and dotted upscaler inputs with aspect locking off.

- Product CLI: `gemmy video h3 install|verify|doctor|generate|shortfilm|edit|continue|loop|upscale|interpolate|face-refine|sprites|refmod`. Do not clone Enndee into this Comfy pack.
- Shortfilm pipeline (`gemmy video h3 shortfilm`): Eros two-stage from ganloss video `LByGCGzu67o` (0.2 MP 608×352 then 1344×768 latent refine). Pack video `AykQHPVmG1w` supplies the director loop only — not stock FL2VA/turbo LoRA. `--scenes 1` one 6-panel window; `--scenes N` one Picture per scene. Extra model root `GEMMY_H3_EROS_CHECKPOINTS`.
- Code root: this folder (`runtime_path("minimax-h3")`). Optional dev override: `GEMMY_H3_ROOT`.
- Comfy root (production default): **`ComfyUI/` under this folder** (core **v0.36.0**; native H3 VAE + AddGuide + chunked I/O). Optional dev override only: `GEMMY_H3_COMFY`. Python pins matching that core: `comfy-kitchen==0.2.34` (`sol_attn` for `--vsa` plus fused fp16/int8 VAE decode), `comfy-aimdo==0.5.3` (pin only; never enable), `comfyui-frontend-package==1.52.7`.
- Checkpoints root: `GEMMY_H3_CHECKPOINTS` / `model_paths.minimax_h3` / documented models-root fallbacks. Runner builds a runtime `extra_model_paths` from that env plus Eros, stock Ref2VA backup, and RefMod (`GEMMY_H3_REFMODS` / `<checkpoints>/refmods`) roots so `UNETLoader` can see both speech DiTs and MiniMaxH3Mod can load saved latents. Persistent `:8188` uses on-disk `ComfyUI/extra_model_paths.yaml`.
- Scripts honor `GEMMY_H3_CHECKPOINTS` for TE/VAE weight paths; code/FL2VA modules resolve from this ROOT.
- Default quality: SageAttention (`fast`). HQ: real FlashAttention-2 (`--quality hq`).
- Default engine: **comfy**. Python engine retains TE→DiT→VAE subprocess split.
- Optional Comfy request fields: `init_audio` (wav encoded into the **target** audio latent via `GemmyH3InitFromAudio`) and `denoise` < 1 on the euler KSampler path. Frozen `--ref-audio` is still context, not the starting performance. Product CLI: `gemmy audio-h3 convert --file <src> --voice <timbre>` (`--dit stock|eros`). Two `--voice` clips bind `comfy-workflow-h3-convert-2voice` (local experiment). Convert prepends 0.5s hush on init and drops snap leftover after the line.
- Comfy-only speed/style flags (rejected on python engine): `--turbo`, `--realism`, `--sol`, `--vsa` (Ref2VA gate transplant; mutex `--sol`/`--cache`), `--jev` (009jev Jev-guided native SLA; omit `--steps` = 4-step `res_multistep`; `--steps 20` = HQ euler/simple + native SLA; mutex `--vsa`/`--sol`/`--cache`; needs `TYPESAFE_API_KEY` from process env, config `env_overrides`, or repo `.env`; **not default**), `--sla-fixed 1|3|5|10` (true fixed native SLA; never calls Jev; no API key), `--sla-table PATH` (N×50 per-(step,layer) keep; `initial_policy=table`; never calls Jev; mutex `--jev`/`--sla-fixed`/`--no-sla`; later layer 0 stays 5). HQ fill is `jev_variability.majority_keep_table(..., conservative=True)`: high-conf majority at n=4 is kept (not padded to 10); STATIC `min_n=5` is labels only. Recommended recipes live under `experiments/jev-hq-20step/tables/h3-fl2va-family-20-sla-v2.json` (not the generate default), `--jev-log-dataset` (teacher JSONL via `jev_teacher.py`; 20-step rows go to a separate dataset, not the 4-step sets), `--cache`.
- Realism People LoRA (`--realism`): fal `h3-realism-people-t2v-i2v-r2v.safetensors` via plain `LoraLoaderModelOnly` **after** turbo; trigger `r34l1sm` auto-prepended; default strength 1.0 (0.6–0.8 lighter).
- Modes: `i2v` **default** (auto Krea 2 still when no `--first-frame`), `t2va`, `fl2va` (both stills), `ref2va` (`--ref-image` / `--ref-video` / `--ref-audio` / `--ref-av`; default DiT = **Eros INT8 two-stage** 608→1344 landscape, 352→768 portrait). Eros UNET is already the turbo merge — `--turbo` / Larryvrh LoRA do not stack (generate ignores `--turbo` on product Ref2VA). Stock Comfy-Org Ref2VA is `--weights` on `G:\Models\minimax-h3-backup`. `--dit singularity` is opt-in: the AI Brief dual-sample on the full WarmBloodAban INT8 (12 denoise steps, 544×960 unless overridden; mutex `--weights`; not Eros two-stage; four processes — encode exits before the UNET, turbo exits before the last 10 steps, decode is VAE-only). Live `--ref-audio` (1 or 2 clips) on default Eros two-stage binds `ganloss-stage1-still-audio` / `-2audio` / `-refmod-audio` then the matching stage-2 package. Stock `--weights` binds `comfy-workflow-h3-ref2va-audio` / `-2audio`. Audio cannot be the sole input. `--vsa` plus `--ref-audio` is not packaged.
- Experimental: `--allow-keyframe-refs` packs FL2VA keyframes with `--ref-*` (defaults to Ref2VA DiT).
- Ref2VA caps (package): ≤9 images, ≤3 video slots, ≤3 audio slots, ≤12 files; audio cannot be sole input.
- Python worker encodes product media to DiT latents via `h3_vae_encode.py`, then `python -m minimax_h3 sample`.
- FaceRefine worker loads `ComfyUI-H3-FaceRefine/nodes.py` (not the package `__init__`) after putting `comfy_root` on `sys.path`. H3 CLIPLoader is `device=cpu` (Qwen3-VL 32B NVFP4); `device=default` access-violates on 16 GB after a 13.4 GB partial TE load.
- Continue (default): `gemmy video h3 continue` → `gemmy_h3_comfy_continue.py` (Comfy `native_guide`: previous AV tail as `minimax_keyframes`, empty target, no denoise mask). Native Guide **re-renders** the tail at the head of the new window; assemble uses SatoDive stitch (cut source at that overlap, colour-match, source audio through the overlap) — not ffmpeg concat of two full clips. Multi-window continue keeps one headless Comfy (`run_h3_workflow.py --serve`) and **interns UNET/CLIP/VAE by filename** so window 2 cannot disk-load a second 15 GB TE + DiT (that hard-locked this 16 GB box). `--ref-mod` / one `--ref-image` stack `minimax_refs` (Eros). `--legacy-masked-av` freeze-prefix + trim, then ffmpeg concat. `--legacy-fl2va` → `scripts/h3_fl2va_continue.py`.
- Loop: `gemmy video h3 loop` / `gemmy video-h3-loop` — Gemmy-owned scene plan + review/retry/reroll + assemble. Comfy stays the per-scene sampler. Optional plan-level `mode` / `weights` / `cache` / `steps` / `shift_video` / `shift_audio` (shortfilm sets Ref2VA Eros, cache off, 8 steps, shift 12/3). Contract: `docs/H3_CONTEXT_LOOP.md`.
- Shortfilm: `gemmy video h3 shortfilm` / `video-h3-shortfilm` — ganloss 6-grid (`--scenes 1`) or pack per-scene loop (`--scenes N`). Contract: `docs/H3_SHORTFILM_PIPELINE.md`. Two-stage on 16 GB is **two processes**: `ganloss_two_stage` = SplitSigmas@4 persist latent, then `run_eros_stage2` (3D SR + 3-step) in a fresh Comfy. One-graph refine dumps the DiT (~24 min/step). Per-scene IR is live-action (no storyboard/grid/`[Shot]`/`scene N of M` — Eros prints a beat sheet). Quality lock: `outputs/locked_h3_shortfilm_parlor_20260827/` (one 10s take; do not overwrite).
- Masked edit: `gemmy video h3 edit` / `video-h3-edit` — SAM3.1 track + overlay + analyze gate; crop is **Ref2VA `ref_video`** (ganloss JSON), not encode+noise_mask inpaint. Empty 0.2 MP latent, `ganloss_two_stage` SplitSigmas@4 persist, then `latent_refine` 3D SR + 3-step ManualSigmas; uncrop onto the plate. SAM3 is *where we must edit*, not a tight key: uncrop pastes matching crop background so leftover source hair volume cannot print as a hedge cutout; high-diff unmasked people stay original (`h3_mask_edit._paste_alpha`). `--feather` stays opaque on the subject. `--stop-after-mask` exits before Eros. Do not vendor GPL MaskVidExperiments. Source audio muxed. Guide: `gemmy guides --id video.h3.edit`.
- Upscale: `gemmy video h3 upscale` → pixel `scripts/h3_upscale.py` or `gemmy_h3_latent_upscale.py`. **RTX backend = vendored Pro node** `ComfyUI/custom_nodes/ComfyUI-NVIDIA-RTX-VSR-Pro` (`RTXVideoSuperResolution`) via `run_h3_workflow.py`. **`--backend latent`** is the official LBH recipe: `MinimaxH3LatentUpscaler3D` enlarge + `build_h3_workflow(..., continuation_mode="latent_refine")` (ManualSigmas `0.9035, 0.6316, 0.3158, 0.0000` euler, or `--denoise` KSampler) + decode; needs `--prompt`; native 480p/720p only. **`--backend latent-preview`** is enlarge-then-decode. LBH `mode` is a Comfy V3 DynamicCombo: `mode` plus dotted `mode.scale` / `mode.width` / `mode.height`. Fallbacks: Video2X → ffmpeg. Never copy pixel SR or a preview enlarge into same-size `masked_av`. A finished latent-refine sidecar is a new-size sample; continue there. Do **not** reintroduce a homemade direct-nvvfx loop as the product rtx path.
- FaceRefine: optional `gemmy video h3 face-refine` via a **local** `ComfyUI-H3-FaceRefine` clone (MIT, gitignored) + `face_yolov8m.pt` + `ultralytics` in this runtime venv. Same CPU TE pin as generate. Post on a finished MP4, not inside continue encode. LBH 3D upscaler is the same: local clone only, never committed (no LICENSE on the node repo).
- Motion Context: optional local **GPL-3.0** clone `ComfyUI-H3-Motion-Context` @ **v0.3.1** (`725a731`) for continuous FL2VA **speech** (`audio-h3` graphs `h3-speech-motion-context` / `h3-speech-motion-3`). Gitignored; doctor-notes only. Product Comfy is 0.36.0, so **v0.6.2 is possible later** — this pass does not bump the clone. **Not** `gemmy-h3-context` and **not** a video continue/loop replacement. Keep Spectrum off on those speech graphs. Extra widgets our graphs still pass (`encode_mode` / `anchor_mode` / `audio_mode` / `crop`) are hardcoded in 0.3.1 to the same values.
- Sprites: `gemmy video h3 sprites` runs pinned **ComfyUI-PixelForge-H3** (`japaneserunic/blubs-pixel-nodepack`, MIT) via package `comfy-workflow-h3-pixelforge`. Key (full res) → crop to character → loop trim → sheet/GIF. Do not quantize the full chroma plate. Generation stays on `gemmy image` + `video h3`. No OpenRouter / fal / Nano Banana. Old gary149 cut/atlas scripts under `scripts/sprites/` are not the product path.
- RefMods: `gemmy video h3 refmod create|list|inspect` and generate / continue `--ref-mod`. Pinned MIT **ComfyUI-MiniMaxH3Mod** @ `v0.2.6`. Saved latents under `GEMMY_H3_REFMODS` or `<checkpoints>/refmods`. Create is VAE-only (no DiT). Not a LoRA. Audio extract is experimental (no speaker clone). H3RefModPicker is not pinned.
- Interpolate: `gemmy video h3 interpolate` runs pinned **ComfyUI-NVIDIA-DLSS-Frame-Interpolation** (`Konohamaru04` @ `c755e27`, MIT nodes) via package `comfy-workflow-h3-dlss-interpolate`. Frame Generation post on a finished MP4 (default 24→48). Super Resolution nodes stay unwired — pixel upscale is VSR-Pro. NVIDIA SDK DLLs under `bin/runtime/` are **local Git LFS, not committed**. `interpolate --check` probes real DLL sizes. Do not feed the result into continue/loop encode. Hardware-accelerated GPU scheduling recommended.
- No disk weight offload. Python streams RAM→VRAM. Comfy runner **forces** `--disable-dynamic-vram` + `--disable-mmap` (AIMDO mmap can bounce dirty weight pages onto the checkpoint NVMe — forbidden) and `--fast fp16_accumulation` (H3 video VAE encoder `fp16_conv3d` only; not bare `--fast`). RAM↔VRAM only.
- **Nightshift live look-check:** when `GEMMY_LIVE_PREVIEW` is set, `ComfyUI/run_h3_workflow.py` injects `--preview-method auto` (Latent2RGB, not TAESD) and hooks `ProgressBar` to atomically write that JPEG plus `[h3-preview] step=N/M`. Gemmy H3 spawn forwards the env into the python child. Do not start the Comfy HTTP/websocket server for this. Do not feed the JPEG back into continue/loop.
- `uv` only for the local `.venv` (never pip). Install: `gemmy video h3 install`. Optional RTX path: `uv pip install nvidia-vfx` from NVIDIA index (not required for generate).
- Comfy scratch (`ComfyUI/input|output|temp|user`) is gitignored; engine source is tracked.

# Work Guidance

- Prefer JSON request files over shell-interpolated prompts.
- Same request.json schema for both engines; comfy fields are no-ops on the python worker.
- Python path: keep VRAM sequencing — never load TE + DiT + VAE in one process.
- Comfy path: TE+DiT+VAE may share the Comfy process with legacy NORMAL_VRAM partial load (CPU RAM resident, VRAM working set); never AIMDO/mmap disk paging.
- Comfy core pin is **v0.36.0**. Native H3 VAE + AddGuide + chunked I/O. Do not full-pull Comfy master over this pack. SaveVideo `format` is a DynamicCombo: option-key string plus dotted `format.codec` (rewritten in `run_h3_workflow.py` from packaged 0.33.4 strings). Windows `--disable-mmap` stays `backend=pread` in `comfy/utils.py`. Default video VAE is official Comfy-Org **int8 ConvRot**; `--vae fp16` is the official fp16 file. `--engine python` is fp16 only.
- IR compile on by default for freeform prompts (`scripts/h3_prompt_ir.py`); Ref2VA uses `mode=ref2va` IR.
- Prefer one-shot generate when length fits; use continue for longer same-take extensions; hard-cut new shots instead of forcing a bad seam.
- Never copy pixel-SR frames or a preview-only latent enlarge into continue/loop at the old canvas (cond geometry changes). After `--backend latent` refine, continue at the new canvas using the new sidecar.
- Refresh/sync from an external H3 research tree only when intentionally updating production code inside this runtime; never reintroduce a hard dependency on that tree's path.

# Verification

- `./runtimes/minimax-h3/.venv/Scripts/python.exe -B -m unittest discover -s comfy-workflows/tests -p test_eros_requests.py -v` from the Gemmy root exercises still-only generate and refine bindings, landscape/portrait dimensions, and crop-video compatibility without GPU execution.

- `gemmy video h3 doctor` (must report `comfy_root` under `runtimes\minimax-h3\ComfyUI` when `GEMMY_H3_COMFY` unset)
- `gemmy video h3 verify`
- `gemmy video h3 generate --prompt "…" --duration 5 --dry-run-plan` (prints `video_vae=minimax_h3_video_vae_int8_convrot.safetensors`)
- `gemmy video h3 generate --vae fp16 --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 generate --engine python --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 generate --turbo v4 --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 generate --realism --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 generate --mode ref2va --ref-image still.png --prompt "…" --duration 5 --dry-run-plan` (Eros two-stage 608→1344, steps=8)
- `gemmy video h3 generate --jev --prompt "…" --duration 5 --dry-run-plan` (009jev; 4-step res_multistep; needs `TYPESAFE_API_KEY` + `jev-sdk` venv; not default)
- `gemmy video h3 generate --jev --steps 20 --prompt "…" --duration 5 --dry-run-plan` (HQ euler + native SLA; not remapped to 4)
- `gemmy video h3 generate --mode fl2va --first-frame a.png --last-frame b.png --jev --steps 20 --dry-run-plan`
- `gemmy video h3 generate --sla-fixed 5 --prompt "…" --duration 5 --dry-run-plan` (const keep; no Jev; no API key)
- `gemmy video h3 generate --sla-table majority.json --prompt "…" --duration 5 --dry-run-plan` (N×50 keep table; no Jev; no API key)
- `gemmy video h3 generate --no-sla --prompt "…" --duration 5 --dry-run-plan` (4-step res_multistep; no H3JevNativeSLAPatch)
- `runtimes/minimax-h3/.venv/Scripts/python.exe -B -m unittest discover -s runtimes/minimax-h3/tests -p test_jev*.py -v`
- `runtimes/minimax-h3/.venv/Scripts/python.exe -B -m unittest discover -s runtimes/minimax-h3/tests -p test_local_controller_eval.py -v` (zero-shot eval helpers; no GPU)
- `gemmy video h3 generate --mode ref2va --vsa --ref-image still.png --prompt "…" --duration 5 --dry-run-plan` (Eros two-stage; needs `loras/fasth3_vsa_gate.safetensors`)
- `gemmy video h3 generate --mode ref2va --vsa --weights <stock Ref2VA UNET> --ref-image still.png --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 generate --mode ref2va --ref-image still.png --ref-audio a.wav --no-compile-ir --dry-run-plan` (Eros two-stage still-audio)
- `gemmy video h3 generate --mode ref2va --weights <stock Ref2VA UNET> --ref-image still.png --ref-audio a.wav --ref-audio b.wav --no-compile-ir --dry-run-plan` (stock 2-audio)
- `./runtimes/minimax-h3/.venv/Scripts/python.exe -B -m unittest discover -s comfy-workflows/tests -p test_ref2va_audio_requests.py -v`
- `gemmy video h3 edit --video plate.mp4 --ref-image still.png --mask-prompt head --prompt "…" --duration 5 --dry-run-plan`
- `gemmy video h3 edit --video plate.mp4 --mask-prompt female --stop-after-mask --dry-run-plan`
- `gemmy audio-h3 convert --file take.wav --voice timbre.wav --dry-run-plan` (stock Ref2VA, init_audio + denoise 0.58, 64×64 dark picture)
- `gemmy video h3 continue --prompt "…" --num-windows 2 --dry-run-plan` (native_guide + 39-frame snap)
- `gemmy video h3 continue --ref-mod hero --prompt "The woman in <Picture 1> …" --dry-run-plan` (Eros native-guide + RefMod)
- `gemmy video h3 loop --plan plan.json --dry-run-plan`
- `gemmy video h3 upscale --check`
- `gemmy video h3 upscale --input x.mp4 --backend latent --prompt "…" --canvas 720p --dry-run-plan`
- `gemmy video h3 upscale --input x.mp4 --backend latent-preview --canvas 1080p --dry-run-plan`
- `gemmy video h3 face-refine --input x.mp4 --dry-run-plan`
- `gemmy video h3 interpolate --input x.mp4 --fps 48 --dry-run-plan`
- `gemmy video h3 interpolate --check`
- `gemmy video h3 sprites --input idle.mp4 --loop --atlas --dry-run-plan`
- `gemmy video h3 refmod create --folder refs --name hero --dry-run-plan`
- `gemmy video h3 generate --mode ref2va --ref-mod hero --prompt "…" --dry-run-plan`
- Smoke (long): `gemmy video h3 generate --prompt "…" --duration 5 --steps 8 -o outputs/h3_smoke.mp4`
- Deletion test: with `GEMMY_H3_COMFY` unset, doctor/generate must not require `C:\Projects\minimax-h3` for **code** (weights may still resolve there if that is the configured checkpoints root).

# Child DOX Index

- `ComfyUI/AGENTS.md` — product Comfy pack contracts (runner, builder, pins); upstream Comfy style rules may appear below the H3 block.
