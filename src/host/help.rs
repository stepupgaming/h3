//! Help text for the standalone H3 CLI. Command names are h3, not gemmy video h3.

#[allow(dead_code)]
pub(crate) const VIDEO_HELP: &str = r#"Video generation engines under a shared lifecycle namespace.

Production engine:
  h3 …      MiniMax-H3 joint audio-video DiT (owner 16 GB path)

Lifecycle:
  h3 install
  h3 verify
  h3 doctor
  h3 generate --prompt "…" [options]   # i2v|t2va|fl2va|ref2va
  h3 shortfilm …                       # 6-grid Ref2VA Eros; --scenes N = one still per scene
  h3 edit --video src.mp4 --ref-image still.png --mask-prompt head --prompt "…"
  h3 continue --prompt "…" [options]   # same-shot native latent guide
  h3 loop --plan plan.json [options]   # multi-scene Context Loop
  h3 upscale --input out.mp4 [options] # RTX / LBH latent refine / Video2X / ffmpeg
  h3 interpolate --input out.mp4       # DLSS Frame Generation post (not upscale)
  h3 face-refine --input out.mp4
  h3 sprites --input clip.mp4

See also: h3 --help"#;

pub(crate) const VIDEO_H3_HELP: &str = r#"MiniMax-H3 joint audio-video generation.

Default generate engine is **Comfy** (native H3 graph + pinned accel nodes).
Fallback: --engine python (streamed run_sample TE→DiT→VAE subprocess split).

Owner-box path: pruned int8 DiT, SageAttention default (fast), real FlashAttention-2
for --quality hq. No disk weight offload. Comfy uses dynamic VRAM; python streams
int8 RAM→VRAM.

Code (in this repo):
  runtimes\minimax-h3
    package, scripts, Official FL2VA modules, uv .venv
    ComfyUI\                 # product Comfy engine + pinned custom_nodes
    gemmy_h3_comfy_generate.py   # default --engine comfy
    gemmy_h3_generate.py         # --engine python
  Dev overrides only: GEMMY_H3_ROOT / GEMMY_H3_COMFY
Weights (external multi-GB — never in git):
  h3 install
  downloads base, eros, latent, and face. h3 download --list shows the rest.
  h3 setup --checkpoints D:\h3-models
  Saved roots: %APPDATA%\h3\config.json.
  Env GEMMY_H3_CHECKPOINTS still wins. An existing F:\Models\minimax-h3-eros
  or G:\Models\minimax-h3-backup is kept when you have not set a path.
  Comfy: bundled runtimes\minimax-h3\ComfyUI, or
  h3 setup --comfy <folder with main.py>.
  That copies the H3 runner and shipped nodes, then downloads missing product packs.
  turbo LoRA: checkpoints\loras\minimax_h3_turbo_v4_step600_ema.safetensors
  realism LoRA: checkpoints\loras\h3-realism-people-t2v-i2v-r2v.safetensors (trigger r34l1sm)

Lifecycle:
  h3 install          Python env, ffmpeg, product node packs, and base/eros/latent/face weights
  h3 setup            choose checkpoints folder and Comfy pack
  h3 download         list or fetch weight sets (base, eros, ref2va-stock, …)
  h3 verify           required scripts + weight files present
  h3 doctor           CUDA / Sage / FA2 / Comfy pins / ffmpeg / paths
  h3 generate …       one-shot prompt → MP4 + .h3av.safetensors sidecar
  h3 shortfilm …      6-grid Ref2VA Eros; --scenes N = one still per scene (not default I2V)
  h3 edit …           SAM3 track + Eros crop sample + uncrop (masked region replace)
  h3 continue …       same-shot native latent guide (--legacy-masked-av freeze-prefix; --legacy-fl2va last-RGB)
  h3 loop …           multi-scene Context Loop (plan + review + assemble)
  h3 upscale …        post-decode SR (RTX / LBH latent refine / Video2X / ffmpeg)
  h3 interpolate …    DLSS Frame Generation post (24→48 default; not an upscale path)
  h3 face-refine …    optional face regenerate-and-stitch post
  h3 sprites …        PixelForge keyed sprite loop / sheet / GIF
  h3 refmod …         save/list/inspect reusable H3 reference latents

Generate modes:
  --mode i2v      first-frame lock (DEFAULT). Auto Krea 2 still when --first-frame omitted
  --mode t2va     text → video+audio (no still)
  --mode fl2va    first+last lock (needs both stills)
  --mode ref2va   reference-to-video+audio. Default: Eros two-stage 608→1344
                  (parlor lock). Stock Comfy-Org Ref2VA is --weights on G:\Models\minimax-h3-backup
                  --dit singularity runs the AI Brief dual-sample graph
                  (full INT8, 12 denoise steps, 544x960 unless you pass size).

Engine / speed (Comfy default):
  --engine comfy|python          default comfy; python = legacy streamed path (fp16 VAE only)
  --turbo off|v4|v1|ema_wan      Larryvrh Turbo LoRA + Turbo Sampler (product = v4)
  --turbo-strength 1.0           Larryvrh default (tweak 0.8–1.2)
  --turbo-steps 4|6|8            when turbo on and --steps still 20, defaults to 4
  --realism                      fal MiniMax-H3-Realism-People LoRA (T2V/I2V/R2V)
  --realism-strength 1.0         card default 1.0; 0.6–0.8 lighter
  --sol / --no-sol               Saganaki H3 Scheduled Sol + KJ mem-sage compose
  --vsa                          Ref2VA Visual Sparse Attention (Kablex gate; mutex --sol/--cache)
  --vsa-sparsity 0.75            video-tile sparsity (keep = 1 - sparsity)
  --vsa-gate NAME                gate file under checkpoints/loras/ (default fasth3_vsa_gate.safetensors)
  --jev                          opt-in 009jev native SLA (omit --steps = 4-step res_multistep; --steps 20 = HQ euler; not default; mutex --vsa/--sol/--cache; needs TYPESAFE_API_KEY)
  --sla-fixed 1|3|5|10           true fixed native-SLA keep (bypasses Jev; no API key; mutex --jev)
  --sla-table PATH               per-(step,layer) keep table JSON N×50 (bypasses Jev; 4×50 = 4-step recipe; 20×50 = HQ euler)
  --no-sla                       4-step res_multistep without native SLA (matched Jev control; not default 20-step euler)
  --jev-log-dataset DIR          append teacher JSONL after --jev / --sla-fixed / --sla-table generate
  --cache off|spectrum|easy|fbc  mutual exclusion; spectrum = xmarre product cache
  --dit-quant int8|w4a8          weight pack (int8 HQ default)
  --vae fp16|int8                official Comfy-Org video VAE (default int8)
  --video-vae NAME               filename under checkpoints/vae/ (wins over --vae)

Key generate flags:
  --prompt / --prompt-file   scene / motion text (IR-compiled by default)
  --still-prompt TEXT        Krea still prompt (default: same as --prompt)
  --first-frame PATH         skip auto Krea; use this still for I2V/FL2VA
  --last-frame PATH          FL2VA end still
  --ref-image PATH           Ref2VA image ref (repeatable, ≤9)
  --ref-video PATH           Ref2VA video ref (repeatable, ≤3 slots w/ --ref-av)
  --ref-audio PATH           Ref2VA audio ref (repeatable, ≤3; not sole input).
                             Default Eros two-stage still+audio; stock --weights also.
  --ref-av VIDEO,AUDIO       Ref2VA paired video+audio (repeatable)
  --ref-mod NAME[:STR[:N]]   saved RefMod (repeatable, ≤4). t2va or ref2va.
  --allow-keyframe-refs      experimental FL2VA keyframes + --ref-* (Ref2VA DiT)
  --no-auto-still            fail I2V if --first-frame missing (no Krea)
  --duration <s>             seconds @ 24 fps (snapped to 17k+5 grid; default 5)
  --frames <n>               exact frame request before snap (overrides duration)
  --canvas 480p|720p         product size (aliases 0.4 / 0.9). Default 480p.
                             480p = 864×480 (~0.4 MP)   native DiT default
                             720p = 1280×736 (~0.9 MP)  native DiT quality
                             1080p / 2.0 is upscale-only (above open Base DiT cap)
  --width/--height           explicit canvas (×32); default from --canvas
  --steps                    diffusion steps (default 20; turbo defaults 4)
  --shift-video / --shift-audio  MiniMaxH3SigmaShift (stock 12/3; 0 0 skips the node)
  --quality fast|hq          sage (default) or real FlashAttention-2
  --no-compile-ir            skip open IR field-order compiler
  --weights PATH             DiT override (default FL2VA, or Eros INT8 when --mode ref2va)
  --dit default|singularity  default keeps FL2VA / Eros. singularity runs
                             the AI Brief dual-sample graph on
                             F:\Models\minimax-h3-singularity\diffusion_models\Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors
                             (scheduler 6, extend 2, split at 2: 12 denoise steps;
                             544x960 unless you pass --width/--height or --canvas).
                             Ref2VA only. One --ref-image. Mutex --weights.
                             Not the default. Not Eros two-stage.
  --dry-run-plan             print resolved plan only
  --keep-work                keep latents/text/decode sidecars
  --json / -v

Continue (same-shot longer takes; default native_guide, not last-RGB):
  h3 continue --prompt "…" --num-windows 2 -o long.mp4
  h3 continue --input existing.mp4 --prompt "…" --num-windows 2
  h3 continue --input existing.mp4 --ref-mod hero --prompt "The woman in <Picture 1> …"
  --ref-mod NAME[:STR[:N]]   saved RefMod (repeatable, ≤4). Switches DiT to Eros Ref2VA.
  --ref-image PATH           one live Ref2VA still (stacks with --ref-mod via Apply)
  --context-frames 39 (AV-safe snap 39/90/141), --input VAE-tails when no .h3av sidecar
  --legacy-masked-av keeps freeze-prefix copy + 0/1 masks
  --legacy-fl2va keeps WanGP last-RGB + --overlap 1 for A/B only
  Same --canvas 480p|720p ladder as generate. Native-guide is FL2VA or Ref2VA;
  --ref-mod / --ref-image need a .h3av sidecar (not VAE-tail import).

Loop (multi-scene Context Loop; CLI review, not Comfy UI):
  h3 loop --plan plan.json -o loop.mp4
  gemmy video-h3-loop --plan plan.json --review
  --retry / --reroll / --approve / --stop / --start-scene N / --scene-range A:B
  See: gemmy guides --id video.h3.loop

RefMods (saved VAE latents, not LoRAs; no extra weights):
  h3 refmod create --folder refs --name hero
  h3 refmod list
  h3 refmod inspect hero
  h3 generate --mode ref2va --ref-mod hero --prompt "…"
  Library: GEMMY_H3_REFMODS or <checkpoints>/refmods
  Audio extract is experimental; speaker identity transfer is not claimed.
  See: gemmy guides --id video.h3.refmod

Upscale:
  h3 upscale --input out.mp4 --canvas 1080p -o out_1080p.mp4
  h3 upscale --input out.mp4 --backend latent --prompt "same scene continues" --canvas 720p
  h3 upscale --input out.mp4 --backend latent-preview --canvas 1080p
  h3 upscale --check
  --canvas 480p|720p|1080p   (aliases 0.4 / 0.9 / 2.0; 1080p = 1920×1088 ~2.0 MP)
  --backend auto|rtx|latent|latent-preview|video2x|ffmpeg
    auto prefers RTX VSR Pro + nvidia-vfx
    latent = LBH recommended (enlarge leftover AV latent, then a second H3 sample
             at the new native size; needs --prompt; default canvas 720p; 1080p
             fails closed). Persist a new .h3av at that canvas; continue/loop
             there is valid. Do not stuff the enlarged latent into same-size
             masked_av at the old canvas.
    latent-preview = enlarge then decode only (1080p ok; not a continue source)

Interpolate (DLSS Frame Generation post; not inside continue encode):
  h3 interpolate --input out.mp4 --fps 48 -o out_48.mp4
  h3 interpolate --check
  --fps 23.976|25|29.97|30|48|50|59.94|60|90|120   default 48 (exact 2× from 24); max 6× source
  --engine auto|native|cascade                   Auto uses native DLSSG then cascade
  Pixel upscale stays `video h3 upscale --backend rtx`. Do not stack this with T8 DLSS-NR.

Face refine / sprites (optional posts; not inside continue encode):
  h3 face-refine --input out.mp4 -o out_face.mp4
  h3 sprites --input idle.mp4 --loop --atlas -o sprites/

Prompt writing (official MiniMax guides; fetch in robot mode):
  gemmy guides --catalog
  gemmy guides --id video.h3.base    T2VA / I2VA / FL2VA / L2VA (I2V is this + first-frame)
  gemmy guides --id video.h3.ref     Ref2VA six-section format
  gemmy guides --id video.h3.loop    multi-scene continuation contract
  Freeform --prompt is IR-compiled by default; --no-compile-ir sends text as written.

Default product path: prompt → Krea 2 hero still @ canvas → Comfy H3 I2V (Sage, 20 steps).
Speed stacks (Comfy only): --turbo v4, --sol, --cache spectrum (do not stack spectrum+easy+fbc).
Ref2VA VSA: --mode ref2va --vsa (Eros two-stage, or stock UNET with --weights; needs fasth3_vsa_gate.safetensors; do not stack with --sol/--cache).
009jev native SLA: --jev (omit --steps / --steps 4 = author 4-step res_multistep; --steps 15|20|32 = stock HQ euler/simple + native SLA; not default; mutex --vsa/--sol/--cache; needs TYPESAFE_API_KEY on the Comfy process). True fixed keep: --sla-fixed 1|3|5|10 (never calls Jev). Per-cell table: --sla-table PATH (N×50 JSON; never calls Jev). Teacher JSONL: --jev-log-dataset DIR.
Style (Comfy only): --realism (plain LoraLoaderModelOnly after turbo; auto-prepends r34l1sm).
Python fallback: --engine python (TE/DiT/VAE subprocess split; no turbo/realism/sol/cache).
Production DiT: pruned int8 ConvRot (w4a8 optional via --dit-quant w4a8).

Examples:
  h3 doctor
  h3 generate --prompt "A red fox trots through fresh snow" --duration 5 -o outputs/fox.mp4
  h3 generate --turbo v4 --prompt "…" --duration 5
  h3 generate --realism --prompt "portrait of a woman in soft daylight" --duration 5
  h3 generate --turbo v4 --realism --prompt "…" --duration 5
  h3 generate --turbo v4 --sol --cache spectrum --prompt "…" --duration 5
  h3 generate --engine python --prompt "…" --duration 5
  h3 generate --canvas 720p --prompt "…" --duration 5
  h3 generate --first-frame still.png --prompt "camera slowly pushes in" --duration 5
  h3 generate --mode t2va --prompt "abstract aurora over a lake" --duration 5
  h3 generate --mode ref2va --ref-image subject.png --prompt "walks through a neon market" --duration 5
  h3 generate --mode ref2va --ref-mod hero --prompt "the woman in <Picture 1> walks through a neon market" --duration 5
  h3 refmod create --folder refs --name hero
  h3 refmod list
  h3 generate --mode ref2va --vsa --ref-image subject.png --prompt "walks through a neon market" --duration 5
  h3 generate --jev --prompt "A red fox trots through fresh snow" --duration 5 --dry-run-plan
  h3 generate --sla-fixed 5 --no-compile-ir --first-frame still.png --prompt-file prompt.txt --width 864 --height 864 --duration 5 --dry-run-plan
  h3 generate --sla-table majority.json --no-compile-ir --first-frame still.png --prompt-file prompt.txt --width 864 --height 864 --duration 5 --dry-run-plan
  h3 generate --no-sla --no-compile-ir --first-frame still.png --prompt-file prompt.txt --width 864 --height 864 --duration 5 --dry-run-plan
  h3 generate --mode ref2va --vsa --weights G:\\Models\\minimax-h3-backup\\diffusion_models\\minimax_h3_ref2va_pruned_int8_convrot.safetensors --ref-image subject.png --prompt "…" --duration 5
  h3 shortfilm --character-sheet sheet.jpg --storyboard board.png --prompt "…" --dry-run-plan
  h3 generate --mode fl2va --first-frame a.png --last-frame b.png --ref-image id.png --allow-keyframe-refs --prompt "…"
  h3 continue --prompt "…" --num-windows 2 -o long.mp4
  h3 continue --input existing.mp4 --ref-mod hero --prompt "The woman in <Picture 1> walks on"
  h3 loop --plan plan.json --dry-run-plan
  h3 upscale --input long.mp4 --canvas 1080p -o long_1080p.mp4
  h3 upscale --input long.mp4 --backend latent --prompt "…" --canvas 720p
  h3 upscale --input long.mp4 --backend latent-preview --canvas 1080p
  h3 interpolate --input long.mp4 --fps 48
  h3 interpolate --check
  h3 face-refine --input long.mp4
  h3 sprites --input idle.mp4 --loop --atlas
  h3 generate --prompt "…" --quality hq --steps 20 --duration 5
  h3 shortfilm --help     Short Film Director + Eros Ref2VA (separate pipeline)
  h3 edit --help          Masked region replace (SAM3 + Eros crop)"#;

pub(crate) const VIDEO_H3_REFMOD_HELP: &str = r#"MiniMax-H3 RefMods — reusable VAE-encoded reference latents.

A RefMod is a ~1 MB .safetensors cache of H3 video/audio VAE latents. It is not
a LoRA and does not train the DiT. Create needs the stock H3 VAEs only.

  h3 refmod create --folder refs --name hero
  h3 refmod create --image a.png --image b.png --name hero
  h3 refmod create --folder refs --audio voice.wav --name hero
  h3 refmod list
  h3 refmod inspect hero
  gemmy video-h3-refmod create --folder refs --name hero --dry-run-plan

Then generate or continue:
  h3 generate --mode ref2va --ref-mod hero --prompt "the woman in <Picture 1> …"
  h3 generate --mode ref2va --ref-image still.png --ref-mod hero --prompt "…"
  h3 continue --input take.mp4 --ref-mod hero --prompt "The woman in <Picture 1> walks on"

Library: GEMMY_H3_REFMODS or <H3 checkpoints>/refmods (mapped in extra_model_paths).
Default visual mode is encode (Full Reference / identity). Audio extract is
experimental; speaker identity transfer is not claimed.
See: gemmy guides --id video.h3.refmod
"#;

pub(crate) const VIDEO_H3_SHORTFILM_HELP: &str = r#"Best of both: Eros two-stage (ganloss video) + director loop (The AI Brief pack).
Not default `h3 generate` I2V.

Two source videos — do not mix their graphs:

  LByGCGzu67o  ganloss 6-grid Ref2VA. Main model = 10 Eros Max Turbo (not a LoRA).
               Stage 1 = 0.2 MP (608×352), euler/simple/8, shift 12/3, cache off.
               Do not decode. 3D latent upscale + second sample → 1344×768.

  AykQHPVmG1w  The AI Brief pack. Krea multi-shot stills + labeled board +
               stock MiniMax H3 Contex Loop (chain plan, @hero_face/@storyboard,
               review, assemble). Standard H3 UNET. Turbo LoRA off. Latent
               upscaler is NOT in the shipped pack.

This command: Eros two-stage sampler (first video) inside Gemmy `video h3 loop`
director chain (second video). `--scenes 1` = one 6-panel window. `--scenes N` =
one still per scene, continuation_mode none (director cuts), assemble.

Do not I2V the board as --first-frame. Character sheet is stills-only (not a
second DiT ref — that split-screens).

`--scenes 1` (default) reproduces
https://github.com/amao2001/ganloss-latent-space/blob/main/workflow/2026-08-26%20minimax_h3_r2v_story_board.json

  One 6-panel photoreal storyboard as the sole MiniMaxH3ReferenceToVideo Picture.
  Eros TURBO: euler / simple / 8 steps / MiniMaxH3SigmaShift 12/3 / cache off.
  Official six IR sections. <Picture 1> is that board. Treat panels as chronological
  shot beats, not a split-screen composite. Do not pass a character sheet as a
  second DiT ref.

`--scenes N` (N>1) is the pack path: Krea identity-edit stills (`gemmy image
--engine krea2 --input-image` at 1536×864 / 12 steps), one still as Picture 1
per scene, hard-cut `h3 loop` (continuation_mode none). Character
sheets and beat stills must be sharp photoreal — Eros copies the still.
Undersized sheets (short-edge < 1536) / stills (< 864) / 6-grid boards (< 1536)
are SeedVR2'd first. Do not stuff six rooms into one 10s window. Per-scene IR
is live-action: do not name a storyboard / grid / split / contact sheet /
[Shot N] / scene N of M (Eros prints a beat sheet). Stills are one photograph
of one room — no framed paintings. Quality lock (2026-08-27):
outputs/locked_h3_shortfilm_parlor_20260827/ (one 10s take; do not overwrite).
Preview is a contact_sheet.png file. Do not wire it into Studio (studio/ is
dead; see the sibling gemmy-nightshift repo).

This command does **not** change `h3 generate` defaults (stock FL2VA I2V).
Do not I2V the storyboard as --first-frame.

Runnable DiT on RTX 5060 Ti 16 GB (no disk offload):
  F:\Models\minimax-h3-eros\diffusion_models\10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors
  (cicalooo/10Eros-Max-h3-int8-convrot)

Named BF16 (download; not loaded on 16 GB):
  F:\Models\minimax-h3-eros\diffusion_models\10Eros_Max_h3_TURBO_ref2va_beta2.safetensors
  (TenStrip/10Eros-Max)

Override: GEMMY_H3_EROS_CHECKPOINTS.

Usage:
  h3 shortfilm --storyboard board.png --prompt "…" --phase video --dry-run-plan
  gemmy video-h3-shortfilm --panel p1.png --panel p2.png --panel p3.png --panel p4.png --panel p5.png --panel p6.png --prompt "…" --phase video
  h3 shortfilm --scenes 2 --panel s1.png --panel s2.png --prompt "…" --phase video --dry-run-plan
  h3 shortfilm --character-sheet sheet.jpg --scenes 4 --prompt "…" --phase stills
  h3 shortfilm --phase download --dry-run-plan

Key flags:
  --storyboard PATH        6-panel board (sole Ref2VA Picture on --scenes 1; never a first-frame)
  --panel PATH             photoreal stills (repeatable); unlabeled 3×2 compose if --scenes 1 and no --storyboard
  --character-sheet PATH   pack stills only — not a DiT ref. Sharp photoreal; short-edge <1536 is SeedVR2'd.
  --panel-prompt TEXT      per-panel identity-edit prompts (1536×864, 12 steps; undersized panels SeedVR2)
  --prompt / --prompt-file film brief → six-section IR
  --grid 6|9|16            3×2 (default) / 3×3 / 4×4 (one-window board)
  --phase all|stills|video|download
  --scenes N               1 = ganloss one-window; N>1 = one still per scene, hard-cut loop
  --duration <s>           one H3 window / per-scene seconds (default 10)
  --steps N                Eros TURBO steps (default 8 from ganloss BasicScheduler)
  --out-dir DIR            stills / loop plan directory
  -o / --output            MP4
  --plan-out PATH          write gemmy-h3-loop-v1 JSON
  --dry-run-plan           print resolved plan only
  --json / -v

Guides: gemmy guides --id video.h3.shortfilm   (also video.h3.ref / video.h3.loop)
Workflow: docs\H3_SHORTFILM_PIPELINE.md
Weights: docs\MODEL_LOCATIONS.md (MiniMax-H3 Eros)

Examples:
  h3 shortfilm --phase video --storyboard six_panel.png --prompt "rooftop dance at night" --dry-run-plan
  h3 shortfilm --phase video --storyboard six_panel.png --prompt "…" -o film.mp4
  h3 shortfilm --scenes 2 --panel roof.png --panel diner.png --prompt "walks then sits" --phase video --dry-run-plan
  h3 shortfilm --phase download --dry-run-plan"#;

pub(crate) const VIDEO_H3_EDIT_HELP: &str = r#"Masked MiniMax-H3 region replace (SAM3.1 track + Eros crop sample + uncrop).

Not whole-frame Ref2VA. The mask answers *where* to change; --ref-image is
*who*. Source camera, unmasked people, and soundtrack stay.

Source: ganloss 2026-08-31 minimax_h3_r2v_video_mask_edit.json + Veteran AI
video PJWfUAO1Oco (Masking & Tracking). Crop/uncrop is Gemmy-owned — we do
not vendor GPL MaskVidExperiments. SAM3.1 is Comfy builtin.

16 GB: SAM3 runs in its own Comfy process, then unloads. Default gate:
`gemmy analyze` looks at the tinted overlay (same 12B path as loop --review).
Pass → ganloss Eros two-stage (crop as ref_video, empty 0.2 MP SplitSigmas@4,
then 3D SR + 3-step), then uncrop. Fail → stop before DiT.
No painter. Correct with --mask-grow, a new --mask-prompt, or --mask.

  h3 edit --video plate.mp4 --ref-image sheet.png \
      --mask-prompt head --prompt "replace her head with the reference" -o out.mp4

  h3 edit --video plate.mp4 --mask-prompt female --stop-after-mask
  h3 edit --video plate.mp4 --ref-image sheet.png --mask mask.mp4 \
      --prompt "…" -o out.mp4

  --video PATH             source plate (required)
  --ref-image PATH         replacement still (repeatable; required unless --stop-after-mask)
  --mask-prompt TEXT       SAM3.1 track text (head / female / …)
  --mask PATH              precomputed mask video (skips SAM3)
  --object-id 0            SAM3 object indices (comma-separated)
  --max-objects 1          0 = keep every detection
  --confidence 0.5
  --crop-mode combined|tracked     combined = one static box
  --crop-scale 1.5         padding around the subject
  --crop-mp 1.0            enlarge crop toward this megapixel count
  --feather 8              uncrop blend into background (subject stays opaque)
  --mask-grow 0            dilate mask before crop
  --mask-out PATH          copy cleaned mask MP4 after track
  --mask-overlay-out PATH  copy tinted overlay MP4 after track
  --verify-mask            analyze overlay before Eros (default on)
  --no-verify-mask         skip the analyze gate
  --stop-after-mask        track + overlay + verify; no Eros
  --prompt / --prompt-file video-editing IR (compiled unless --no-compile-ir)
  --duration / --frames    0 = from the source video (24 fps, 17k+5 snap)
  --steps 8                Eros
  -o / --output
  --dry-run-plan --json -v --keep-work

Flat alias: gemmy video-h3-edit
Guide: gemmy guides --id video.h3.edit
SAM3 weights: docs\MODEL_LOCATIONS.md (Comfy-Org/sam3.1)
"#;
