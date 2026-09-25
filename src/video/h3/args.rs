//! Clap surfaces for `h3`.

use super::canvas::H3CanvasPreset;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "h3")]
#[command(about = "MiniMax-H3 install, doctor, generate, shortfilm, edit, continue, loop, upscale, interpolate, face-refine, sprites, refmod")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3Args {
    #[command(subcommand)]
    pub(crate) command: H3Command,
}

#[derive(Subcommand, Debug)]
pub(crate) enum H3Command {
    /// Python env, ffmpeg, product node packs, and base/eros/latent/face weights.
    Install(H3InstallArgs),
    /// Choose the checkpoints folder, or point at a ComfyUI folder (main.py).
    Setup(H3SetupArgs),
    /// List or download weight sets (base, eros, ref2va-stock, singularity, turbo, …).
    Download(H3DownloadArgs),
    /// Verify runtime python, scripts, and required weight files.
    Verify(H3VerifyArgs),
    /// Probe CUDA, Sage/FA2, weights, ffmpeg, and paths.
    Doctor(H3DoctorArgs),
    /// Generate a joint audio-video clip with MiniMax-H3.
    Generate(H3GenerateArgs),
    /// Same-shot multi-window continuation (native latent guide; WanGP last-RGB is --legacy-fl2va).
    Continue(H3ContinueArgs),
    /// Multi-scene Context Loop production (plan + review + assemble).
    Loop(H3LoopArgs),
    /// Post-decode deliverable upscale (RTX / LBH latent refine / Video2X / ffmpeg).
    Upscale(H3UpscaleArgs),
    /// DLSS Frame Generation post on a finished MP4 (not an upscale path).
    Interpolate(H3InterpolateArgs),
    /// Optional face regenerate-and-stitch post pass on a finished MP4.
    #[command(name = "face-refine")]
    FaceRefine(H3FaceRefineArgs),
    /// PixelForge keyed sprite loop, sheet, and GIF from an H3 clip.
    Sprites(H3SpritesArgs),
    /// Short Film Director + 6-grid Ref2VA Eros storyboard pipeline (separate from default I2V).
    Shortfilm(H3ShortfilmArgs),
    /// Masked region replace: SAM3 track + Eros crop sample + uncrop (ganloss / Veteran AI).
    Edit(H3EditArgs),
    /// Save / list / inspect reusable H3 reference latents (RefMods).
    Refmod(H3RefModArgs),
}

#[derive(Parser, Debug)]
pub(crate) struct H3SetupArgs {
    /// Save this folder as the checkpoints root (`%APPDATA%\h3\config.json`).
    #[arg(long)]
    pub(crate) checkpoints: Option<std::path::PathBuf>,

    /// Use this Comfy folder. It must already contain `run_h3_workflow.py` and `custom_nodes/gemmy-h3-context`.
    #[arg(long, conflicts_with = "comfy_bundled")]
    pub(crate) comfy: Option<std::path::PathBuf>,

    /// Clear a saved Comfy path and use `runtimes/minimax-h3/ComfyUI`.
    #[arg(long, conflicts_with = "comfy")]
    pub(crate) comfy_bundled: bool,

    /// Save the Eros weight folder. Default is an existing `F:\Models\minimax-h3-eros`, else `<checkpoints>\eros`.
    #[arg(long)]
    pub(crate) eros: Option<std::path::PathBuf>,

    /// Save the stock Ref2VA folder. Default is an existing `G:\Models\minimax-h3-backup`, else `<checkpoints>\ref2va-stock`.
    #[arg(long)]
    pub(crate) ref2va_stock: Option<std::path::PathBuf>,

    /// Save the Singularity weight folder. Default is an existing `F:\Models\minimax-h3-singularity`, else `<checkpoints>\singularity`.
    #[arg(long)]
    pub(crate) singularity: Option<std::path::PathBuf>,

    /// Run `uv sync` even when the venv python is already present.
    #[arg(long)]
    pub(crate) sync: bool,

    /// Do not run `uv sync`.
    #[arg(long)]
    pub(crate) no_runtime: bool,

    #[arg(long)]
    pub(crate) verbose: bool,

    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct H3DownloadArgs {
    /// Weight set names. Omit to print the list. Example: `h3 download base eros`.
    #[arg(value_name = "SET")]
    pub(crate) sets: Vec<String>,

    /// Print sets and whether each file is already on disk.
    #[arg(long)]
    pub(crate) list: bool,

    /// Print URLs and destinations without downloading.
    #[arg(long)]
    pub(crate) dry_run: bool,

    #[arg(long)]
    pub(crate) verbose: bool,

    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct H3InstallArgs {
    /// Run `uv sync` in runtimes/minimax-h3 when the venv is missing/stale.
    #[arg(long, default_value_t = true)]
    pub(crate) runtime: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_runtime: bool,

    /// Only print the plan.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct H3VerifyArgs {
    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct H3DoctorArgs {
    /// Import torch + probe CUDA in the H3 venv.
    #[arg(long, default_value_t = true)]
    pub(crate) cuda: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_cuda: bool,

    /// Probe SageAttention import.
    #[arg(long, default_value_t = true)]
    pub(crate) sage: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_sage: bool,

    /// Probe flash_attn import (HQ path).
    #[arg(long, default_value_t = true)]
    pub(crate) fa2: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_fa2: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Mode {
    /// Text-to-video+audio (no still lock).
    #[value(name = "t2va", alias = "t2v")]
    T2va,
    /// Image-to-video: first-frame lock (default). Auto-generates the still with
    /// Krea 2 when `--first-frame` is omitted.
    #[default]
    #[value(name = "i2v", alias = "i2va")]
    I2v,
    /// First + last frame lock (both stills required).
    #[value(name = "fl2va", alias = "fl")]
    Fl2va,
    /// Reference-to-video+audio (subject/style refs). Needs Ref2VA DiT weights.
    #[value(name = "ref2va", alias = "ref")]
    Ref2va,
}

impl H3Mode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::T2va => "t2va",
            Self::I2v => "i2v",
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Quality {
    /// SageAttention SA2 fp8pp (default fast).
    #[default]
    #[value(name = "fast", alias = "sage")]
    Fast,
    /// Real FlashAttention-2 (HQ, slower).
    #[value(name = "hq", alias = "fa2", alias = "high")]
    Hq,
}

/// DiT generate engine. Comfy is the product default (after doctor gate).
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Engine {
    /// Native ComfyUI H3 graph (sage/fa2 CLI + Turbo/Sol/Spectrum nodes).
    #[default]
    #[value(name = "comfy")]
    Comfy,
    /// Legacy streamed Python `minimax_h3` sample worker.
    #[value(name = "python", alias = "native")]
    Python,
}

impl H3Engine {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Comfy => "comfy",
            Self::Python => "python",
        }
    }
}

/// Larryvrh / lab turbo LoRA selection for the Comfy engine.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Turbo {
    #[default]
    #[value(name = "off", alias = "none")]
    Off,
    /// Larryvrh v4 step600 EMA (product default turbo).
    #[value(name = "v4", alias = "larryvrh")]
    V4,
    /// Larryvrh v1 / ckpt850-class (4-step heavy motion fallback).
    #[value(name = "v1")]
    V1,
    /// WanGP/DeepBeepMeep ema ckpt850 lab file (not product default).
    #[value(name = "ema_wan", alias = "wan")]
    EmaWan,
}

impl H3Turbo {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::V4 => "v4",
            Self::V1 => "v1",
            Self::EmaWan => "ema_wan",
        }
    }
}

/// Step-skip / feature cache (Comfy engine). Mutually exclusive.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Cache {
    #[default]
    #[value(name = "off", alias = "none")]
    Off,
    /// xmarre ComfyUI-Spectrum-MiniMax-H3.
    #[value(name = "spectrum")]
    Spectrum,
    /// Native Comfy EasyCache (do not stack with spectrum/fbc).
    #[value(name = "easy")]
    Easy,
    /// duckyshell FirstBlockCache (experimental).
    #[value(name = "fbc")]
    Fbc,
}

impl H3Cache {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Spectrum => "spectrum",
            Self::Easy => "easy",
            Self::Fbc => "fbc",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3DitQuant {
    #[default]
    #[value(name = "int8")]
    Int8,
    #[value(name = "w4a8")]
    W4a8,
}

impl H3DitQuant {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::W4a8 => "w4a8",
        }
    }
}

/// Opt-in DiT family on generate. `default` keeps the mode's product UNET.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3Dit {
    #[default]
    #[value(name = "default")]
    Default,
    /// WarmBloodAban Minimax-h3_Singularity Ref2VA INT8 (full v1.3 file). Ref2VA graph only.
    #[value(name = "singularity")]
    Singularity,
}

/// Official Comfy-Org MiniMax-H3 **video** VAE pack. Audio fp32 is always required
/// and is not a `--vae` choice.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum H3VideoVae {
    #[value(name = "int8")]
    Int8,
    #[value(name = "fp16")]
    Fp16,
}

impl H3VideoVae {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
        }
    }
}

/// Shared `--vae` / `--video-vae` surface. Omitted flags let continue/loop reuse
/// a sidecar VAE; product default when nothing is set is official int8.
#[derive(Args, Debug, Clone, Default)]
pub(crate) struct H3VaeSelect {
    /// Official Comfy-Org video VAE: int8 (default) or fp16.
    #[arg(long, value_enum)]
    pub(crate) vae: Option<H3VideoVae>,

    /// Override video VAE filename under checkpoints/vae/ (wins over --vae).
    #[arg(long, value_name = "NAME")]
    pub(crate) video_vae: Option<String>,
}

impl H3VaeSelect {
    pub(crate) fn push_argv(&self, argv: &mut Vec<String>) {
        if let Some(v) = self.vae {
            argv.push("--vae".into());
            argv.push(v.as_str().into());
        }
        if let Some(name) = &self.video_vae {
            argv.push("--video-vae".into());
            argv.push(name.clone());
        }
    }
}

/// `h3 upscale --backend`.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3UpscaleBackend {
    /// RTX VSR Pro, then Video2X, then ffmpeg.
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "rtx")]
    Rtx,
    /// LBH recommended: enlarge leftover AV latent, then a second H3 sample at the new size.
    #[value(name = "latent")]
    Latent,
    /// LBH preview only: enlarge leftover AV latent and decode (no second H3 sample).
    #[value(name = "latent-preview")]
    LatentPreview,
    #[value(name = "video2x")]
    Video2x,
    #[value(name = "ffmpeg")]
    Ffmpeg,
}

impl H3UpscaleBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Rtx => "rtx",
            Self::Latent => "latent",
            Self::LatentPreview => "latent-preview",
            Self::Video2x => "video2x",
            Self::Ffmpeg => "ffmpeg",
        }
    }

    pub(crate) fn is_latent(self) -> bool {
        matches!(self, Self::Latent | Self::LatentPreview)
    }

    pub(crate) fn refine(self) -> bool {
        matches!(self, Self::Latent)
    }
}

impl H3Quality {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Hq => "hq",
        }
    }

    pub(crate) fn attn(self) -> &'static str {
        match self {
            Self::Fast => "sage",
            Self::Hq => "fa2",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 generate")]
#[command(about = "Generate joint A/V with MiniMax-H3")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3GenerateArgs {
    /// Scene / action prompt. Freeform is IR-compiled by default.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read prompt from a UTF-8 text file (overrides --prompt when set).
    #[arg(long, value_name = "PATH")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Product mode. Default is I2V (Krea 2 still → H3 motion).
    #[arg(long, value_enum, default_value_t = H3Mode::I2v)]
    pub(crate) mode: H3Mode,

    /// Attention quality: fast=Sage (default), hq=FlashAttention-2.
    #[arg(long, value_enum, default_value_t = H3Quality::Fast)]
    pub(crate) quality: H3Quality,

    /// Generate engine: comfy (default product path) or python (streamed run_sample).
    #[arg(long, value_enum, default_value_t = H3Engine::Comfy)]
    pub(crate) engine: H3Engine,

    /// Turbo LoRA (Comfy): off | v4 (Larryvrh product) | v1 | ema_wan (lab).
    #[arg(long, value_enum, default_value_t = H3Turbo::Off)]
    pub(crate) turbo: H3Turbo,

    /// Turbo LoRA strength (Larryvrh default 1.0; typical 0.8–1.2).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) turbo_strength: f64,

    /// Override steps when using turbo (default: keep --steps, else 4 if steps still 20).
    #[arg(long, value_name = "N")]
    pub(crate) turbo_steps: Option<u32>,

    /// Apply fal MiniMax-H3-Realism-People LoRA (Comfy; T2V/I2V/R2V). Trigger `r34l1sm` auto-prepended.
    #[arg(long, default_value_t = false)]
    pub(crate) realism: bool,

    /// Realism People LoRA strength (card default 1.0; 0.6–0.8 lighter).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) realism_strength: f64,

    /// Enable Saganaki Sol-Attn compose on Comfy (opt-in; needs Triton/SM).
    #[arg(long, default_value_t = false)]
    pub(crate) sol: bool,

    /// Disable Sol even if a preset would enable it.
    #[arg(long, default_value_t = false)]
    pub(crate) no_sol: bool,

    /// Opt-in Ref2VA Visual Sparse Attention (Kablex gate transplant). Mutex with --sol / --cache.
    #[arg(long, default_value_t = false)]
    pub(crate) vsa: bool,

    /// VSA video-tile sparsity (keep ratio = 1 - sparsity). Default 0.75 matches the published recipe.
    #[arg(long, default_value_t = 0.75)]
    pub(crate) vsa_sparsity: f64,

    /// VSA gate filename under checkpoints/loras/ (default fasth3_vsa_gate.safetensors).
    #[arg(long, value_name = "NAME", default_value = "fasth3_vsa_gate.safetensors")]
    pub(crate) vsa_gate: String,

    /// Opt-in 009jev: Jev-guided native SLA. Omit --steps (or --steps 4) for the author 4-step res_multistep recipe; --steps 15|20|32 uses stock HQ euler/simple. Mutex with --vsa/--sol/--cache. Requires TYPESAFE_API_KEY. Not the default generate path.
    #[arg(long, default_value_t = false)]
    pub(crate) jev: bool,

    /// True fixed native-SLA keep percent (1|3|5|10). Bypasses Jev entirely — no TYPESAFE_API_KEY. Mutex with --jev.
    #[arg(long, value_name = "PERCENT", value_parser = parse_sla_fixed)]
    pub(crate) sla_fixed: Option<u8>,

    /// Per-(step, layer) native-SLA keep table JSON (N×50). Bypasses Jev; not a single --sla-fixed keep. Mutex with --jev / --sla-fixed / --no-sla. 4×50 stays the 4-step recipe; 20×50 is HQ euler.
    #[arg(long, value_name = "PATH")]
    pub(crate) sla_table: Option<PathBuf>,

    /// Append Jev/SLA teacher JSONL (generations.jsonl + layer_decisions.jsonl) after generate.
    #[arg(long, value_name = "DIR")]
    pub(crate) jev_log_dataset: Option<PathBuf>,

    /// 4-step res_multistep without native SLA / H3JevNativeSLAPatch (matched Jev control). Mutex with --jev / --sla-fixed / --sla-table. Not default 20-step euler.
    #[arg(long, default_value_t = false)]
    pub(crate) no_sla: bool,

    /// Step cache: off | spectrum | easy | fbc (Comfy; mutually exclusive).
    #[arg(long, value_enum, default_value_t = H3Cache::Off)]
    pub(crate) cache: H3Cache,

    /// DiT weight quant pack: int8 (HQ default) | w4a8 (preview).
    #[arg(long, value_enum, default_value_t = H3DitQuant::Int8)]
    pub(crate) dit_quant: H3DitQuant,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// First-frame still (PNG/JPG). For default I2V, omit to auto-generate with Krea 2.
    #[arg(long, value_name = "PATH")]
    pub(crate) first_frame: Option<PathBuf>,

    /// Last-frame still (fl2va). PNG/JPG.
    #[arg(long, value_name = "PATH")]
    pub(crate) last_frame: Option<PathBuf>,

    /// Ref2VA image reference (PNG/JPG/WEBP). Repeatable, ≤9. Needs --mode ref2va.
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_image: Vec<PathBuf>,

    /// Ref2VA video reference (mp4/webm/…). Repeatable, ≤3 video slots with --ref-av.
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_video: Vec<PathBuf>,

    /// Ref2VA audio reference (wav/mp3/…). Repeatable, ≤3 audio total. Cannot be sole input.
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_audio: Vec<PathBuf>,

    /// Ref2VA paired video+audio: VIDEO_PATH,AUDIO_PATH (repeatable).
    #[arg(long, value_name = "VIDEO,AUDIO")]
    pub(crate) ref_av: Vec<String>,

    /// Saved RefMod name (`name` or `name:strength` or `name:strength:copies`). Repeatable, ≤4.
    #[arg(long, value_name = "NAME[:STRENGTH[:COPIES]]")]
    pub(crate) ref_mod: Vec<String>,

    /// Prompt for the auto Krea 2 first-frame still (default: same as --prompt).
    #[arg(long, value_name = "TEXT")]
    pub(crate) still_prompt: Option<String>,

    /// Do not auto-generate a Krea still when I2V has no --first-frame.
    #[arg(long, default_value_t = false)]
    pub(crate) no_auto_still: bool,

    /// Experimental: pack FL2VA --first-frame/--last-frame with --ref-image
    /// (Multishot-style DiT memory). Prefer Ref2VA weights.
    #[arg(long, default_value_t = false)]
    pub(crate) allow_keyframe_refs: bool,

    /// Output MP4 path.
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Product canvas: 480p (864×480 ~0.4 MP, default) | 720p (1280×736 ~0.9 MP).
    /// 1080p is upscale-only (above open Base DiT cap). Aliases: 0.4 / 0.9 / 2.0.
    #[arg(long, value_enum, value_name = "PRESET")]
    pub(crate) canvas: Option<H3CanvasPreset>,

    /// Canvas width override (multiples of 32). Default from --canvas / 480p.
    #[arg(long)]
    pub(crate) width: Option<u32>,

    /// Canvas height override (multiples of 32). Default from --canvas / 480p.
    #[arg(long)]
    pub(crate) height: Option<u32>,

    /// Duration in seconds at 24 fps (snapped to H3 17k+5 grid). 0 = use --frames.
    #[arg(long, default_value_t = 5.0)]
    pub(crate) duration: f64,

    /// Exact frame count before snap (0 = derive from duration).
    #[arg(long, default_value_t = 0)]
    pub(crate) frames: u32,

    /// Diffusion steps. Omitted: 20 stock HQ; 4 for --jev/--sla-fixed/--no-sla; 8 for default Eros Ref2VA. Pass --steps 20 with --jev for HQ euler + native SLA (not remapped to 4).
    #[arg(long)]
    pub(crate) steps: Option<u32>,

    /// MiniMaxH3SigmaShift video. Stock H3 and both source shortfilm graphs use 12.
    /// Pass 0 with --shift-audio 0 to skip the node.
    #[arg(long, default_value_t = 12.0)]
    pub(crate) shift_video: f64,

    /// MiniMaxH3SigmaShift audio. Stock H3 and both source shortfilm graphs use 3.
    /// Pass 0 with --shift-video 0 to skip the node.
    #[arg(long, default_value_t = 3.0)]
    pub(crate) shift_audio: f64,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Compile freeform prompt into open IR field order (default on).
    #[arg(long, default_value_t = true)]
    pub(crate) compile_ir: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_compile_ir: bool,

    /// DiT weights override (default: FL2VA pruned int8, or Eros INT8 when --mode ref2va).
    /// Stock Comfy-Org Ref2VA lives on G: as backup — pass that path here to use it.
    #[arg(long, value_name = "PATH")]
    pub(crate) weights: Option<PathBuf>,

    /// Opt-in DiT family. `default` keeps FL2VA / Eros. `singularity` loads the
    /// WarmBloodAban full INT8 on the AI Brief dual-sample graph (not Eros, not default).
    #[arg(long, value_enum, default_value_t = H3Dit::Default)]
    pub(crate) dit: H3Dit,

    /// Keep work dir artifacts (latents, text embeds, decode sidecars).
    #[arg(long, default_value_t = false)]
    pub(crate) keep_work: bool,

    /// Emit DiT sample profile.json.
    #[arg(long, default_value_t = false)]
    pub(crate) profile: bool,

    /// Validate and print the resolved plan without launching the worker.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,

    /// Internal Comfy sampler recipe. `ganloss_two_stage` is the official
    /// SplitSigmas@4 + latent 3D + 3-step graph (shortfilm). Not a user flag.
    #[arg(long, hide = true, default_value = "none")]
    pub(crate) continuation_mode: String,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 continue")]
#[command(about = "Multi-window same-shot continuation (native latent guide)")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3ContinueArgs {
    /// Scene / motion prompt (re-encoded per window when TE memory is on).
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read prompt from a UTF-8 text file (overrides --prompt when set).
    #[arg(long, value_name = "PATH")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Existing MP4 to continue (VAE-tail once if no `.h3av.safetensors` sidecar).
    #[arg(long, value_name = "PATH")]
    pub(crate) input: Option<PathBuf>,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Optional start still for window 1 (PNG/JPG).
    #[arg(long, value_name = "PATH")]
    pub(crate) start_image: Option<PathBuf>,

    /// Saved RefMod (`name` or `name:strength[:copies]`). Repeatable, ≤4.
    /// Switches the continue DiT to Ref2VA (Eros unless --weights).
    #[arg(long, value_name = "NAME[:STRENGTH[:COPIES]]")]
    pub(crate) ref_mod: Vec<String>,

    /// Live Ref2VA still for native-guide continue. At most one.
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_image: Vec<PathBuf>,

    /// Final stitched MP4 path.
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Work directory for per-window latents/decodes (default beside output).
    #[arg(long, value_name = "PATH")]
    pub(crate) work_dir: Option<PathBuf>,

    /// Product canvas: 480p (864×480 ~0.4 MP, default) | 720p (1280×736 ~0.9 MP).
    /// 1080p is upscale-only. Aliases: 0.4 / 0.9 / 2.0.
    #[arg(long, value_enum, value_name = "PRESET")]
    pub(crate) canvas: Option<H3CanvasPreset>,

    /// Canvas width override (multiples of 32). Default from --canvas / 480p.
    #[arg(long)]
    pub(crate) width: Option<u32>,

    /// Canvas height override (multiples of 32). Default from --canvas / 480p.
    #[arg(long)]
    pub(crate) height: Option<u32>,

    /// Frames per window before snap (default 362 ≈ 15s).
    #[arg(long, default_value_t = 362)]
    pub(crate) window: u32,

    /// Exact window count (alternative to --total-frames).
    #[arg(long)]
    pub(crate) num_windows: Option<u32>,

    /// Target committed frames (snapped). Default script path is 2 windows.
    #[arg(long)]
    pub(crate) total_frames: Option<u32>,

    /// Same-shot context frames before AV-safe snap (default 39). Ignored by --legacy-fl2va.
    #[arg(long, default_value_t = 39)]
    pub(crate) context_frames: u32,

    /// Overlap frames dropped on join (legacy WanGP FL2VA only; default 1).
    #[arg(long, default_value_t = 1)]
    pub(crate) overlap: u32,

    /// Diffusion steps per window (default 20).
    #[arg(long, default_value_t = 20)]
    pub(crate) steps: u32,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Increment seed per window (default: same seed every window).
    #[arg(long, default_value_t = false)]
    pub(crate) seed_inc: bool,

    /// Attention quality: fast=Sage (default), hq=FlashAttention-2.
    #[arg(long, value_enum, default_value_t = H3Quality::Fast)]
    pub(crate) quality: H3Quality,

    /// Multishot memory: keep last N shot-end frames (0 = stock continue).
    #[arg(long, default_value_t = 0)]
    pub(crate) memory_frames: u32,

    /// Multishot memory: persistent start anchor (0/1).
    #[arg(long, default_value_t = 0)]
    pub(crate) anchor_frames: u32,

    /// Force TE vision memory re-encode (default on when memory bank + prompt).
    #[arg(long, default_value_t = false)]
    pub(crate) te_memory: bool,

    /// Disable TE vision memory even if the bank is tracked.
    #[arg(long, default_value_t = false)]
    pub(crate) no_te_memory: bool,

    /// Pack memory bank as DiT --ref-image (needs Ref2VA weights + allow-keyframe-refs).
    #[arg(long, default_value_t = false)]
    pub(crate) dit_memory_refs: bool,

    /// Keep per-window decode/latents under work dir.
    #[arg(long, default_value_t = false)]
    pub(crate) keep_segs: bool,

    /// Compile freeform prompt into open IR field order (default on).
    #[arg(long, default_value_t = true)]
    pub(crate) compile_ir: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_compile_ir: bool,

    /// Freeze-prefix continue (copy tail into the target + 0/1 masks).
    #[arg(long, default_value_t = false)]
    pub(crate) legacy_masked_av: bool,

    /// Explicit WanGP last-RGB COND stitch (not native_guide / masked_av).
    #[arg(long, default_value_t = false)]
    pub(crate) legacy_fl2va: bool,

    /// DiT weights override (default FL2VA; Eros Ref2VA when --ref-mod / --ref-image).
    #[arg(long, value_name = "PATH")]
    pub(crate) weights: Option<PathBuf>,

    /// Validate and print the resolved plan without launching.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 upscale")]
#[command(about = "Post-decode upscale (RTX / LBH latent refine / Video2X / ffmpeg)")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3UpscaleArgs {
    /// Input MP4 (H3 deliverable).
    #[arg(long, value_name = "PATH")]
    pub(crate) input: Option<PathBuf>,

    /// Output MP4 path (default: <input>_up_xN.mp4).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Backend: auto (rtx→video2x→ffmpeg), rtx, latent (LBH refine), latent-preview, video2x, ffmpeg.
    #[arg(long, value_enum, default_value_t = H3UpscaleBackend::Auto)]
    pub(crate) backend: H3UpscaleBackend,

    /// Scale factor (RTX native 2 or 4; default 2). Ignored when --canvas / W×H set.
    /// Latent refine ignores scale and defaults to --canvas 720p.
    #[arg(long, default_value_t = 2)]
    pub(crate) scale: u32,

    /// Product target canvas: 480p | 720p | 1080p (1920×1088 ~2.0 MP). Aliases: 0.4 / 0.9 / 2.0.
    /// Latent refine is a DiT sample: 480p/720p only. 1080p needs latent-preview or RTX.
    #[arg(long, value_enum, value_name = "PRESET")]
    pub(crate) canvas: Option<H3CanvasPreset>,

    /// Target width (optional; prefer --canvas).
    #[arg(long)]
    pub(crate) width: Option<u32>,

    /// Target height (optional; prefer --canvas).
    #[arg(long)]
    pub(crate) height: Option<u32>,

    /// Ref2VA still(s) for `--backend latent` when `--mode ref2va` (ganloss stage-2 keeps Picture 1).
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_image: Vec<PathBuf>,

    /// Ref2VA audio clip(s) for `--backend latent` when `--mode ref2va` (ganloss stage-2 keeps Audio 1).
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_audio: Vec<PathBuf>,

    /// Saved RefMod for Eros stage-2 / latent refine (`name` or `name:strength[:copies]`). Repeatable, ≤4.
    #[arg(long, value_name = "NAME[:STRENGTH[:COPIES]]")]
    pub(crate) ref_mod: Vec<String>,

    /// Scene prompt for `--backend latent` refine (required). Same beat as the source clip.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read refine prompt from a UTF-8 text file (overrides --prompt when set).
    #[arg(long, value_name = "PATH")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Refine cond mode. Default t2va (prompt only). Use i2v/fl2va with --first-frame/--last-frame.
    #[arg(long, value_enum, default_value_t = H3Mode::T2va)]
    pub(crate) mode: H3Mode,

    /// First-frame still for refine I2V/FL2VA.
    #[arg(long, value_name = "PATH")]
    pub(crate) first_frame: Option<PathBuf>,

    /// Last-frame still for refine FL2VA.
    #[arg(long, value_name = "PATH")]
    pub(crate) last_frame: Option<PathBuf>,

    /// Attention quality for refine: fast=Sage (default), hq=FlashAttention-2.
    #[arg(long, value_enum, default_value_t = H3Quality::Fast)]
    pub(crate) quality: H3Quality,

    /// Turbo LoRA on the refine model stack (sampler stays official LBH euler + ManualSigmas).
    #[arg(long, value_enum, default_value_t = H3Turbo::Off)]
    pub(crate) turbo: H3Turbo,

    /// Turbo LoRA strength.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) turbo_strength: f64,

    /// fal MiniMax-H3-Realism-People LoRA on the refine stack. Trigger `r34l1sm` auto-prepended.
    #[arg(long, default_value_t = false)]
    pub(crate) realism: bool,

    /// Realism People LoRA strength.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) realism_strength: f64,

    /// Enable Saganaki Sol-Attn compose on the refine stack.
    #[arg(long, default_value_t = false)]
    pub(crate) sol: bool,

    /// Disable Sol even if a preset would enable it.
    #[arg(long, default_value_t = false)]
    pub(crate) no_sol: bool,

    /// Opt-in Ref2VA Visual Sparse Attention on the refine stack. Mutex with --sol / --cache.
    #[arg(long, default_value_t = false)]
    pub(crate) vsa: bool,

    /// VSA video-tile sparsity (keep ratio = 1 - sparsity). Default 0.75.
    #[arg(long, default_value_t = 0.75)]
    pub(crate) vsa_sparsity: f64,

    /// VSA gate filename under checkpoints/loras/.
    #[arg(long, value_name = "NAME", default_value = "fasth3_vsa_gate.safetensors")]
    pub(crate) vsa_gate: String,

    /// Step cache on the refine stack: off | spectrum | easy | fbc.
    #[arg(long, value_enum, default_value_t = H3Cache::Off)]
    pub(crate) cache: H3Cache,

    /// DiT weight quant pack for refine.
    #[arg(long, value_enum, default_value_t = H3DitQuant::Int8)]
    pub(crate) dit_quant: H3DitQuant,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Refine RNG seed.
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Used only with --denoise (KSampler path). Default LBH refine uses ManualSigmas, not this.
    #[arg(long, default_value_t = 8)]
    pub(crate) steps: u32,

    /// Optional KSampler denoise instead of the official LBH ManualSigmas schedule.
    #[arg(long)]
    pub(crate) denoise: Option<f64>,

    /// Official LBH 3-step refine schedule. Default: 0.9035, 0.6316, 0.3158, 0.0000.
    #[arg(long, value_name = "CSV")]
    pub(crate) refine_sigmas: Option<String>,

    /// Compile freeform refine prompt into open IR field order (default on).
    #[arg(long, default_value_t = true)]
    pub(crate) compile_ir: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_compile_ir: bool,

    /// DiT weights override for refine.
    #[arg(long, value_name = "PATH")]
    pub(crate) weights: Option<PathBuf>,

    /// Pro node quality (default ULTRA; also HIGH/MEDIUM/LOW).
    #[arg(long, default_value = "ULTRA")]
    pub(crate) rtx_quality: String,

    /// Video2X RealESRGAN model name.
    #[arg(long, default_value = "realesrgan-plus")]
    pub(crate) video2x_model: String,

    /// Optional audio to mux after video-only upscale.
    #[arg(long, value_name = "PATH")]
    pub(crate) audio: Option<PathBuf>,

    /// Print backend availability and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) check: bool,

    /// Validate and print the resolved plan without launching.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 loop")]
#[command(about = "Multi-scene Context Loop production")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3LoopArgs {
    /// Scene plan JSON (`gemmy-h3-loop-v1`).
    #[arg(long, value_name = "PATH")]
    pub(crate) plan: PathBuf,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Review each scene with `gemmy analyze` (12B) before advancing.
    #[arg(long, default_value_t = false)]
    pub(crate) review: bool,

    /// Regenerate the current pending/reviewed scene (re-reads the plan prompt).
    #[arg(long, default_value_t = false)]
    pub(crate) retry: bool,

    /// Regenerate the current scene with a new seed.
    #[arg(long, default_value_t = false)]
    pub(crate) reroll: bool,

    /// Accept the current reviewed scene and continue.
    #[arg(long, default_value_t = false)]
    pub(crate) approve: bool,

    /// Stop after the current scene and assemble accepted segments only.
    #[arg(long, default_value_t = false)]
    pub(crate) stop: bool,

    /// Resume at this 1-based scene index.
    #[arg(long, value_name = "N")]
    pub(crate) start_scene: Option<u32>,

    /// Inclusive scene range `A:B` (1-based).
    #[arg(long, value_name = "A:B")]
    pub(crate) scene_range: Option<String>,

    /// Final assembled MP4 (default: beside the plan).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum ShortfilmPhase {
    /// Identity-locked stills (if needed) then Ref2VA Eros video / loop plan.
    #[default]
    #[value(name = "all")]
    All,
    /// Character sheet → identity-edit panels. One-window also writes unlabeled ganloss board.
    #[value(name = "stills")]
    Stills,
    /// Ref2VA Eros from an existing sheet + board (no stills).
    #[value(name = "video")]
    Video,
    /// Resume-safe Hugging Face download of named Eros BF16 + INT8 ConvRot twin.
    #[value(name = "download")]
    Download,
}

impl ShortfilmPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Stills => "stills",
            Self::Video => "video",
            Self::Download => "download",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 shortfilm")]
#[command(about = "Short Film Director + 6-grid Ref2VA Eros pipeline (separate from default I2V)")]
#[command(long_about = crate::host::help::VIDEO_H3_SHORTFILM_HELP)]
pub(crate) struct H3ShortfilmArgs {
    /// Optional identity sheet. Pack stills only — never a second DiT ref.
    #[arg(long, value_name = "PATH")]
    pub(crate) character_sheet: Option<PathBuf>,

    /// Six-panel (or grid) storyboard. Sole Ref2VA Picture. Never an H3 first-frame.
    #[arg(long, value_name = "PATH")]
    pub(crate) storyboard: Option<PathBuf>,

    /// Existing identity-locked still panels (compose a board when --storyboard is omitted).
    #[arg(long, value_name = "PATH")]
    pub(crate) panel: Vec<PathBuf>,

    /// Per-panel Krea identity-edit prompts (repeatable). Reused if fewer than --grid.
    #[arg(long, value_name = "TEXT")]
    pub(crate) panel_prompt: Vec<String>,

    /// Scene / film brief. Compiled into official Ref2VA six-section IR.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read prompt from a UTF-8 text file (overrides --prompt when set).
    #[arg(long, value_name = "PATH")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Panel count / board layout: 6 (3×2 default), 9 (3×3), 16 (4×4).
    #[arg(long, default_value_t = 6)]
    pub(crate) grid: u32,

    /// Pipeline phase.
    #[arg(long, value_enum, default_value_t = ShortfilmPhase::All)]
    pub(crate) phase: ShortfilmPhase,

    /// Output MP4 (video phase).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Work / stills directory (panels + board + loop plan).
    #[arg(long, value_name = "DIR")]
    pub(crate) out_dir: Option<PathBuf>,

    /// One-window duration seconds @ 24 fps (snapped). Ganloss graph widget is 10.
    #[arg(long, default_value_t = 10.0)]
    pub(crate) duration: f64,

    /// Eros TURBO ref2va steps. Parsed from ganloss storyboard graph
    /// (`BasicScheduler` simple, 8, denoise 1). Not stock H3 20-step.
    /// Pack Contex Loop is stock FL2VA with per-shot 15 — not this DiT.
    #[arg(long, default_value_t = 8)]
    pub(crate) steps: u32,

    /// `1` = ganloss one-window (one 6-panel as Picture 1).
    /// `N>1` = pack path: one still per scene, hard-cut `h3 loop`.
    #[arg(long, default_value_t = 1)]
    pub(crate) scenes: u32,

    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Opt-in Ref2VA Visual Sparse Attention on Eros stages. Mutex with --sol / --cache.
    #[arg(long, default_value_t = false)]
    pub(crate) vsa: bool,

    /// VSA video-tile sparsity (keep ratio = 1 - sparsity). Default 0.75.
    #[arg(long, default_value_t = 0.75)]
    pub(crate) vsa_sparsity: f64,

    /// VSA gate filename under checkpoints/loras/.
    #[arg(long, value_name = "NAME", default_value = "fasth3_vsa_gate.safetensors")]
    pub(crate) vsa_gate: String,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Write the loop plan JSON here (default: beside output / out-dir).
    #[arg(long, value_name = "PATH")]
    pub(crate) plan_out: Option<PathBuf>,

    /// Validate and print the resolved pipeline plan without launching workers.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 face-refine")]
#[command(about = "Optional H3 FaceRefine post pass on a finished MP4")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3FaceRefineArgs {
    /// Input MP4 (finished segment or deliverable).
    #[arg(long, value_name = "PATH")]
    pub(crate) input: PathBuf,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Output MP4 path.
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3InterpolateFps {
    #[value(name = "23.976")]
    F23976,
    #[value(name = "25")]
    F25,
    #[value(name = "29.97")]
    F2997,
    #[value(name = "30")]
    F30,
    #[default]
    #[value(name = "48")]
    F48,
    #[value(name = "50")]
    F50,
    #[value(name = "59.94")]
    F5994,
    #[value(name = "60")]
    F60,
    #[value(name = "90")]
    F90,
    #[value(name = "120")]
    F120,
}

impl H3InterpolateFps {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::F23976 => "23.976",
            Self::F25 => "25",
            Self::F2997 => "29.97",
            Self::F30 => "30",
            Self::F48 => "48",
            Self::F50 => "50",
            Self::F5994 => "59.94",
            Self::F60 => "60",
            Self::F90 => "90",
            Self::F120 => "120",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3InterpolateEngine {
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "native")]
    Native,
    #[value(name = "cascade")]
    Cascade,
}

impl H3InterpolateEngine {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Native => "native",
            Self::Cascade => "cascade",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3InterpolateQuality {
    #[value(name = "auto")]
    Auto,
    #[default]
    #[value(name = "max")]
    Max,
    #[value(name = "best")]
    Best,
    #[value(name = "good")]
    Good,
}

impl H3InterpolateQuality {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Max => "max",
            Self::Best => "best",
            Self::Good => "good",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3InterpolateCodec {
    #[default]
    #[value(name = "h264")]
    H264,
    #[value(name = "h264-nvenc")]
    H264Nvenc,
    #[value(name = "h265")]
    H265,
    #[value(name = "h265-nvenc")]
    H265Nvenc,
    #[value(name = "av1")]
    Av1,
    #[value(name = "av1-nvenc")]
    Av1Nvenc,
    #[value(name = "prores")]
    Prores,
}

impl H3InterpolateCodec {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::H264Nvenc => "h264-nvenc",
            Self::H265 => "h265",
            Self::H265Nvenc => "h265-nvenc",
            Self::Av1 => "av1",
            Self::Av1Nvenc => "av1-nvenc",
            Self::Prores => "prores",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 interpolate")]
#[command(about = "DLSS Frame Generation post on a finished H3 MP4")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3InterpolateArgs {
    /// Input MP4 (finished generate / loop / upscale deliverable).
    #[arg(long, value_name = "PATH")]
    pub(crate) input: Option<PathBuf>,

    /// Output MP4 (default: `<stem>_<fps>fps.mp4` beside the input).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Target FPS. Must not exceed 6× the source rate.
    #[arg(long, value_enum, default_value_t = H3InterpolateFps::F48)]
    pub(crate) fps: H3InterpolateFps,

    /// DLSSG engine: auto (native grid then cascade) | native | cascade.
    #[arg(long, value_enum, default_value_t = H3InterpolateEngine::Auto)]
    pub(crate) engine: H3InterpolateEngine,

    /// Temp encode quality inside the DLSS node.
    #[arg(long, value_enum, default_value_t = H3InterpolateQuality::Max)]
    pub(crate) quality: H3InterpolateQuality,

    /// Temp encode codec inside the DLSS node.
    #[arg(long, value_enum, default_value_t = H3InterpolateCodec::H264)]
    pub(crate) codec: H3InterpolateCodec,

    /// Print NVIDIA runtime / Git LFS DLL presence and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) check: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 sprites")]
#[command(about = "PixelForge keyed sprite loop, sheet, and GIF from an H3 clip")]
#[command(long_about = crate::host::help::VIDEO_H3_HELP)]
pub(crate) struct H3SpritesArgs {
    /// Input MP4 (idle-pin or action clip).
    #[arg(long, value_name = "PATH")]
    pub(crate) input: PathBuf,

    /// Output directory (keyed frames + sheet + GIF + optional atlas).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// Keep at most this many frames after decimate (0 in the graph means no cap; CLI default 48).
    #[arg(long, default_value_t = 48)]
    pub(crate) frames: u32,

    /// Keep every Nth source frame (2 turns 24 fps H3 into 12 fps sprites).
    #[arg(long, default_value_t = 2)]
    pub(crate) every_nth: u32,

    /// Chroma key color (`auto`, `#00FF00`, `#FF00FF`, …).
    #[arg(long, default_value = "auto")]
    pub(crate) key_color: String,

    /// Trim to a seamless cycle (PixelForge loop auto).
    #[arg(long, default_value_t = false)]
    pub(crate) r#loop: bool,

    /// Also write atlas.json next to the keyed frames.
    #[arg(long, default_value_t = false)]
    pub(crate) atlas: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Default)]
pub(crate) enum H3CropMode {
    /// One static crop covering the subject's whole travel.
    #[default]
    Combined,
    /// Constant-size crop that stays still until the subject would leave it.
    Tracked,
}

impl H3CropMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Combined => "combined",
            Self::Tracked => "tracked",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "h3 edit")]
#[command(about = "Masked H3 region replace (SAM3 track + Eros crop sample + uncrop)")]
#[command(long_about = crate::host::help::VIDEO_H3_EDIT_HELP)]
pub(crate) struct H3EditArgs {
    /// Source plate (mp4). Camera, motion, and soundtrack stay; only the mask changes.
    #[arg(long, value_name = "PATH")]
    pub(crate) video: PathBuf,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Replacement identity still (repeatable). Picture 1 in the Ref2VA prompt.
    #[arg(long, value_name = "PATH")]
    pub(crate) ref_image: Vec<PathBuf>,

    /// SAM3.1 text to track (e.g. head, female). Required unless --mask is set.
    #[arg(long, value_name = "TEXT")]
    pub(crate) mask_prompt: Option<String>,

    /// Precomputed mask video or image sequence (white = edit). Skips SAM3.
    #[arg(long, value_name = "PATH")]
    pub(crate) mask: Option<PathBuf>,

    /// SAM3 object indices to keep (comma-separated). Default 0.
    #[arg(long, default_value = "0")]
    pub(crate) object_id: String,

    /// SAM3 max objects (0 = internal cap). Default 1.
    #[arg(long, default_value_t = 1)]
    pub(crate) max_objects: u32,

    /// SAM3 detection threshold (video: 0.5).
    #[arg(long, default_value_t = 0.5)]
    pub(crate) confidence: f64,

    /// Crop planner: combined (default) or tracked.
    #[arg(long, value_enum, default_value_t = H3CropMode::Combined)]
    pub(crate) crop_mode: H3CropMode,

    /// Padding around the subject bbox (video: 1.5).
    #[arg(long, default_value_t = 1.5)]
    pub(crate) crop_scale: f64,

    /// Enlarge the crop toward this megapixel count before sampling (video: 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) crop_mp: f64,

    /// Uncrop feather in pixels.
    #[arg(long, default_value_t = 8)]
    pub(crate) feather: u32,

    /// Dilate the SAM3 mask before crop (pixels).
    #[arg(long, default_value_t = 0)]
    pub(crate) mask_grow: u32,

    /// Copy the cleaned mask MP4 here after track.
    #[arg(long, value_name = "PATH")]
    pub(crate) mask_out: Option<PathBuf>,

    /// Copy the tinted overlay MP4 here after track.
    #[arg(long, value_name = "PATH")]
    pub(crate) mask_overlay_out: Option<PathBuf>,

    /// Run `gemmy analyze` on the overlay before Eros (default on).
    #[arg(long, default_value_t = true)]
    pub(crate) verify_mask: bool,

    /// Skip the analyze gate (still writes the overlay).
    #[arg(long, default_value_t = false)]
    pub(crate) no_verify_mask: bool,

    /// Stop after track + overlay (+ verify). No Eros. Resume with --mask.
    #[arg(long, default_value_t = false)]
    pub(crate) stop_after_mask: bool,

    /// What to put in the mask (identity / outfit). Compiled to video-editing IR.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Read prompt from a UTF-8 text file.
    #[arg(long, value_name = "PATH")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Duration seconds at 24 fps (0 = from the source video). Snapped to 17k+5.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) duration: f64,

    /// Exact frame count before snap (0 = from duration / source).
    #[arg(long, default_value_t = 0)]
    pub(crate) frames: u32,

    /// Eros steps (ganloss / Veteran AI: euler simple 8).
    #[arg(long, default_value_t = 8)]
    pub(crate) steps: u32,

    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Output MP4.
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// SAM3.1 checkpoint basename (default sam3.1_multiplex_fp16.safetensors).
    #[arg(long, value_name = "NAME")]
    pub(crate) sam3_ckpt: Option<String>,

    #[arg(long, default_value_t = true)]
    pub(crate) compile_ir: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) no_compile_ir: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) keep_work: bool,
}

#[derive(Parser, Debug)]
#[command(name = "h3 refmod")]
#[command(about = "Create, list, or inspect MiniMax-H3 RefMods")]
#[command(long_about = crate::host::help::VIDEO_H3_REFMOD_HELP)]
pub(crate) struct H3RefModArgs {
    #[command(subcommand)]
    pub(crate) command: H3RefModCommand,
}

#[derive(Subcommand, Debug)]
pub(crate) enum H3RefModCommand {
    /// Encode stills/video/audio through the H3 VAE into a reusable .safetensors.
    Create(H3RefModCreateArgs),
    /// List saved RefMods.
    List(H3RefModListArgs),
    /// Print metadata for one saved RefMod.
    Inspect(H3RefModInspectArgs),
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct H3RefModCreateArgs {
    /// Saved name (dropdown id). Default: folder basename or `character`.
    #[arg(long)]
    pub(crate) name: Option<String>,

    /// Folder of stills and/or video clips.
    #[arg(long, value_name = "DIR")]
    pub(crate) folder: Option<PathBuf>,

    #[command(flatten)]
    pub(crate) vae_select: H3VaeSelect,

    /// Still image (repeatable).
    #[arg(long, value_name = "PATH")]
    pub(crate) image: Vec<PathBuf>,

    /// Video clip (repeatable).
    #[arg(long, value_name = "PATH")]
    pub(crate) video: Vec<PathBuf>,

    /// Optional audio clip (experimental; does not clone a speaker).
    #[arg(long, value_name = "PATH")]
    pub(crate) audio: Option<PathBuf>,

    /// Copy the saved file here (default: checkpoints/refmods/<name>.safetensors).
    #[arg(long, short = 'o', value_name = "PATH")]
    pub(crate) output: Option<PathBuf>,

    /// encode (Full Reference / identity) or training (Compressed Reference).
    #[arg(long, default_value = "encode")]
    pub(crate) mode: String,

    /// Metadata label only (identity / style / generic / …).
    #[arg(long, default_value = "identity")]
    pub(crate) concept_type: String,

    /// Short-edge encode resolution.
    #[arg(long, default_value_t = 1024)]
    pub(crate) ref_resolution: u32,

    /// Token budget (0 disables). Identity default 8192.
    #[arg(long, default_value_t = 8192)]
    pub(crate) max_tokens: u32,

    /// Optional description stored in the file header.
    #[arg(long, default_value = "")]
    pub(crate) description: String,

    /// Subfolder under the RefMod library.
    #[arg(long, default_value = "")]
    pub(crate) subfolder: String,

    #[arg(long, default_value_t = false)]
    pub(crate) dry_run_plan: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct H3RefModListArgs {
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct H3RefModInspectArgs {
    /// RefMod name or path to a .safetensors file.
    pub(crate) name: String,

    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,

    #[arg(long, short = 'v', default_value_t = false)]
    pub(crate) verbose: bool,
}

pub(crate) fn parse_sla_fixed(raw: &str) -> Result<u8, String> {
    match raw.trim() {
        "1" => Ok(1),
        "3" => Ok(3),
        "5" => Ok(5),
        "10" => Ok(10),
        other => Err(format!(
            "--sla-fixed must be 1, 3, 5, or 10 (got {other})"
        )),
    }
}
