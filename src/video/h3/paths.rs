//! Resolve MiniMax-H3 **runtime** (code) vs **checkpoints** (weights) roots.

use super::args::{H3VaeSelect, H3VideoVae};
use crate::host::config::H3Config;
use crate::host::paths::{models_rel, runtime_path};
use crate::host::util::absolute_path;
use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

thread_local! {
    /// Optional override for the external checkpoints tree.
    static CHECKPOINTS_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static COMFY_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static EROS_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static STOCK_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static SINGULARITY_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    /// Test-only: `None` = read env/config; `Some(None)` = force missing; `Some(Some(k))` = force key.
    static TYPESAFE_API_KEY_OVERRIDE: RefCell<Option<Option<String>>> = const { RefCell::new(None) };
    static JEV_SDK_PYTHON_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Apply saved roots. Env vars still win inside each resolver.
///
/// Checkpoints: `h3 setup`, then Gemmy `model_paths.minimax_h3`.
/// Code stays under `runtimes/minimax-h3` unless `GEMMY_H3_ROOT` is set.
pub(crate) fn apply_roots_from_config(config: &H3Config) {
    let checkpoints = config
        .setup
        .checkpoints
        .clone()
        .or_else(|| config.model_paths.minimax_h3.clone());
    set_checkpoints_override(checkpoints);
    set_root_override(&COMFY_OVERRIDE, config.setup.comfy.clone());
    set_root_override(&EROS_OVERRIDE, config.setup.eros.clone());
    set_root_override(&STOCK_OVERRIDE, config.setup.ref2va_stock.clone());
    set_root_override(&SINGULARITY_OVERRIDE, config.setup.singularity.clone());
}

fn set_root_override(
    slot: &'static std::thread::LocalKey<RefCell<Option<PathBuf>>>,
    path: Option<PathBuf>,
) {
    slot.with(|cell| {
        *cell.borrow_mut() = path;
    });
}

pub(crate) fn set_checkpoints_override(path: Option<PathBuf>) {
    CHECKPOINTS_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = path;
    });
}

/// MiniMax-H3 **code** root (package, scripts, Official FL2VA modules, uv env).
///
/// Order:
/// 1. `GEMMY_H3_ROOT` (dev override only)
/// 2. `runtimes/minimax-h3` (production default — fully internalized)
pub(crate) fn h3_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_ROOT") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    runtime_root()
}

/// Checkpoints root (DiT / TE / VAE weights). Multi-GB; never vendored into the repo.
///
/// Order:
/// 1. `GEMMY_H3_CHECKPOINTS`
/// 2. h3 setup checkpoints, else Gemmy `model_paths.minimax_h3`
/// 3. sibling `../minimax-h3/checkpoints` when that directory exists
/// 4. `<models_root>\minimax-h3` when a models root exists
/// 5. `%USERPROFILE%\h3-models` on a machine with none of the above
pub(crate) fn checkpoints_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_CHECKPOINTS") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Some(path) = CHECKPOINTS_OVERRIDE.with(|c| c.borrow().clone()) {
        return path;
    }
    if let Some(sibling) = crate::host::paths::repo_root()
        .parent()
        .map(|p| p.join("minimax-h3").join("checkpoints"))
        .filter(|p| p.is_dir())
    {
        return sibling;
    }
    // Also accept a full project tree pointed at by models_rel with nested checkpoints.
    let models_proj = models_rel("minimax-h3");
    let nested = models_proj.join("checkpoints");
    if nested.is_dir() {
        return nested;
    }
    if crate::host::paths::models_root().is_some() {
        // weights-only layout: models_root/minimax-h3/{diffusion_models,text_encoders,vae}
        return models_proj;
    }
    crate::host::paths::user_h3_models()
}

pub(crate) fn runtime_root() -> PathBuf {
    runtime_path("minimax-h3")
}

pub(crate) fn worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_generate.py")
}

/// Comfy in-process worker (same request.json + engine fields).
pub(crate) fn comfy_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_comfy_generate.py")
}

/// Resolve ComfyUI root for the H3 Comfy engine.
///
/// Production default is **internal only**: `runtimes/minimax-h3/ComfyUI`
/// (via `h3_root()/ComfyUI`). External research trees are never the default.
/// Optional override: `GEMMY_H3_COMFY` (dev A/B only).
pub(crate) fn comfy_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_COMFY") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Some(path) = COMFY_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return path;
    }
    // Production: fully internalized under the H3 runtime.
    h3_root().join("ComfyUI")
}

pub(crate) fn comfy_runner_script() -> PathBuf {
    comfy_root().join("run_h3_workflow.py")
}

pub(crate) fn turbo_lora_v4() -> PathBuf {
    checkpoints_root().join("loras/minimax_h3_turbo_v4_step600_ema.safetensors")
}

/// fal MiniMax-H3-Realism-People-LoRA (T2V/I2V/R2V). Trigger: `r34l1sm`.
pub(crate) fn realism_people_lora() -> PathBuf {
    checkpoints_root().join("loras/h3-realism-people-t2v-i2v-r2v.safetensors")
}

/// FastH3 VSA gate transplant for `--vsa` (external; not a LoRA).
pub(crate) const VSA_GATE_DEFAULT: &str = "fasth3_vsa_gate.safetensors";

/// Author 009jev: 4 sigmas-from-steps + `sample_res_multistep`.
pub(crate) const JEV_STEPS: u32 = 4;
pub(crate) const JEV_INITIAL_POLICY: &str = "jev_first";

/// Dedicated TypeSafe/Jev SDK interpreter (not Comfy's Python).
pub(crate) fn jev_sdk_python() -> PathBuf {
    if let Some(path) = JEV_SDK_PYTHON_OVERRIDE.with(|c| c.borrow().clone()) {
        return path;
    }
    if let Ok(raw) = std::env::var("GEMMY_H3_JEV_SDK_PYTHON") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    h3_root()
        .join("jev-sdk")
        .join(".venv")
        .join("Scripts")
        .join("python.exe")
}

pub(crate) fn jev_native_sla_dir() -> PathBuf {
    comfy_root()
        .join("custom_nodes")
        .join("ComfyUI-MiniMax-H3-009jev")
}

/// `TYPESAFE_API_KEY` for 009jev: process env, then config `env_overrides`, then repo `.env`.
pub(crate) fn typesafe_api_key() -> Option<String> {
    if let Some(over) = TYPESAFE_API_KEY_OVERRIDE.with(|c| c.borrow().clone()) {
        return over.filter(|s| !s.trim().is_empty());
    }
    if let Ok(v) = std::env::var("TYPESAFE_API_KEY") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    if let Some(v) = crate::host::config::H3Config::load()
        .ok()
        .and_then(|cfg| cfg.env_overrides.get("TYPESAFE_API_KEY").cloned())
    {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    let dotenv = crate::host::paths::repo_root().join(".env");
    if let Ok(text) = fs::read_to_string(dotenv) {
        if let Some(v) = parse_typesafe_key_from_env_file(&text) {
            return Some(v);
        }
    }
    None
}

pub(crate) fn typesafe_api_key_present() -> bool {
    typesafe_api_key().is_some()
}

pub(crate) fn parse_typesafe_key_from_env_file(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let rest = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = rest.split_once('=') else {
            continue;
        };
        if key.trim() != "TYPESAFE_API_KEY" {
            continue;
        }
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_string();
        if !value.is_empty() {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
pub(crate) fn set_typesafe_api_key_override(value: Option<Option<&str>>) {
    TYPESAFE_API_KEY_OVERRIDE.with(|c| {
        *c.borrow_mut() = value.map(|inner| inner.map(str::to_string));
    });
}

#[cfg(test)]
pub(crate) fn set_jev_sdk_python_override(path: Option<PathBuf>) {
    JEV_SDK_PYTHON_OVERRIDE.with(|c| *c.borrow_mut() = path);
}

pub(crate) fn vsa_gate_path(name: &str) -> PathBuf {
    let file = name.trim();
    let file = if file.is_empty() {
        VSA_GATE_DEFAULT
    } else {
        file
    };
    checkpoints_root().join("loras").join(file)
}

pub(crate) fn continue_script() -> PathBuf {
    runtime_root().join(r"scripts\h3_fl2va_continue.py")
}

pub(crate) fn comfy_continue_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_comfy_continue.py")
}

pub(crate) fn upscale_script() -> PathBuf {
    runtime_root().join(r"scripts\h3_upscale.py")
}

pub(crate) fn latent_upscale_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_latent_upscale.py")
}

pub(crate) fn face_refine_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_face_refine.py")
}

pub(crate) fn edit_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_comfy_edit.py")
}

pub(crate) const SAM3_CKPT: &str = "sam3.1_multiplex_fp16.safetensors";

/// SAM3.1 multiplex checkpoint (Comfy-Org/sam3.1).
///
/// Order: `GEMMY_SAM3_CHECKPOINT` file, then `GEMMY_SAM3_CHECKPOINTS` dir,
/// then `<h3 checkpoints>/checkpoints/`, then `F:\Models\sam3\checkpoints\`.
pub(crate) fn sam3_checkpoint_path(basename: &str) -> PathBuf {
    let name = if basename.trim().is_empty() {
        SAM3_CKPT
    } else {
        basename.trim()
    };
    if let Ok(raw) = std::env::var("GEMMY_SAM3_CHECKPOINT") {
        let path = PathBuf::from(raw.trim());
        if path.is_file() {
            return path;
        }
    }
    if let Ok(raw) = std::env::var("GEMMY_SAM3_CHECKPOINTS") {
        let dir = PathBuf::from(raw.trim());
        let cand = if dir.ends_with("checkpoints") {
            dir.join(name)
        } else {
            dir.join("checkpoints").join(name)
        };
        if cand.is_file() {
            return cand;
        }
    }
    let under_h3 = checkpoints_root().join("checkpoints").join(name);
    if under_h3.is_file() {
        return under_h3;
    }
    PathBuf::from(r"F:\Models\sam3\checkpoints").join(name)
}

pub(crate) fn sprites_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_sprites.py")
}

pub(crate) fn interpolate_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_interpolate.py")
}

pub(crate) fn refmod_worker_script() -> PathBuf {
    runtime_root().join("gemmy_h3_refmod.py")
}

/// Library of saved RefMod `.safetensors` (VAE-encoded H3 references).
///
/// Order: `GEMMY_H3_REFMODS`, then `<checkpoints>/refmods`.
pub(crate) fn refmods_dir() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_REFMODS") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    checkpoints_root().join("refmods")
}

/// Python that can import `minimax_h3` and run H3 scripts (runtime uv venv).
pub(crate) fn h3_python() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_PYTHON") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    h3_root().join(r".venv\Scripts\python.exe")
}

pub(crate) const DEFAULT_FL2VA_DIT: &str =
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
/// UNETLoader filename for stock Comfy-Org Ref2VA (speech `--dit stock`).
pub(crate) const STOCK_REF2VA_UNET: &str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors";
/// Stock Comfy-Org Ref2VA — **backup only** for video generate. Speech default.
pub(crate) const STOCK_REF2VA_DIT: &str =
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors";

/// Named TenStrip BF16 TURBO Ref2VA (ganloss / YouTube storyboard method).
/// ~40.2 GB — do **not** load on 16 GB (no disk offload).
pub(crate) const EROS_REF2VA_BF16: &str =
    "10Eros_Max_h3_TURBO_ref2va_beta2.safetensors";
/// Runnable INT8 ConvRot twin for this box (cicalooo). ~21 GB.
pub(crate) const EROS_REF2VA_INT8: &str =
    "10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors";
pub(crate) const EROS_BF16_REPO: &str = "TenStrip/10Eros-Max";
pub(crate) const EROS_INT8_REPO: &str = "cicalooo/10Eros-Max-h3-int8-convrot";
/// Ganloss Eros TURBO steps (`BasicScheduler` simple / 8). Not stock FL2VA 20.
pub(crate) const EROS_REF2VA_STEPS: u32 = 8;

/// Eros Ref2VA weights, isolated from the stock FL2VA tree.
///
/// Order: `GEMMY_H3_EROS_CHECKPOINTS`, h3 setup `--eros`,
/// `F:\Models\minimax-h3-eros` when that directory already exists,
/// otherwise `<checkpoints>\eros`.
pub(crate) fn eros_checkpoints_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_EROS_CHECKPOINTS") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Some(path) = EROS_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return path;
    }
    let legacy = PathBuf::from(r"F:\Models\minimax-h3-eros");
    if legacy.is_dir() {
        return legacy;
    }
    checkpoints_root().join("eros")
}

pub(crate) fn eros_dit_dir() -> PathBuf {
    eros_checkpoints_root().join("diffusion_models")
}

/// Named 40.2 GB BF16 file (download/keep; not the generate default).
pub(crate) fn eros_ref2va_bf16_path() -> PathBuf {
    eros_dit_dir().join(EROS_REF2VA_BF16)
}

/// Runnable generate default for product `--mode ref2va` and shortfilm.
pub(crate) fn eros_ref2va_int8_path() -> PathBuf {
    eros_dit_dir().join(EROS_REF2VA_INT8)
}

/// WarmBloodAban Singularity Ref2VA INT8. Opt-in `--dit singularity` only.
/// The AI Brief dual-sample workflow and the video description both load this
/// full file (not the pruned sibling on the same repo).
pub(crate) const SINGULARITY_REF2VA_INT8: &str =
    "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors";

/// Singularity DiT root. Isolated from Eros and stock FL2VA.
///
/// Order: `GEMMY_H3_SINGULARITY_CHECKPOINTS`, h3 setup `--singularity`,
/// `F:\Models\minimax-h3-singularity` when that directory already exists,
/// otherwise `<checkpoints>\singularity`.
pub(crate) fn singularity_checkpoints_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_SINGULARITY_CHECKPOINTS") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Some(path) = SINGULARITY_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return path;
    }
    let legacy = PathBuf::from(r"F:\Models\minimax-h3-singularity");
    if legacy.is_dir() {
        return legacy;
    }
    checkpoints_root().join("singularity")
}

pub(crate) fn singularity_ref2va_int8_path() -> PathBuf {
    singularity_checkpoints_root()
        .join("diffusion_models")
        .join(SINGULARITY_REF2VA_INT8)
}

/// Parked stock Comfy-Org Ref2VA. Override: `GEMMY_H3_REF2VA_STOCK` or `h3 setup`.
pub(crate) fn stock_ref2va_backup_root() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_REF2VA_STOCK") {
        let path = PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            return if path.is_file() {
                path.parent()
                    .and_then(|p| p.parent())
                    .unwrap_or(&path)
                    .to_path_buf()
            } else {
                path
            };
        }
    }
    if let Some(path) = STOCK_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return path;
    }
    let legacy = PathBuf::from(r"G:\Models\minimax-h3-backup");
    if legacy.is_dir() {
        return legacy;
    }
    checkpoints_root().join("ref2va-stock")
}

pub(crate) fn stock_ref2va_backup_path() -> PathBuf {
    if let Ok(raw) = std::env::var("GEMMY_H3_REF2VA_STOCK") {
        let path = PathBuf::from(raw.trim());
        if path.is_file() {
            return path;
        }
        if !path.as_os_str().is_empty() {
            return path.join(STOCK_REF2VA_DIT);
        }
    }
    stock_ref2va_backup_root().join(STOCK_REF2VA_DIT)
}
pub(crate) const DEFAULT_TE: &str =
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
pub(crate) const DEFAULT_TE_TOK: &str = "text_encoders/qwen25_tokenizer";
pub(crate) const VIDEO_VAE_INT8_FILE: &str = "minimax_h3_video_vae_int8_convrot.safetensors";
pub(crate) const VIDEO_VAE_FP16_FILE: &str = "minimax_h3_video_vae_fp16.safetensors";
pub(crate) const AUDIO_VAE_FILE: &str = "minimax_h3_audio_vae_fp32.safetensors";
/// Product default: official Comfy-Org int8 video VAE.
pub(crate) const DEFAULT_VIDEO_VAE: &str = "vae/minimax_h3_video_vae_int8_convrot.safetensors";
pub(crate) const OPTIONAL_VIDEO_VAE_FP16: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
pub(crate) const DEFAULT_AUDIO_VAE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";

/// Resolved video VAE basename + path. Audio fp32 is always required separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedVideoVae {
    pub(crate) filename: String,
    pub(crate) path: PathBuf,
    pub(crate) fingerprint_id: String,
}

pub(crate) fn sanitize_vae_basename(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("--video-vae needs a filename under checkpoints/vae/");
    }
    let name = Path::new(trimmed)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim();
    if name.is_empty()
        || name.contains("..")
        || name.contains('/')
        || name.contains('\\')
        || name.contains(':')
    {
        bail!("--video-vae must be a basename under vae/ (got {raw:?})");
    }
    if name.ends_with(".safetensors") {
        Ok(name.to_string())
    } else {
        Ok(format!("{name}.safetensors"))
    }
}

pub(crate) fn video_vae_fingerprint_id(filename: &str) -> String {
    filename
        .strip_suffix(".safetensors")
        .unwrap_or(filename)
        .to_string()
}

pub(crate) fn sidecar_vae_to_filename(sidecar_vae: &str) -> String {
    let stem = Path::new(sidecar_vae.trim())
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(sidecar_vae.trim());
    if stem.ends_with(".safetensors") {
        stem.to_string()
    } else {
        format!("{stem}.safetensors")
    }
}

/// Resolve the official Comfy-Org video VAE (or a `--video-vae` override).
///
/// `--video-vae NAME` wins over `--vae`. When both flags are omitted and
/// `sidecar_vae` is set (continue/loop), reuse that sidecar stem. Otherwise
/// the product default is official int8. `--engine python` accepts fp16 only.
pub(crate) fn resolve_video_vae(
    select: &H3VaeSelect,
    sidecar_vae: Option<&str>,
    python_engine: bool,
    require_file: bool,
) -> Result<ResolvedVideoVae> {
    let filename = if let Some(name) = select.video_vae.as_deref() {
        sanitize_vae_basename(name)?
    } else if let Some(kind) = select.vae {
        match kind {
            H3VideoVae::Int8 => VIDEO_VAE_INT8_FILE.to_string(),
            H3VideoVae::Fp16 => VIDEO_VAE_FP16_FILE.to_string(),
        }
    } else if let Some(sid) = sidecar_vae.map(str::trim).filter(|s| !s.is_empty()) {
        sidecar_vae_to_filename(sid)
    } else {
        VIDEO_VAE_INT8_FILE.to_string()
    };

    if python_engine && filename != VIDEO_VAE_FP16_FILE {
        bail!(
            "--engine python accepts only the official fp16 video VAE ({VIDEO_VAE_FP16_FILE}). \
             Int8 + kitchen fused decode is Comfy-only. Use --vae fp16 or --engine comfy."
        );
    }

    let path = checkpoints_root().join("vae").join(&filename);
    if require_file && !path.is_file() {
        bail!(
            "video VAE not found: {}\n\
             Official Comfy-Org trio lives under checkpoints/vae/ \
             (int8 default, fp16 for --vae fp16 / --engine python). \
             See docs\\MODEL_LOCATIONS.md.",
            path.display()
        );
    }

    Ok(ResolvedVideoVae {
        fingerprint_id: video_vae_fingerprint_id(&filename),
        filename,
        path,
    })
}

pub(crate) fn read_sidecar_vae(mp4: &Path) -> Option<String> {
    let mut meta = mp4.to_path_buf();
    meta.set_extension("h3av.json");
    let text = fs::read_to_string(meta).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("vae")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

/// Resolve the pruned int8 ConvRot DiT partition for FL2VA or stock Ref2VA backup.
pub(crate) fn dit_weights_resolved(ref2va: bool) -> PathBuf {
    if ref2va {
        return stock_ref2va_backup_path();
    }
    checkpoints_root().join(DEFAULT_FL2VA_DIT)
}

pub(crate) fn default_dit_weights() -> PathBuf {
    dit_weights_resolved(false)
}

/// Default DiT for the product mode. `--mode ref2va` is Eros two-stage, not stock.
pub(crate) fn default_dit_weights_for_mode(mode: super::args::H3Mode) -> PathBuf {
    match mode {
        super::args::H3Mode::Ref2va => eros_ref2va_int8_path(),
        _ => dit_weights_resolved(false),
    }
}

pub(crate) fn required_weight_paths() -> Vec<(&'static str, PathBuf)> {
    let ck = checkpoints_root();
    vec![
        ("dit_fl2va", dit_weights_resolved(false)),
        ("text_encoder", ck.join(DEFAULT_TE)),
        ("tokenizer_dir", ck.join(DEFAULT_TE_TOK)),
        ("video_vae", ck.join(DEFAULT_VIDEO_VAE)),
        ("audio_vae", ck.join("vae").join(AUDIO_VAE_FILE)),
    ]
}

pub(crate) fn required_script_paths(root: &Path) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("h3_text_encode", root.join(r"scripts\h3_text_encode.py")),
        ("h3_vae_decode", root.join(r"scripts\h3_vae_decode.py")),
        ("h3_vae_encode", root.join(r"scripts\h3_vae_encode.py")),
        ("h3_prompt_ir", root.join(r"scripts\h3_prompt_ir.py")),
        (
            "h3_fl2va_continue",
            root.join(r"scripts\h3_fl2va_continue.py"),
        ),
        ("h3_upscale", root.join(r"scripts\h3_upscale.py")),
        ("minimax_h3_pkg", root.join(r"minimax_h3\__init__.py")),
        (
            "official_fl2va",
            root.join(r"MiniMax-H3-Official\FL2VA"),
        ),
    ]
}

pub(crate) fn ensure_exists(path: &Path, label: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    bail!(
        "{label} not found: {}\n\
         Code root: runtimes\\minimax-h3 (Comfy pack: runtimes\\minimax-h3\\ComfyUI).\n\
         Dev override only: GEMMY_H3_ROOT / GEMMY_H3_COMFY.\n\
         Weights: set GEMMY_H3_CHECKPOINTS / model_paths.minimax_h3, or see docs\\MODEL_LOCATIONS.md (MiniMax-H3).",
        path.display()
    );
}

pub(crate) fn abs(path: &Path) -> Result<PathBuf> {
    absolute_path(path).with_context(|| format!("resolve {}", path.display()))
}

/// Nightshift sets `GEMMY_LIVE_PREVIEW` to `<job>/out/live-preview.jpg`.
/// Forward it into the Comfy python child so headless `run_h3_workflow.py`
/// can write sampler JPEGs. Not a CLI flag.
pub(crate) fn forward_live_preview(cmd: &mut Command) {
    if let Ok(path) = std::env::var("GEMMY_LIVE_PREVIEW") {
        if !path.trim().is_empty() {
            cmd.env("GEMMY_LIVE_PREVIEW", path);
        }
    }
}

/// Forward extra DiT search roots so Comfy `UNETLoader` can see Eros, stock Ref2VA, and Singularity.
pub(crate) fn forward_extra_dit_roots(cmd: &mut Command) {
    let eros = eros_checkpoints_root();
    if eros.is_dir() {
        cmd.env("GEMMY_H3_EROS_CHECKPOINTS", eros.as_os_str());
    }
    let stock = stock_ref2va_backup_root();
    if stock.is_dir() {
        cmd.env("GEMMY_H3_REF2VA_STOCK", stock.as_os_str());
    }
    let singularity = singularity_checkpoints_root();
    if singularity.is_dir() {
        cmd.env("GEMMY_H3_SINGULARITY_CHECKPOINTS", singularity.as_os_str());
    }
}

/// Forward the product RefMod library so headless extra_model_paths and
/// MiniMaxH3RefModsLoader see the same folder as `h3 refmod`.
pub(crate) fn forward_refmods_dir(cmd: &mut std::process::Command) {
    cmd.env("GEMMY_H3_REFMODS", refmods_dir().as_os_str());
}

fn yaml_base_path(path: &Path) -> String {
    let mut s = path.display().to_string().replace('\\', "/");
    if !s.ends_with('/') {
        s.push('/');
    }
    s
}

/// Product `extra_model_paths.yaml` for persistent Comfy (:8188) and discovery.
///
/// Headless generate still writes a temp yaml in `run_h3_workflow.py`. This file
/// is what `python main.py` on :8188 reads. Speech `--dit stock` needs the G:
/// backup root; `--dit eros` needs the Eros root.
pub(crate) fn extra_model_paths_yaml() -> String {
    let mut out = String::new();
    out.push_str("# Product extra_model_paths for persistent Comfy (:8188).\n");
    out.push_str("# Regenerated by gemmy audio-h3 install / h3 install.\n");
    out.push_str("# Headless generate still writes a temp yaml via run_h3_workflow.py.\n");
    out.push_str("#\n");
    out.push_str("# Roots: GEMMY_H3_CHECKPOINTS (FL2VA/TE/VAE), GEMMY_H3_EROS_CHECKPOINTS,\n");
    out.push_str("# GEMMY_H3_REF2VA_STOCK (parked Comfy-Org Ref2VA for speech --dit stock),\n");
    out.push_str("# GEMMY_H3_SINGULARITY_CHECKPOINTS (opt-in --dit singularity),\n");
    out.push_str("# GEMMY_H3_REFMODS (saved VAE latents; default <checkpoints>/refmods).\n\n");
    let mods = refmods_dir();
    out.push_str("minimax_h3_refmods:\n");
    out.push_str(&format!("    base_path: {}\n", yaml_base_path(&mods)));
    out.push_str("    refmods: ./\n\n");
    out.push_str("minimax_h3_local:\n");
    out.push_str(&format!(
        "    base_path: {}\n",
        yaml_base_path(&checkpoints_root())
    ));
    out.push_str("    is_default: true\n");
    out.push_str("    diffusion_models: diffusion_models/\n");
    out.push_str("    text_encoders: text_encoders/\n");
    out.push_str("    vae: vae/\n");
    out.push_str("    loras: loras/\n");
    out.push_str("    refmods: refmods/\n");
    out.push_str("    latent_upscale_models: latent_upscale_models/\n");
    out.push_str("    ultralytics: ultralytics/\n");
    out.push_str("    ultralytics_bbox: ultralytics/bbox/\n\n");
    let eros = eros_checkpoints_root();
    if eros.is_dir() {
        out.push_str("minimax_h3_eros:\n");
        out.push_str(&format!("    base_path: {}\n", yaml_base_path(&eros)));
        out.push_str("    diffusion_models: diffusion_models/\n\n");
    }
    let stock = stock_ref2va_backup_root();
    if stock.is_dir() {
        out.push_str("minimax_h3_ref2va_stock:\n");
        out.push_str(&format!("    base_path: {}\n", yaml_base_path(&stock)));
        out.push_str("    diffusion_models: diffusion_models/\n");
    }
    let singularity = singularity_checkpoints_root();
    if singularity.is_dir() {
        out.push_str("minimax_h3_singularity:\n");
        out.push_str(&format!(
            "    base_path: {}\n",
            yaml_base_path(&singularity)
        ));
        out.push_str("    diffusion_models: diffusion_models/\n");
    }
    out
}

pub(crate) fn write_comfy_extra_model_paths() -> Result<PathBuf> {
    let path = comfy_root().join("extra_model_paths.yaml");
    fs::write(&path, extra_model_paths_yaml())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::super::args::{H3VaeSelect, H3VideoVae};
    use super::*;

    #[test]
    fn dotenv_parses_typesafe_key_and_ignores_comments() {
        let text = "# comment\nexport TYPESAFE_API_KEY=\"abc_test\"\nOTHER=nope\n";
        assert_eq!(
            parse_typesafe_key_from_env_file(text).as_deref(),
            Some("abc_test")
        );
        assert_eq!(parse_typesafe_key_from_env_file("# only\n"), None);
        assert_eq!(parse_typesafe_key_from_env_file("TYPESAFE_API_KEY=\n"), None);
    }

    #[test]
    fn extra_model_paths_yaml_includes_stock_and_eros_when_present() {
        let yaml = extra_model_paths_yaml();
        assert!(
            yaml.contains("minimax_h3_local:"),
            "missing local checkpoints section"
        );
        assert!(
            yaml.contains("diffusion_models: diffusion_models/"),
            "missing diffusion_models mapping"
        );
        assert!(
            yaml.contains("minimax_h3_refmods:"),
            "missing dedicated RefMod extra_model_paths section"
        );
        assert!(
            yaml.contains("refmods: ./") || yaml.contains("refmods: refmods/"),
            "missing refmods mapping"
        );
        if eros_checkpoints_root().is_dir() {
            assert!(
                yaml.contains("minimax_h3_eros:"),
                "Eros root exists but extra_model_paths omits it"
            );
            assert!(
                yaml.contains("minimax-h3-eros") || yaml.to_ascii_lowercase().contains("eros"),
                "Eros path missing from yaml: {yaml}"
            );
        }
        if stock_ref2va_backup_root().is_dir() {
            assert!(
                yaml.contains("minimax_h3_ref2va_stock:"),
                "stock Ref2VA backup exists but extra_model_paths omits it"
            );
            assert!(
                yaml.contains("minimax-h3-backup"),
                "stock backup path missing from yaml: {yaml}"
            );
        }
        assert_eq!(STOCK_REF2VA_UNET, "minimax_h3_ref2va_pruned_int8_convrot.safetensors");
        assert_eq!(
            EROS_REF2VA_INT8,
            "10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors"
        );
        assert_eq!(
            SINGULARITY_REF2VA_INT8,
            "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors"
        );
        if singularity_checkpoints_root().is_dir() {
            assert!(
                yaml.contains("minimax_h3_singularity:"),
                "Singularity root exists but extra_model_paths omits it"
            );
        }
    }

    #[test]
    fn default_video_vae_is_official_int8() {
        let resolved = resolve_video_vae(&H3VaeSelect::default(), None, false, false).unwrap();
        assert_eq!(resolved.filename, VIDEO_VAE_INT8_FILE);
        assert_eq!(
            resolved.fingerprint_id,
            "minimax_h3_video_vae_int8_convrot"
        );
    }

    #[test]
    fn vae_fp16_override() {
        let select = H3VaeSelect {
            vae: Some(H3VideoVae::Fp16),
            video_vae: None,
        };
        let resolved = resolve_video_vae(&select, None, false, false).unwrap();
        assert_eq!(resolved.filename, VIDEO_VAE_FP16_FILE);
    }

    #[test]
    fn video_vae_name_wins_over_enum() {
        let select = H3VaeSelect {
            vae: Some(H3VideoVae::Int8),
            video_vae: Some(VIDEO_VAE_FP16_FILE.into()),
        };
        let resolved = resolve_video_vae(&select, None, false, false).unwrap();
        assert_eq!(resolved.filename, VIDEO_VAE_FP16_FILE);
    }

    #[test]
    fn sidecar_reused_when_flags_omitted() {
        let resolved = resolve_video_vae(
            &H3VaeSelect::default(),
            Some("minimax_h3_video_vae_fp16"),
            false,
            false,
        )
        .unwrap();
        assert_eq!(resolved.filename, VIDEO_VAE_FP16_FILE);
    }

    #[test]
    fn python_engine_rejects_int8() {
        let select = H3VaeSelect {
            vae: Some(H3VideoVae::Int8),
            video_vae: None,
        };
        let err = resolve_video_vae(&select, None, true, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("python"), "{msg}");
        assert!(msg.contains("fp16"), "{msg}");
    }

    #[test]
    fn missing_override_fail_closes_on_dry_run() {
        let select = H3VaeSelect {
            vae: None,
            video_vae: Some("not_a_real_vae.safetensors".into()),
        };
        let err = resolve_video_vae(&select, None, false, true).unwrap_err();
        assert!(err.to_string().contains("not_a_real_vae"));
    }

    #[test]
    fn comfy_pin_is_036_kitchen_0234() {
        let v = include_str!("../../../runtimes/minimax-h3/ComfyUI/comfyui_version.py");
        assert!(
            v.contains("0.36.0"),
            "comfyui_version.py must report 0.36.0: {v}"
        );
        let py = include_str!("../../../runtimes/minimax-h3/pyproject.toml");
        assert!(
            py.contains("comfy-kitchen==0.2.34"),
            "kitchen pin must be 0.2.34"
        );
        assert!(py.contains("comfyui-frontend-package==1.52.7"));
        assert!(py.contains("comfy-aimdo==0.5.3"));
        assert_eq!(DEFAULT_AUDIO_VAE, format!("vae/{AUDIO_VAE_FILE}"));
    }
}
