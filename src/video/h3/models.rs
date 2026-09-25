//! Weight / asset presence checks for MiniMax-H3.

use super::paths::{
    checkpoints_root, h3_python, h3_root, required_script_paths, required_weight_paths,
    worker_script, OPTIONAL_VIDEO_VAE_FP16,
};
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AssetReport {
    pub id: String,
    pub path: String,
    pub present: bool,
    pub is_file: bool,
    pub is_dir: bool,
    pub bytes: Option<u64>,
    pub required: bool,
    pub ok: bool,
    pub message: String,
}

fn report(id: &str, path: PathBuf, required: bool) -> AssetReport {
    let present = path.exists();
    let is_file = path.is_file();
    let is_dir = path.is_dir();
    let bytes = if is_file {
        std::fs::metadata(&path).ok().map(|m| m.len())
    } else {
        None
    };
    let ok = if required {
        present && (is_file || is_dir)
    } else {
        true
    };
    let message = if ok {
        "ok".into()
    } else if !present {
        "missing".into()
    } else {
        "not a file/dir as expected".into()
    };
    AssetReport {
        id: id.into(),
        path: path.display().to_string(),
        present,
        is_file,
        is_dir,
        bytes,
        required,
        ok,
        message,
    }
}

pub(crate) fn collect_asset_reports() -> Vec<AssetReport> {
    let mut out = Vec::new();
    let root = h3_root();
    out.push(report("h3_root", root.clone(), true));
    out.push(report("h3_python", h3_python(), true));
    out.push(report("worker", worker_script(), true));
    out.push(report("checkpoints_root", checkpoints_root(), true));
    for (id, path) in required_script_paths(&root) {
        out.push(report(id, path, true));
    }
    for (id, path) in required_weight_paths() {
        out.push(report(id, path, true));
    }
    out.push(report(
        "video_vae_fp16",
        checkpoints_root().join(OPTIONAL_VIDEO_VAE_FP16),
        false,
    ));
    // Stock Comfy-Org Ref2VA — backup on G:. Product `--mode ref2va` is Eros.
    out.push(report(
        "dit_ref2va_stock_backup",
        super::paths::stock_ref2va_backup_path(),
        false,
    ));
    // Eros INT8 — product `--mode ref2va` and shortfilm.
    out.push(report(
        "dit_eros_ref2va_int8",
        super::paths::eros_ref2va_int8_path(),
        false,
    ));
    out.push(report(
        "dit_eros_ref2va_bf16",
        super::paths::eros_ref2va_bf16_path(),
        false,
    ));
    out.push(report(
        "dit_singularity",
        super::paths::singularity_ref2va_int8_path(),
        false,
    ));
    out.push(report(
        "vsa_gate",
        super::paths::vsa_gate_path(super::paths::VSA_GATE_DEFAULT),
        false,
    ));
    out
}

pub(crate) fn all_required_ok(reports: &[AssetReport]) -> bool {
    reports.iter().filter(|r| r.required).all(|r| r.ok)
}

pub(crate) fn validate_runtime_ready() -> Result<()> {
    use super::paths::{
        comfy_worker_script, ensure_exists, h3_python, h3_root, worker_script,
    };

    ensure_exists(&h3_root(), "MiniMax-H3 code root")?;
    ensure_exists(&h3_python(), "MiniMax-H3 python")?;
    ensure_exists(&worker_script(), "MiniMax-H3 python worker")?;
    ensure_exists(&comfy_worker_script(), "MiniMax-H3 Comfy worker")?;

    let reports = collect_asset_reports();
    if all_required_ok(&reports) {
        return Ok(());
    }
    let missing: Vec<_> = reports
        .iter()
        .filter(|r| r.required && !r.ok)
        .map(|r| format!("{} ({})", r.id, r.path))
        .collect();
    anyhow::bail!(
        "MiniMax-H3 runtime not ready. Missing/invalid:\n  - {}\n\
         Run: h3 install && h3 doctor\n\
         See docs\\MODEL_LOCATIONS.md (MiniMax-H3).",
        missing.join("\n  - ")
    )
}
