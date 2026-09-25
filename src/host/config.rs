//! Narrow read of Gemmy's config, plus this CLI's own setup file.
//!
//! `%APPDATA%\gemmy\config.json` is read for `model_paths.minimax_h3` and
//! `env_overrides` only. That file is never written.
//! `%APPDATA%\h3\config.json` stores the roots from `h3 setup`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
#[cfg(test)]
use std::path::Path;

thread_local! {
    static H3_CONFIG_PATH_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ModelPaths {
    #[serde(default)]
    pub minimax_h3: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct H3Config {
    #[serde(default)]
    pub model_paths: ModelPaths,
    #[serde(default)]
    pub env_overrides: BTreeMap<String, String>,
    #[serde(skip)]
    pub setup: H3Setup,
}

/// Roots saved by `h3 setup`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct H3Setup {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoints: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comfy: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eros: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref2va_stock: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub singularity: Option<PathBuf>,
}

impl H3Config {
    pub(crate) fn load() -> Result<Self> {
        let mut config = Self::load_gemmy()?;
        config.setup = load_setup()?;
        Ok(config)
    }

    fn load_gemmy() -> Result<Self> {
        let Some(path) = gemmy_config_path() else {
            return Ok(Self::default());
        };
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| {
            format!(
                "parse {} for H3 weight path and env_overrides",
                path.display()
            )
        })
    }
}

pub(crate) fn h3_config_path() -> Result<PathBuf> {
    if let Some(path) = H3_CONFIG_PATH_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return Ok(path);
    }
    let appdata = std::env::var("APPDATA").context("APPDATA is not set")?;
    Ok(PathBuf::from(appdata).join("h3").join("config.json"))
}

pub(crate) fn load_setup() -> Result<H3Setup> {
    let path = h3_config_path()?;
    if !path.is_file() {
        return Ok(H3Setup::default());
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub(crate) fn save_setup(setup: &H3Setup) -> Result<()> {
    let path = h3_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(setup).context("serialize h3 config")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, body).with_context(|| format!("write {}", tmp.display()))?;
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))?;
    Ok(())
}

fn gemmy_config_path() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    Some(PathBuf::from(appdata).join("gemmy").join("config.json"))
}

#[cfg(test)]
pub(crate) fn set_h3_config_path_override(path: Option<&Path>) {
    H3_CONFIG_PATH_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = path.map(Path::to_path_buf);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_roundtrip_does_not_touch_gemmy_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        set_h3_config_path_override(Some(&path));
        let setup = H3Setup {
            checkpoints: Some(PathBuf::from(r"D:\h3-models")),
            comfy: None,
            eros: Some(PathBuf::from(r"D:\h3-models\eros")),
            ref2va_stock: None,
            singularity: None,
        };
        save_setup(&setup).unwrap();
        let loaded = load_setup().unwrap();
        assert_eq!(loaded, setup);
        set_h3_config_path_override(None);
    }
}
