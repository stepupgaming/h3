use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os("GEMMY_REPO_ROOT") {
        return PathBuf::from(root);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn runtime_path(relative: &str) -> PathBuf {
    repo_root().join("runtimes").join(relative)
}

pub fn models_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("GEMMY_MODELS_ROOT") {
        let path = PathBuf::from(root);
        if !path.as_os_str().is_empty() {
            return Some(path);
        }
    }
    let legacy = PathBuf::from(r"F:\Models");
    if legacy.is_dir() {
        return Some(legacy);
    }
    None
}

/// Default checkpoints folder when no models drive and no saved root exist.
pub fn user_h3_models() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(|home| PathBuf::from(home).join("h3-models"))
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("h3-models"))
}

pub fn models_rel(relative: impl AsRef<Path>) -> PathBuf {
    models_root()
        .unwrap_or_else(|| PathBuf::from(r"F:\Models"))
        .join(relative)
}

pub fn pin_huggingface_cache(cmd: &mut Command) {
    let existing_home = std::env::var_os("HF_HOME").filter(|v| !v.is_empty());
    let existing_hub = std::env::var_os("HUGGINGFACE_HUB_CACHE")
        .or_else(|| std::env::var_os("HF_HUB_CACHE"))
        .filter(|v| !v.is_empty());

    let resolved_home = existing_home
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("GEMMY_HF_HOME")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| models_root().map(|root| root.join("huggingface")));

    let Some(home) = resolved_home else {
        return;
    };
    let hub = existing_hub
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join("hub"));

    cmd.env("HF_HOME", &home)
        .env("HUGGINGFACE_HUB_CACHE", &hub)
        .env("HF_HUB_CACHE", &hub);
}

pub fn default_output_path(filename: String) -> PathBuf {
    repo_root().join("outputs").join(filename)
}

pub fn find_ffmpeg() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("GEMMY_FFMPEG") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }

    let repo_local = runtime_path(r"ffmpeg\bin\ffmpeg.exe");
    if repo_local.exists() {
        return Ok(repo_local);
    }

    if let Ok(output) = crate::host::workers::env::native_tool_command("where")
        .arg("ffmpeg.exe")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let first = stdout.lines().next().map(|s| s.trim());
        if let Some(path) = first {
            if !path.is_empty() && Path::new(path).exists() {
                return Ok(PathBuf::from(path));
            }
        }
    }

    let candidates = [
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ];
    for path in &candidates {
        if Path::new(path).exists() {
            return Ok(PathBuf::from(path));
        }
    }

    bail!(
        "ffmpeg.exe not found. Install ffmpeg on PATH, place it at runtimes\\ffmpeg\\bin\\ffmpeg.exe, \
         or set GEMMY_FFMPEG."
    )
}
