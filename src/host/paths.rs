use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os("GEMMY_REPO_ROOT") {
        let path = PathBuf::from(root);
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(root) = layout_root(dir) {
                return root;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Install directory when `h3.exe` sits beside `runtimes/minimax-h3`.
fn layout_root(dir: &Path) -> Option<PathBuf> {
    if dir.join("runtimes").join("minimax-h3").is_dir() {
        Some(dir.to_path_buf())
    } else {
        None
    }
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

#[cfg(test)]
mod tests {
    use super::layout_root;
    use std::fs;

    #[test]
    fn layout_root_requires_the_engine_directory() {
        let root = std::env::temp_dir().join(format!("h3-layout-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("runtimes").join("minimax-h3")).unwrap();
        assert_eq!(layout_root(&root).as_deref(), Some(root.as_path()));
        let bare = root.join("empty");
        fs::create_dir_all(&bare).unwrap();
        assert!(layout_root(&bare).is_none());
        let _ = fs::remove_dir_all(&root);
    }
}
