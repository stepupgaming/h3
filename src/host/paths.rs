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
        if ffmpeg_file(&path) {
            return Ok(path);
        }
    }

    let repo_local = runtime_path(r"ffmpeg\bin\ffmpeg.exe");
    if ffmpeg_file(&repo_local) {
        return Ok(repo_local);
    }

    for path in ffmpeg_search_paths() {
        if ffmpeg_file(&path) {
            return Ok(path);
        }
    }

    bail!(
        "ffmpeg.exe not found. Run `h3 install`, place it at runtimes\\ffmpeg\\bin\\ffmpeg.exe, \
         or set GEMMY_FFMPEG."
    )
}

fn ffmpeg_file(path: &Path) -> bool {
    std::fs::metadata(path).map(|meta| meta.is_file()).unwrap_or(false)
}

fn ffmpeg_search_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(output) = crate::host::workers::env::native_tool_command("where")
        .arg("ffmpeg.exe")
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let line = line.trim();
            if !line.is_empty() {
                out.push(PathBuf::from(line));
            }
        }
    }
    for dir in path_dirs_from_env().into_iter().chain(registry_path_dirs()) {
        out.push(dir.join("ffmpeg.exe"));
    }
    out.push(PathBuf::from(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"));
    out.push(PathBuf::from(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"));
    out.push(PathBuf::from(r"C:\ffmpeg\bin\ffmpeg.exe"));
    if let Some(local) = local_appdata_dir() {
        out.push(local.join(r"Microsoft\WinGet\Links\ffmpeg.exe"));
        let packages = local.join(r"Microsoft\WinGet\Packages");
        if let Ok(entries) = std::fs::read_dir(&packages) {
            for entry in entries.flatten() {
                let Ok(children) = std::fs::read_dir(entry.path()) else {
                    continue;
                };
                for child in children.flatten() {
                    out.push(child.path().join("bin").join("ffmpeg.exe"));
                }
            }
        }
    }
    out
}

fn local_appdata_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(value));
    }
    std::env::var_os("USERPROFILE")
        .map(|home| PathBuf::from(home).join("AppData").join("Local"))
}

fn path_dirs_from_env() -> Vec<PathBuf> {
    let Ok(raw) = std::env::var("PATH") else {
        return Vec::new();
    };
    raw.split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn registry_path_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for key in [
        r"HKCU\Environment",
        r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
    ] {
        let Some(value) = reg_query_path(key) else {
            continue;
        };
        for part in expand_env_vars(&value).split(';') {
            let part = part.trim();
            if !part.is_empty() {
                dirs.push(PathBuf::from(part));
            }
        }
    }
    dirs
}

fn reg_query_path(key: &str) -> Option<String> {
    let output = crate::host::workers::env::native_tool_command("reg")
        .args(["query", key, "/v", "Path"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_reg_path_value(&String::from_utf8_lossy(&output.stdout))
}

fn parse_reg_path_value(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("Path") {
            continue;
        }
        for marker in ["REG_EXPAND_SZ", "REG_SZ"] {
            if let Some(index) = trimmed.find(marker) {
                let value = trimmed[index + marker.len()..].trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn expand_env_vars(input: &str) -> String {
    let mut out = String::new();
    let mut rest = input;
    while let Some(start) = rest.find('%') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        if let Some(end) = after.find('%') {
            let name = &after[..end];
            if name.is_empty() {
                out.push('%');
                rest = after;
                continue;
            }
            match std::env::var(name) {
                Ok(value) => out.push_str(&value),
                Err(_) => {
                    out.push('%');
                    out.push_str(name);
                    out.push('%');
                }
            }
            rest = &after[end + 1..];
        } else {
            out.push('%');
            rest = after;
        }
    }
    out.push_str(rest);
    out
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

    #[test]
    fn reg_path_value_keeps_spaces() {
        let output = "\r\nHKEY_CURRENT_USER\\Environment\r\n    Path    REG_EXPAND_SZ    C:\\Program Files\\ffmpeg\\bin;%LOCALAPPDATA%\\Microsoft\\WinGet\\Links\r\n";
        assert_eq!(
            super::parse_reg_path_value(output).as_deref(),
            Some(r"C:\Program Files\ffmpeg\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links")
        );
    }

    #[test]
    fn expand_env_vars_replaces_known_names() {
        let expanded = super::expand_env_vars("pre-%SystemRoot%-post");
        assert!(expanded.contains("Windows") || expanded.contains("WINDOWS") || expanded.contains('%'));
        assert!(expanded.starts_with("pre-"));
        assert!(expanded.ends_with("-post"));
    }
}
