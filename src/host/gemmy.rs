//! Optional handoff to a full Gemmy install for stills and the analyst.
//! This CLI never calls `current_exe()` for those stages: that binary is `h3`.

use anyhow::{Result, bail};
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn gemmy_exe() -> Option<PathBuf> {
    for query in ["gemmy.exe", "gemmy"] {
        let Ok(output) = Command::new("where").arg(query).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let path = PathBuf::from(line.trim());
            if !path.is_file() {
                continue;
            }
            let name = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            if name == "gemmy" {
                return Some(path);
            }
        }
    }
    None
}

pub(crate) fn require_gemmy(why: &str) -> Result<PathBuf> {
    if let Some(path) = gemmy_exe() {
        return Ok(path);
    }
    bail!(
        "{why}. `gemmy` is not on PATH. This CLI does not ship Krea, SeedVR2, or the Gemma analyst."
    )
}
