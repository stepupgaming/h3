use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn timestamp_slug() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

pub(crate) fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("Failed to resolve current directory")?
            .join(path))
    }
}

pub(crate) fn run_simple_command(cmd: Command, label: &str, verbose: bool) -> Result<()> {
    crate::host::workers::process::WorkerSpec::new(cmd, label)
        .verbose(verbose)
        .run_inherited()
        .map(|_| ())
}
