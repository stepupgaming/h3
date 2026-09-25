use anyhow::{Context, Result, bail};
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

pub(crate) const SUPERVISOR_START_GATE_ENV: &str = "GEMMY_SUPERVISOR_START_GATE";
pub(crate) const SUPERVISOR_CANCEL_SIGNAL_ENV: &str = "GEMMY_CANCEL_SIGNAL_PATH";

pub const HOST_PYTHON_ENV_VARS: &[&str] = &[
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONSAFEPATH",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
    "CONDA_PROMPT_MODIFIER",
    "PIP_CONFIG_FILE",
    "PIP_REQUIRE_VIRTUALENV",
    "UV_PROJECT_ENVIRONMENT",
];

pub fn apply_isolated_runtime_env(cmd: &mut Command) {
    for name in HOST_PYTHON_ENV_VARS {
        cmd.env_remove(name);
    }
    cmd.env("PYTHONNOUSERSITE", "1");
    cmd.env("PYTHONUTF8", "1");
}

/// Applies user-managed env overrides from config.json BEFORE any
/// Gemmy-owned variables are set, so Gemmy-managed values always win.
pub fn apply_user_env_overrides(cmd: &mut Command) {
    if let Ok(config) = crate::host::config::H3Config::load() {
        apply_env_overrides(cmd, &config.env_overrides);
    }
}

fn apply_env_overrides(cmd: &mut Command, overrides: &std::collections::BTreeMap<String, String>) {
    for (key, value) in overrides {
        cmd.env(key, value);
    }
}

pub fn native_tool_command(program: impl AsRef<OsStr>) -> Command {
    let mut cmd = Command::new(program);
    apply_user_env_overrides(&mut cmd);
    apply_isolated_runtime_env(&mut cmd);
    cmd
}

pub fn python_worker_command(python: impl AsRef<OsStr>) -> Command {
    let mut cmd = native_tool_command(python);
    cmd.arg("-I");
    cmd
}

pub fn gemmy_child_command(gemmy_exe: impl AsRef<OsStr>) -> Command {
    let mut cmd = native_tool_command(gemmy_exe);
    cmd.env("GEMMY_NO_ROUTE", "1");
    // A nested `gemmy` child runs under a parent that already owns the process
    // queue lock, so it must skip re-acquiring it or it deadlocks behind its
    // own parent (see DOX: composite workflows do not deadlock behind selves).
    //
    // Always stamp both the active marker and the holder pid. On Windows the
    // exclusive lock file body is often unreadable to the child, so inheritance
    // must not depend on parsing gemmy.queue.lock.
    cmd.env("GEMMY_QUEUE_ACTIVE", "1");
    let holder_pid = std::env::var("GEMMY_QUEUE_HOLDER_PID")
        .ok()
        .filter(|value| {
            value
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|pid| *pid != 0)
                .is_some()
        })
        .unwrap_or_else(|| std::process::id().to_string());
    cmd.env("GEMMY_QUEUE_HOLDER_PID", holder_pid);
    cmd
}

/// Supervised `gemmy.exe` children pause at the first Rust entrypoint until
/// their parent has assigned them to the Windows Job Object and durably
/// recorded the PID. Clearing the variable before normal dispatch prevents a
/// nested Gemmy child from inheriting a consumed gate.
pub(crate) fn wait_for_supervisor_start_gate() -> Result<()> {
    let Some(path) = std::env::var_os(SUPERVISOR_START_GATE_ENV).map(std::path::PathBuf::from)
    else {
        return Ok(());
    };
    if !path.is_absolute() {
        bail!("supervisor start gate must be an absolute path");
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        // Cooperative cancel can arrive before ownership is released.
        check_supervisor_cancel_signal()?;
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_file() => {
                // This is called before Gemmy starts worker threads, so the
                // Rust 2024 process-environment mutation race does not apply.
                unsafe {
                    std::env::remove_var(SUPERVISOR_START_GATE_ENV);
                }
                let _ = std::fs::remove_file(&path);
                return Ok(());
            }
            Ok(_) => bail!(
                "supervisor start gate is not a regular file: {}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect supervisor start gate {}", path.display()));
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "timed out waiting for supervisor process ownership gate {}",
                path.display()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Fail immediately when the supervisor has published a cooperative cancel signal.
///
/// Safe to call from long-running CLI entrypoints and worker loops. Missing env
/// or missing file means "not cancelled".
pub(crate) fn check_supervisor_cancel_signal() -> Result<()> {
    let Some(path) = std::env::var_os(SUPERVISOR_CANCEL_SIGNAL_ENV).map(std::path::PathBuf::from)
    else {
        return Ok(());
    };
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() => {
            bail!(
                "job cancelled by supervisor (cooperative signal at {})",
                path.display()
            );
        }
        Ok(_) => bail!(
            "supervisor cancel signal path is not a regular file: {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("inspect supervisor cancel signal {}", path.display())),
    }
}

pub fn prepend_path(cmd: &mut Command, path: impl AsRef<Path>) -> Result<()> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut entries = vec![path.as_ref().to_path_buf()];
    entries.extend(std::env::split_paths(&current));
    let joined: OsString =
        std::env::join_paths(entries).context("Failed to build isolated worker PATH")?;
    cmd.env("PATH", joined);
    Ok(())
}

