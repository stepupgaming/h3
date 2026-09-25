//! Same lock file Gemmy uses (`%APPDATA%\gemmy\gemmy.queue.lock`) so `h3` and
//! `gemmy` cannot sample the 16 GB card at the same time.
//!
//! Help, install, verify, doctor, dry-run plans, `--check`, and refmod
//! list/inspect do not take the lock. A delegated `gemmy` child inherits
//! `GEMMY_QUEUE_ACTIVE` so it does not wait on its parent.

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

pub(crate) struct QueueGuard {
    file: std::fs::File,
}

impl QueueGuard {
    pub(crate) fn acquire_if_needed(args: &[OsString]) -> Result<Option<Self>> {
        if !should_queue(args) {
            return Ok(None);
        }
        if std::env::var_os("GEMMY_QUEUE_ACTIVE").as_deref() == Some(std::ffi::OsStr::new("1")) {
            if let Ok(pid) = std::env::var("GEMMY_QUEUE_HOLDER_PID") {
                if let Ok(pid) = pid.trim().parse::<u32>() {
                    if pid == std::process::id() || process_alive(pid) {
                        return Ok(None);
                    }
                }
            }
        }

        let path = lock_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;

        let started = Instant::now();
        let mut next_status = Instant::now();
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => break,
                Err(_) => {
                    if Instant::now() >= next_status {
                        eprintln!("[queue] waiting for {}", path.display());
                        next_status = Instant::now() + Duration::from_secs(15);
                    }
                    if started.elapsed() > Duration::from_secs(60 * 60) {
                        bail!("timed out waiting for GPU queue lock {}", path.display());
                    }
                    std::thread::sleep(Duration::from_millis(750));
                }
            }
        }

        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        let shown = args
            .iter()
            .skip(1)
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, "pid={}", std::process::id())?;
        writeln!(file, "args={shown}")?;
        file.flush()?;
        unsafe {
            std::env::set_var("GEMMY_QUEUE_ACTIVE", "1");
            std::env::set_var("GEMMY_QUEUE_HOLDER_PID", std::process::id().to_string());
        }
        Ok(Some(Self { file }))
    }
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        let _ = self.file.unlock();
        unsafe {
            std::env::remove_var("GEMMY_QUEUE_ACTIVE");
            std::env::remove_var("GEMMY_QUEUE_HOLDER_PID");
        }
    }
}

fn should_queue(args: &[OsString]) -> bool {
    let tokens: Vec<String> = args
        .iter()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    if tokens.is_empty() {
        return false;
    }
    if tokens
        .iter()
        .any(|token| token == "--help" || token == "-h" || token == "--dry-run-plan")
    {
        return false;
    }
    let words: Vec<&str> = tokens
        .iter()
        .filter(|token| !token.starts_with('-'))
        .map(String::as_str)
        .collect();
    match words.first().copied() {
        Some("install" | "verify" | "doctor" | "setup" | "download") => false,
        Some("upscale" | "interpolate") => !tokens.iter().any(|token| token == "--check"),
        Some("refmod") => !matches!(words.get(1).copied(), Some("list" | "inspect")),
        Some(_) => true,
        None => false,
    }
}

fn lock_path() -> Result<PathBuf> {
    let appdata = std::env::var_os("APPDATA").context("APPDATA is required for the shared GPU queue lock")?;
    Ok(PathBuf::from(appdata)
        .join("gemmy")
        .join("gemmy.queue.lock"))
}

fn process_alive(pid: u32) -> bool {
    let Ok(output) = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
    else {
        return false;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let lower = text.to_ascii_lowercase();
    !lower.contains("no tasks") && text.contains(&pid.to_string())
}
