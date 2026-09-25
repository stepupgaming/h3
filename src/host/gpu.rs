//! GPU handoff for a video-only CLI: stop `llama-server.exe` if it holds VRAM.
//! Does not restore a Gemmy server and does not publish a Dynamic VRAM budget.
//! H3 Comfy already forces `--disable-dynamic-vram`.

use anyhow::{Context, Result, bail};
use std::time::{Duration, Instant};

use crate::host::workers::env::native_tool_command;

pub(crate) fn with_gpu_handoff<T, F>(verbose: bool, action: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    stop_llama_servers(verbose)?;
    action()
}

pub(crate) fn stop_llama_servers(verbose: bool) -> Result<()> {
    if verbose {
        eprintln!("[gpu] Stopping llama-server processes before GPU handoff...");
    }
    if !llama_server_running()? {
        if verbose {
            eprintln!("[gpu] no llama-server.exe processes were running");
        }
        return Ok(());
    }

    let output = native_tool_command("taskkill")
        .args(["/F", "/T", "/IM", "llama-server.exe"])
        .output()
        .context("Failed to invoke taskkill for llama-server.exe")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let not_found = combined.contains("not found")
        || combined.contains("no tasks")
        || combined.contains("not running");
    if !output.status.success() && !not_found {
        bail!(
            "taskkill failed to stop llama-server.exe ({}): {}",
            output.status,
            combined.trim()
        );
    }

    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if !llama_server_running()? {
            if verbose {
                eprintln!("[gpu] llama-server.exe stopped");
            }
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    bail!("llama-server.exe is still running after taskkill; free VRAM before GPU handoff")
}

fn llama_server_running() -> Result<bool> {
    let output = native_tool_command("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/NH"])
        .output()
        .context("Failed to probe llama-server.exe via tasklist")?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    Ok(stdout.contains("llama-server.exe"))
}
