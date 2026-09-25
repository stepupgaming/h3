//! `h3 install|verify` — uv env under runtimes/minimax-h3 + presence checks.

use super::args::{H3InstallArgs, H3VerifyArgs};
use super::models::{all_required_ok, collect_asset_reports};
use super::paths::{
    checkpoints_root, comfy_root, h3_python, h3_root, runtime_root, worker_script,
};
use crate::host::config::H3Config;
use crate::host::util::run_simple_command;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::Path;
use std::process::Command;

#[derive(Serialize)]
struct InstallSummary {
    h3_root: String,
    runtime_root: String,
    checkpoints_root: String,
    python: String,
    worker: String,
    runtime_ready: bool,
    actions: Vec<String>,
    notes: Vec<String>,
    assets: Vec<super::models::AssetReport>,
}

pub(crate) fn run_install(args: H3InstallArgs, _config: &H3Config) -> Result<()> {
    let root = h3_root();
    let mut actions = Vec::new();
    let mut notes = Vec::new();

    if !root.is_dir() {
        bail!(
            "MiniMax-H3 runtime root not found: {}\n\
             Expected internalized code at runtimes\\minimax-h3 (or set GEMMY_H3_ROOT).",
            root.display()
        );
    }

    let want_runtime = args.runtime && !args.no_runtime;
    if want_runtime {
        if args.dry_run {
            actions.push(format!(
                "dry-run: would uv sync in {}",
                root.display()
            ));
        } else {
            ensure_h3_uv_env(&root, args.verbose)?;
            actions.push("h3 runtime uv sync completed or already present".into());
        }
    } else {
        notes.push("skipped runtime uv sync (--no-runtime)".into());
    }

    let worker = worker_script();
    if !worker.is_file() {
        bail!("H3 worker missing: {}", worker.display());
    }
    actions.push(format!("worker present: {}", worker.display()));

    if args.dry_run {
        actions.push(format!(
            "dry-run: would write {}",
            comfy_root().join("extra_model_paths.yaml").display()
        ));
    } else {
        match super::paths::write_comfy_extra_model_paths() {
            Ok(extra) => actions.push(format!(
                "wrote Comfy extra_model_paths {}",
                extra.display()
            )),
            Err(err) => notes.push(format!("extra_model_paths.yaml not written: {err:#}")),
        }
    }

    notes.push(format!(
        "Weights stay external under {}. `h3 install` does not download them. \
         `h3 download --list` then `h3 download base` (and eros, ref2va-stock, …). \
         `h3 setup` chooses the checkpoints folder and the Comfy pack.",
        checkpoints_root().display()
    ));

    let assets = collect_asset_reports();
    let runtime_ready = all_required_ok(&assets);
    let summary = InstallSummary {
        h3_root: root.display().to_string(),
        runtime_root: runtime_root().display().to_string(),
        checkpoints_root: checkpoints_root().display().to_string(),
        python: h3_python().display().to_string(),
        worker: worker.display().to_string(),
        runtime_ready,
        actions,
        notes,
        assets,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("[h3 install] h3_root={}", summary.h3_root);
        println!(
            "[h3 install] checkpoints_root={}",
            summary.checkpoints_root
        );
        println!("[h3 install] python={}", summary.python);
        for a in &summary.actions {
            println!("[h3 install] {a}");
        }
        for n in &summary.notes {
            println!("[h3 install] note: {n}");
        }
        if summary.runtime_ready {
            println!("[h3 install] ready — all required assets present");
        } else {
            println!("[h3 install] incomplete — run: h3 doctor");
            for r in summary.assets.iter().filter(|r| r.required && !r.ok) {
                println!("[h3 install] missing: {} @ {}", r.id, r.path);
            }
        }
    }

    if !summary.runtime_ready && !args.dry_run {
        // Install itself succeeds if uv worked; missing weights are a verify/doctor concern.
        eprintln!(
            "[h3 install] warning: required weights/scripts still missing — generate will fail until present"
        );
    }
    Ok(())
}

pub(crate) fn run_verify(args: H3VerifyArgs, _config: &H3Config) -> Result<()> {
    let assets = collect_asset_reports();
    let ok = all_required_ok(&assets);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": ok,
                "h3_root": h3_root().display().to_string(),
                "checkpoints_root": checkpoints_root().display().to_string(),
                "assets": assets,
            }))?
        );
    } else {
        println!("[h3 verify] h3_root={}", h3_root().display());
        println!(
            "[h3 verify] checkpoints_root={}",
            checkpoints_root().display()
        );
        for r in &assets {
            let mark = if r.ok { "ok" } else { "FAIL" };
            println!("[h3 verify] [{mark}] {} — {}", r.id, r.path);
            if args.verbose && r.bytes.is_some() {
                println!("[h3 verify]   bytes={}", r.bytes.unwrap());
            }
        }
        if ok {
            println!("[h3 verify] all required assets present");
        } else {
            bail!("h3 verify failed — see missing assets above");
        }
    }
    if !ok {
        bail!("h3 verify failed");
    }
    Ok(())
}

pub(crate) fn ensure_h3_uv_env(root: &Path, verbose: bool) -> Result<()> {
    let py = root.join(r".venv\Scripts\python.exe");
    let pyproject = root.join("pyproject.toml");
    if !pyproject.is_file() {
        bail!(
            "H3 pyproject.toml missing at {} — expected runtimes\\minimax-h3 (or GEMMY_H3_ROOT).",
            pyproject.display()
        );
    }

    let uv = which_uv()?;
    let mut cmd = Command::new(&uv);
    cmd.arg("sync")
        .current_dir(root)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8");
    if verbose {
        eprintln!(
            "[h3 install] running: {} sync (cwd={})",
            uv.display(),
            root.display()
        );
    }
    run_simple_command(cmd, "h3 uv sync", verbose)
        .with_context(|| format!("uv sync in {}", root.display()))?;

    if !py.is_file() {
        bail!(
            "after uv sync, python still missing: {}\n\
             Install uv and ensure runtimes\\minimax-h3 can create .venv",
            py.display()
        );
    }
    Ok(())
}

fn which_uv() -> Result<std::path::PathBuf> {
    if let Ok(raw) = std::env::var("UV") {
        let p = std::path::PathBuf::from(raw);
        if p.is_file() {
            return Ok(p);
        }
    }
    let mut cmd = crate::host::workers::env::native_tool_command("where");
    cmd.arg("uv.exe");
    if let Ok(output) = cmd.output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = stdout.lines().next() {
            let t = line.trim();
            if !t.is_empty() && Path::new(t).exists() {
                return Ok(std::path::PathBuf::from(t));
            }
        }
    }
    // bare name; let OS resolve
    Ok(std::path::PathBuf::from("uv"))
}
