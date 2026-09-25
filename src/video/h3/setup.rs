//! `h3 setup` — checkpoints folder and Comfy pack.
//!
//! The bundled pack under `runtimes/minimax-h3/ComfyUI` is the one that runs
//! these graphs. Pointing at another folder works only when that folder already
//! contains `run_h3_workflow.py` and `custom_nodes/gemmy-h3-context`. A stock
//! Comfy Portable tree does not.

use super::args::H3SetupArgs;
use super::download::{self, copy_bundled_tokenizer};
use super::install::ensure_h3_uv_env;
use super::paths::{
    checkpoints_root, comfy_root, comfy_runner_script, eros_checkpoints_root, h3_python, h3_root,
    runtime_root, singularity_checkpoints_root, stock_ref2va_backup_root,
};
use crate::host::config::{self, H3Config};
use crate::host::util::absolute_path;
use anyhow::{bail, Result};
use serde::Serialize;
use std::path::Path;

pub(crate) fn comfy_pack_problems(root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    let runner = root.join("run_h3_workflow.py");
    if !runner.is_file() {
        missing.push(format!("missing {}", runner.display()));
    }
    let context = root.join("custom_nodes").join("gemmy-h3-context");
    if !context.is_dir() {
        missing.push(format!("missing {}", context.display()));
    }
    missing
}

pub(crate) fn run_setup(args: H3SetupArgs, config: &H3Config) -> Result<()> {
    let mut setup = config.setup.clone();
    let mut changed = false;

    if let Some(path) = &args.checkpoints {
        setup.checkpoints = Some(absolute_path(path)?);
        changed = true;
    }
    if let Some(path) = &args.eros {
        setup.eros = Some(absolute_path(path)?);
        changed = true;
    }
    if let Some(path) = &args.ref2va_stock {
        setup.ref2va_stock = Some(absolute_path(path)?);
        changed = true;
    }
    if let Some(path) = &args.singularity {
        setup.singularity = Some(absolute_path(path)?);
        changed = true;
    }
    if let Some(path) = &args.comfy {
        let path = absolute_path(path)?;
        let problems = comfy_pack_problems(&path);
        if !problems.is_empty() {
            bail!(
                "that folder is not this H3 Comfy pack:\n  {}\n\
                 A stock Comfy or Comfy Portable install does not include the runner or nodes.\n\
                 Use the bundled pack: h3 setup --comfy-bundled\n\
                 Bundled path: {}",
                problems.join("\n  "),
                h3_root().join("ComfyUI").display()
            );
        }
        setup.comfy = Some(path);
        changed = true;
    }
    if args.comfy_bundled {
        setup.comfy = None;
        changed = true;
    }

    if changed {
        config::save_setup(&setup)?;
        let mut updated = config.clone();
        updated.setup = setup;
        super::paths::apply_roots_from_config(&updated);
        println!(
            "[h3 setup] saved {}",
            config::h3_config_path()?.display()
        );
    }

    let mut notes = Vec::new();
    if std::env::var_os("GEMMY_H3_CHECKPOINTS").is_some() {
        notes.push(
            "GEMMY_H3_CHECKPOINTS is set and overrides the saved checkpoints folder".to_string(),
        );
    }
    if std::env::var_os("GEMMY_H3_COMFY").is_some() {
        notes.push("GEMMY_H3_COMFY is set and overrides the saved Comfy folder".to_string());
    }

    let comfy = comfy_root();
    let problems = comfy_pack_problems(&comfy);
    if !problems.is_empty() {
        notes.push(format!(
            "Comfy pack incomplete at {} — {}",
            comfy.display(),
            problems.join("; ")
        ));
    }

    if !args.no_runtime && (!h3_python().is_file() || args.sync) {
        ensure_h3_uv_env(&runtime_root(), args.verbose)?;
        notes.push("uv sync finished".into());
    } else if args.no_runtime {
        notes.push("skipped uv sync (--no-runtime)".into());
    } else if h3_python().is_file() {
        notes.push(format!("python already present: {}", h3_python().display()));
    }

    if problems.is_empty() {
        match super::paths::write_comfy_extra_model_paths() {
            Ok(path) => notes.push(format!("wrote {}", path.display())),
            Err(err) => notes.push(format!("extra_model_paths.yaml not written: {err:#}")),
        }
        if checkpoints_root().is_dir() || args.checkpoints.is_some() {
            if let Err(err) = copy_bundled_tokenizer() {
                notes.push(format!("tokenizer not copied: {err:#}"));
            }
        }
    }

    let report = SetupReport {
        config: config::h3_config_path()?.display().to_string(),
        checkpoints: checkpoints_root().display().to_string(),
        comfy: comfy.display().to_string(),
        comfy_runner: comfy_runner_script().is_file(),
        eros: eros_checkpoints_root().display().to_string(),
        ref2va_stock: stock_ref2va_backup_root().display().to_string(),
        singularity: singularity_checkpoints_root().display().to_string(),
        python: h3_python().is_file(),
        notes,
        sets: download::set_statuses(),
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("[h3 setup] config={}", report.config);
        println!("[h3 setup] checkpoints={}", report.checkpoints);
        println!(
            "[h3 setup] comfy={} runner={}",
            report.comfy, report.comfy_runner
        );
        println!("[h3 setup] eros={}", report.eros);
        println!("[h3 setup] ref2va-stock={}", report.ref2va_stock);
        println!("[h3 setup] singularity={}", report.singularity);
        println!("[h3 setup] python={}", report.python);
        for note in &report.notes {
            println!("[h3 setup] {note}");
        }
        println!("[h3 setup] weight sets (`h3 download <name>`):");
        for set in &report.sets {
            let mark = if set.ready { "present" } else { "missing" };
            println!("[h3 setup] {mark:7} {}", set.id);
        }
    }
    if !report.comfy_runner {
        bail!(
            "Comfy runner missing. The bundled pack is {}.",
            h3_root().join("ComfyUI").display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_folder_is_not_an_h3_comfy_pack() {
        let dir = tempfile::tempdir().unwrap();
        let problems = comfy_pack_problems(dir.path());
        assert!(problems.iter().any(|line| line.contains("run_h3_workflow.py")));
        assert!(problems.iter().any(|line| line.contains("gemmy-h3-context")));
    }
}

#[derive(Serialize)]
struct SetupReport {
    config: String,
    checkpoints: String,
    comfy: String,
    comfy_runner: bool,
    eros: String,
    ref2va_stock: String,
    singularity: String,
    python: bool,
    notes: Vec<String>,
    sets: Vec<download::SetStatus>,
}
