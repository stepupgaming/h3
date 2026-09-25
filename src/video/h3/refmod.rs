//! `h3 refmod` — create / list / inspect saved H3 reference latents.

use super::args::{H3RefModArgs, H3RefModCommand, H3RefModCreateArgs};
use super::paths::{
    abs, checkpoints_root, comfy_root, h3_python, h3_root, refmod_worker_script, refmods_dir,
    resolve_video_vae,
};
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
struct RefmodPlan {
    action: String,
    name: String,
    images: Vec<PathBuf>,
    videos: Vec<PathBuf>,
    folder: Option<PathBuf>,
    audio: Option<PathBuf>,
    output: Option<PathBuf>,
    mode: String,
    worker: PathBuf,
    python: PathBuf,
    h3_root: PathBuf,
    node_pack: PathBuf,
    refmods_dir: PathBuf,
    package: String,
    video_vae: String,
}

pub(crate) fn run_refmod(args: H3RefModArgs, _config: &H3Config) -> Result<()> {
    match args.command {
        H3RefModCommand::List(a) => run_list_or_inspect("list", None, a.json, a.verbose),
        H3RefModCommand::Inspect(a) => {
            run_list_or_inspect("inspect", Some(a.name), a.json, a.verbose)
        }
        H3RefModCommand::Create(a) => run_create(a),
    }
}

fn run_list_or_inspect(action: &str, name: Option<String>, json_out: bool, verbose: bool) -> Result<()> {
    let python = h3_python();
    if !python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            python.display()
        );
    }
    let worker = refmod_worker_script();
    if !worker.is_file() {
        bail!("refmod worker missing: {}", worker.display());
    }
    let work = std::env::temp_dir().join("gemmy_h3_refmod");
    std::fs::create_dir_all(&work)?;
    let request = json!({
        "action": action,
        "name": name,
        "h3_root": h3_root(),
        "checkpoints_root": checkpoints_root(),
        "python": python,
        "refmods_dir": refmods_dir(),
    });
    let req_path = work.join(format!("{action}_request.json"));
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;
    let mut cmd = crate::host::workers::env::native_tool_command(&python);
    crate::host::paths::pin_huggingface_cache(&mut cmd);
    cmd.env("PYTHONUTF8", "1");
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
    super::paths::forward_refmods_dir(&mut cmd);
    cmd.current_dir(h3_root());
    cmd.arg(&worker).arg("--request").arg(&req_path);
    let spec = WorkerSpec::new(cmd, "video.h3.refmod").verbose(verbose);
    spec.run_inherited()?;
    let _ = json_out;
    Ok(())
}

fn run_create(args: H3RefModCreateArgs) -> Result<()> {
    let plan = build_create_plan(&args)?;
    if args.dry_run_plan {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("[h3 refmod] dry-run plan");
            println!("[h3 refmod] name={}", plan.name);
            println!("[h3 refmod] mode={}", plan.mode);
            println!("[h3 refmod] package={}", plan.package);
            println!("[h3 refmod] video_vae={}", plan.video_vae);
            println!("[h3 refmod] refmods_dir={}", plan.refmods_dir.display());
            if let Some(out) = &plan.output {
                println!("[h3 refmod] output={}", out.display());
            }
        }
        return Ok(());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.worker.is_file() {
        bail!("refmod worker missing: {}", plan.worker.display());
    }
    if !plan.node_pack.join("__init__.py").is_file() {
        bail!(
            "MiniMaxH3Mod pack missing: {} — expected ComfyUI-MiniMaxH3Mod under product Comfy custom_nodes",
            plan.node_pack.display()
        );
    }

    let work = plan
        .output
        .as_ref()
        .and_then(|p| p.parent().map(|d| d.join(".h3_refmod")))
        .unwrap_or_else(|| plan.refmods_dir.join(".work"));
    std::fs::create_dir_all(&work)?;
    let request = json!({
        "action": "create",
        "name": plan.name,
        "mode": plan.mode,
        "images": plan.images,
        "videos": plan.videos,
        "folder": plan.folder,
        "audio": plan.audio,
        "output": plan.output,
        "work_dir": work,
        "h3_root": plan.h3_root,
        "checkpoints_root": checkpoints_root(),
        "python": plan.python,
        "node_pack": plan.node_pack,
        "refmods_dir": plan.refmods_dir,
        "ref_resolution": args.ref_resolution,
        "max_tokens": args.max_tokens,
        "concept_type": args.concept_type,
        "description": args.description,
        "subfolder": args.subfolder,
        "video_vae": plan.video_vae,
    });
    let req_path = work.join("request.json");
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;

    println!(
        "[h3 refmod] create {} ({}) → {}",
        plan.name,
        plan.package,
        plan.refmods_dir.display()
    );
    println!("[h3 refmod] unloading Gemmy before VAE encode (no DiT)...");
    let python = plan.python.clone();
    let worker = plan.worker.clone();
    let h3 = plan.h3_root.clone();
    let verbose = args.verbose;
    let out = plan.output.clone();
    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
        super::paths::forward_extra_dit_roots(&mut cmd);
        super::paths::forward_refmods_dir(&mut cmd);
        cmd.current_dir(&h3);
        cmd.arg(&worker).arg("--request").arg(&req_path);
        let mut spec = WorkerSpec::new(cmd, "video.h3.refmod").verbose(verbose);
        if let Some(path) = out {
            spec = spec.expect_outputs(vec![
                OutputExpectation::file(path).labeled("H3 RefMod safetensors"),
            ]);
        }
        spec.run_inherited().map(|_| ())
    })?;
    Ok(())
}

fn build_create_plan(args: &H3RefModCreateArgs) -> Result<RefmodPlan> {
    let mut images = Vec::new();
    for p in &args.image {
        images.push(abs(p)?);
    }
    let mut videos = Vec::new();
    for p in &args.video {
        videos.push(abs(p)?);
    }
    let folder = args.folder.as_ref().map(|p| abs(p)).transpose()?;
    let audio = args.audio.as_ref().map(|p| abs(p)).transpose()?;
    if images.is_empty() && videos.is_empty() && folder.is_none() && audio.is_none() {
        bail!("refmod create needs --folder, --image, --video, and/or --audio");
    }
    let name = args
        .name
        .clone()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            folder
                .as_ref()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        })
        .unwrap_or_else(|| "character".into());
    let has_visual = !images.is_empty() || !videos.is_empty() || folder.is_some();
    let package = if has_visual && audio.is_some() {
        "comfy-workflow-h3-refmod-create-master"
    } else if audio.is_some() {
        "comfy-workflow-h3-refmod-create-audio"
    } else {
        "comfy-workflow-h3-refmod-create"
    };
    let output = args.output.as_ref().map(|p| abs(p)).transpose()?;
    Ok(RefmodPlan {
        action: "create".into(),
        name,
        images,
        videos,
        folder,
        audio,
        output,
        mode: args.mode.clone(),
        worker: refmod_worker_script(),
        python: h3_python(),
        h3_root: h3_root(),
        node_pack: comfy_root()
            .join("custom_nodes")
            .join("ComfyUI-MiniMaxH3Mod"),
        refmods_dir: refmods_dir(),
        package: package.into(),
        video_vae: resolve_video_vae(
            &args.vae_select,
            None,
            false,
            args.vae_select.video_vae.is_some() || !args.dry_run_plan,
        )?
        .filename,
    })
}
