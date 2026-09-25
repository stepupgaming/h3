//! `h3 face-refine` — optional post pass on a finished MP4.

use super::args::H3FaceRefineArgs;
use super::paths::{
    abs, checkpoints_root, face_refine_worker_script, h3_python, h3_root, resolve_video_vae,
};
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::util::absolute_path;
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize)]
struct FaceRefinePlan {
    input: PathBuf,
    output: PathBuf,
    worker: PathBuf,
    python: PathBuf,
    h3_root: PathBuf,
    node_pack: PathBuf,
    video_vae: String,
}

pub(crate) fn run_face_refine(args: H3FaceRefineArgs, _config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;
    if args.dry_run_plan {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("[h3 face-refine] dry-run plan");
            println!("[h3 face-refine] input={}", plan.input.display());
            println!("[h3 face-refine] output={}", plan.output.display());
            println!("[h3 face-refine] video_vae={}", plan.video_vae);
            println!("[h3 face-refine] node_pack={}", plan.node_pack.display());
            println!("[h3 face-refine] worker={}", plan.worker.display());
        }
        return Ok(());
    }
    if !plan.input.is_file() {
        bail!("--input not found: {}", plan.input.display());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.worker.is_file() {
        bail!("face-refine worker missing: {}", plan.worker.display());
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }

    let request = json!({
        "input": plan.input,
        "output": plan.output,
        "h3_root": plan.h3_root,
        "checkpoints_root": checkpoints_root(),
        "python": plan.python,
        "node_pack": plan.node_pack,
        "video_vae": plan.video_vae,
    });
    let work = plan
        .output
        .parent()
        .map(|p| p.join(".h3_face_refine"))
        .unwrap_or_else(|| PathBuf::from(".h3_face_refine"));
    std::fs::create_dir_all(&work)?;
    let req_path = work.join("request.json");
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;

    println!(
        "[h3 face-refine] {} → {}",
        plan.input.display(),
        plan.output.display()
    );
    println!("[h3 face-refine] unloading Gemmy before FaceRefine...");

    let verbose = args.verbose;
    let python = plan.python.clone();
    let worker = plan.worker.clone();
    let out = plan.output.clone();
    let h3_root = plan.h3_root.clone();
    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
        super::paths::forward_extra_dit_roots(&mut cmd);
        cmd.current_dir(&h3_root);
        cmd.arg(&worker).arg("--request").arg(&req_path);
        WorkerSpec::new(cmd, "video.h3.face-refine")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(out.clone()).labeled("H3 face-refine mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    println!("wrote H3 face-refine video: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "face-refine",
                "plan": plan,
                "output": plan.output,
            }))?
        );
    }
    Ok(())
}

fn build_plan(args: &H3FaceRefineArgs) -> Result<FaceRefinePlan> {
    let input = abs(&args.input)?;
    let output = match &args.output {
        Some(p) => absolute_path(p)?,
        None => {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            let parent = input.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            absolute_path(&parent.join(format!("{stem}_face.mp4")))?
        }
    };
    let node_pack = h3_root()
        .join("ComfyUI")
        .join("custom_nodes")
        .join("ComfyUI-H3-FaceRefine");
    Ok(FaceRefinePlan {
        input,
        output,
        worker: face_refine_worker_script(),
        python: h3_python(),
        h3_root: h3_root(),
        node_pack,
        video_vae: resolve_video_vae(
            &args.vae_select,
            None,
            false,
            args.vae_select.video_vae.is_some() || !args.dry_run_plan,
        )?
        .filename,
    })
}
