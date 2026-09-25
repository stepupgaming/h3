//! `h3 interpolate` — DLSS Frame Generation post on a finished MP4.

use super::args::H3InterpolateArgs;
use super::paths::{abs, checkpoints_root, comfy_root, h3_python, h3_root, interpolate_worker_script};
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
struct InterpolatePlan {
    input: PathBuf,
    output: PathBuf,
    fps: String,
    engine: String,
    quality: String,
    codec: String,
    check_only: bool,
    worker: PathBuf,
    python: PathBuf,
    h3_root: PathBuf,
    node_pack: PathBuf,
    package: &'static str,
}

pub(crate) fn run_interpolate(args: H3InterpolateArgs, _config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;
    if args.dry_run_plan {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("[h3 interpolate] dry-run plan");
            println!("[h3 interpolate] input={}", plan.input.display());
            println!("[h3 interpolate] output={}", plan.output.display());
            println!(
                "[h3 interpolate] fps={} engine={} quality={} codec={}",
                plan.fps, plan.engine, plan.quality, plan.codec
            );
            println!("[h3 interpolate] package={}", plan.package);
            println!("[h3 interpolate] node_pack={}", plan.node_pack.display());
        }
        return Ok(());
    }
    if !plan.check_only && !plan.input.is_file() {
        bail!("--input not found: {}", plan.input.display());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.worker.is_file() {
        bail!("interpolate worker missing: {}", plan.worker.display());
    }
    if !plan.node_pack.join("__init__.py").is_file() {
        bail!(
            "DLSS interpolate pack missing: {} — expected ComfyUI-NVIDIA-DLSS-Frame-Interpolation under product Comfy custom_nodes",
            plan.node_pack.display()
        );
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
        "fps": plan.fps,
        "engine": plan.engine,
        "quality": plan.quality,
        "codec": plan.codec,
        "check_only": plan.check_only,
        "work_dir": plan
            .output
            .parent()
            .map(|p| p.join(".h3_interpolate"))
            .unwrap_or_else(|| PathBuf::from(".h3_interpolate")),
    });
    let work = plan
        .output
        .parent()
        .map(|p| p.join(".h3_interpolate"))
        .unwrap_or_else(|| PathBuf::from(".h3_interpolate"));
    std::fs::create_dir_all(&work)?;
    let req_path = work.join("request.json");
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;

    let verbose = args.verbose;
    let python = plan.python.clone();
    let worker = plan.worker.clone();
    let out = plan.output.clone();
    let h3 = plan.h3_root.clone();
    let check_only = plan.check_only;

    let launch = || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
        super::paths::forward_extra_dit_roots(&mut cmd);
        if let Ok(ff) = crate::host::paths::find_ffmpeg() {
            cmd.env("GEMMY_FFMPEG", &ff);
            cmd.env("DLSS_FFMPEG_PATH", &ff);
            let probe = ff.with_file_name(if ff.extension().is_some() {
                "ffprobe.exe"
            } else {
                "ffprobe"
            });
            if probe.is_file() {
                cmd.env("DLSS_FFPROBE_PATH", probe);
            }
        }
        cmd.current_dir(&h3);
        cmd.arg(&worker).arg("--request").arg(&req_path);
        let mut spec = WorkerSpec::new(cmd, "video.h3.interpolate").verbose(verbose);
        if !check_only {
            spec = spec.expect_outputs(vec![
                OutputExpectation::video(out.clone()).labeled("H3 DLSS interpolate mp4"),
            ]);
        }
        spec.run_inherited().map(|_| ())
    };

    if check_only {
        println!("[h3 interpolate] checking NVIDIA DLSS Frame Generation runtimes");
        launch()?;
    } else {
        println!(
            "[h3 interpolate] DLSS {} fps {} → {}",
            plan.engine,
            plan.fps,
            plan.output.display()
        );
        println!("[h3 interpolate] unloading Gemmy before DLSS Frame Generation...");
        with_gpu_handoff(verbose, launch)?;
    }

    if !check_only {
        println!("wrote H3 interpolated video: {}", plan.output.display());
        println!(
            "[h3 interpolate] do not feed this MP4 back into continue/loop encode"
        );
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "interpolate",
                "plan": plan,
                "output": plan.output,
            }))?
        );
    }
    Ok(())
}

fn build_plan(args: &H3InterpolateArgs) -> Result<InterpolatePlan> {
    let input = if args.check {
        args.input
            .as_ref()
            .map(|p| abs(p))
            .transpose()?
            .unwrap_or_else(|| PathBuf::from("check.mp4"))
    } else {
        let Some(path) = args.input.as_ref() else {
            bail!("--input is required unless --check");
        };
        abs(path)?
    };
    let output = match &args.output {
        Some(p) => absolute_path(p)?,
        None => {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            let parent = input.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            absolute_path(&parent.join(format!("{stem}_{}fps.mp4", args.fps.as_str())))?
        }
    };
    Ok(InterpolatePlan {
        input,
        output,
        fps: args.fps.as_str().to_string(),
        engine: args.engine.as_str().to_string(),
        quality: args.quality.as_str().to_string(),
        codec: args.codec.as_str().to_string(),
        check_only: args.check,
        worker: interpolate_worker_script(),
        python: h3_python(),
        h3_root: h3_root(),
        node_pack: comfy_root().join(r"custom_nodes\ComfyUI-NVIDIA-DLSS-Frame-Interpolation"),
        package: "comfy-workflow-h3-dlss-interpolate",
    })
}
