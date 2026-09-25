//! `h3 sprites` — PixelForge keyed sprite loop from an H3 clip.

use super::args::H3SpritesArgs;
use super::paths::{abs, checkpoints_root, comfy_root, h3_python, h3_root, sprites_worker_script};
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
struct SpritesPlan {
    input: PathBuf,
    output: PathBuf,
    frames: u32,
    every_nth: u32,
    r#loop: bool,
    loop_mode: String,
    atlas: bool,
    key_color: String,
    worker: PathBuf,
    python: PathBuf,
    h3_root: PathBuf,
    node_pack: PathBuf,
    package: &'static str,
}

pub(crate) fn run_sprites(args: H3SpritesArgs, _config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;
    if args.dry_run_plan {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("[h3 sprites] dry-run plan");
            println!("[h3 sprites] input={}", plan.input.display());
            println!("[h3 sprites] output={}", plan.output.display());
            println!(
                "[h3 sprites] frames={} every_nth={} loop_mode={} atlas={}",
                plan.frames, plan.every_nth, plan.loop_mode, plan.atlas
            );
            println!("[h3 sprites] package={}", plan.package);
            println!("[h3 sprites] node_pack={}", plan.node_pack.display());
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
        bail!("sprites worker missing: {}", plan.worker.display());
    }
    if !plan.node_pack.join("__init__.py").is_file() {
        bail!(
            "PixelForge pack missing: {} — expected ComfyUI-PixelForge-H3 under product Comfy custom_nodes",
            plan.node_pack.display()
        );
    }
    std::fs::create_dir_all(&plan.output)
        .with_context(|| format!("create {}", plan.output.display()))?;

    let request = json!({
        "input": plan.input,
        "output": plan.output,
        "h3_root": plan.h3_root,
        "checkpoints_root": checkpoints_root(),
        "python": plan.python,
        "node_pack": plan.node_pack,
        "frames": plan.frames,
        "max_frames": plan.frames,
        "every_nth": plan.every_nth,
        "loop": plan.r#loop,
        "loop_mode": plan.loop_mode,
        "atlas": plan.atlas,
        "key_color": plan.key_color,
        "work_dir": plan.output.join(".work"),
    });
    let work = plan.output.join(".work");
    std::fs::create_dir_all(&work)?;
    let req_path = work.join("request.json");
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;

    println!(
        "[h3 sprites] PixelForge {} → {}",
        plan.input.display(),
        plan.output.display()
    );

    let verbose = args.verbose;
    let python = plan.python.clone();
    let worker = plan.worker.clone();
    let out = plan.output.clone();
    let h3 = plan.h3_root.clone();
    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
        super::paths::forward_extra_dit_roots(&mut cmd);
        if let Ok(ff) = crate::host::paths::find_ffmpeg() {
            cmd.env("GEMMY_FFMPEG", ff);
        }
        cmd.current_dir(&h3);
        cmd.arg(&worker).arg("--request").arg(&req_path);
        WorkerSpec::new(cmd, "video.h3.sprites")
            .verbose(verbose)
            .expect_outputs(vec![OutputExpectation::directory(out.clone())
                .labeled("H3 sprites dir")])
            .run_inherited()
            .map(|_| ())
    })?;

    println!("wrote H3 sprites: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "sprites",
                "plan": plan,
            }))?
        );
    }
    Ok(())
}

fn build_plan(args: &H3SpritesArgs) -> Result<SpritesPlan> {
    let input = abs(&args.input)?;
    let output = match &args.output {
        Some(p) => absolute_path(p)?,
        None => {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("sprites");
            let parent = input.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            absolute_path(&parent.join(stem))?
        }
    };
    if args.frames == 0 {
        bail!("--frames must be >= 1 (use a large number to keep the whole clip after decimate)");
    }
    let loop_mode = if args.r#loop {
        "auto".to_string()
    } else {
        "off".to_string()
    };
    Ok(SpritesPlan {
        input,
        output,
        frames: args.frames,
        every_nth: args.every_nth,
        r#loop: args.r#loop,
        loop_mode,
        atlas: args.atlas,
        key_color: args.key_color.clone(),
        worker: sprites_worker_script(),
        python: h3_python(),
        h3_root: h3_root(),
        node_pack: comfy_root().join(r"custom_nodes\ComfyUI-PixelForge-H3"),
        package: "comfy-workflow-h3-pixelforge",
    })
}
