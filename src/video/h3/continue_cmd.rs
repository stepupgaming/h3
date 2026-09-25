//! `h3 continue` — same-shot multi-window continuation.
//!
//! Default: Comfy native latent guide (`native_guide`). `--legacy-masked-av`
//! is freeze-prefix copy. WanGP last-RGB is `--legacy-fl2va`.

use super::args::{H3ContinueArgs, H3Mode};
use super::canvas::resolve_sample_canvas;
use super::generate::{
    av_sidecar_paths, parse_ref_mods, snap_av_context_frames, snap_h3_frames, H3RefModUse,
};
use super::models::validate_runtime_ready;
use super::paths::{
    abs, checkpoints_root, comfy_continue_worker_script, continue_script, default_dit_weights,
    default_dit_weights_for_mode, h3_python, h3_root, read_sidecar_vae, resolve_video_vae,
    EROS_REF2VA_STEPS,
};
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::paths::{default_output_path, find_ffmpeg};
use crate::host::util::{absolute_path, timestamp_slug};
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
struct ContinuePlan {
    path: String,
    prompt: String,
    input: Option<PathBuf>,
    start_image: Option<PathBuf>,
    weights: PathBuf,
    canvas: Option<String>,
    width: u32,
    height: u32,
    megapixels: f64,
    window: u32,
    num_windows: u32,
    total_frames: Option<u32>,
    context_frames: u32,
    context_snap: u32,
    first_window: String,
    overlap: u32,
    steps: u32,
    seed: u64,
    seed_mode: String,
    quality: String,
    attn: String,
    compile_ir: bool,
    legacy_masked_av: bool,
    legacy_fl2va: bool,
    memory_frames: u32,
    anchor_frames: u32,
    te_memory: bool,
    no_te_memory: bool,
    dit_memory_refs: bool,
    keep_segs: bool,
    output: PathBuf,
    work_dir: PathBuf,
    h3_root: PathBuf,
    checkpoints_root: PathBuf,
    python: PathBuf,
    script: PathBuf,
    video_vae: String,
    graph_mode: String,
    ref_mod: Vec<H3RefModUse>,
    ref_image: Option<PathBuf>,
}

pub(crate) fn run_continue(args: H3ContinueArgs, _config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;

    if args.dry_run_plan {
        let v = serde_json::to_value(&plan)?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            println!("[h3 continue] dry-run plan");
            println!("[h3 continue] path={}", plan.path);
            println!(
                "[h3 continue] {}x{} (~{:.2} MP{}) window={} steps={} seed={} seed_mode={}",
                plan.width,
                plan.height,
                plan.megapixels,
                plan
                    .canvas
                    .as_ref()
                    .map(|c| format!(", {c}"))
                    .unwrap_or_default(),
                plan.window,
                plan.steps,
                plan.seed,
                plan.seed_mode
            );
            println!("[h3 continue] num_windows={}", plan.num_windows);
            if let Some(t) = plan.total_frames {
                println!("[h3 continue] total_frames={t}");
            }
            if plan.legacy_fl2va {
                println!("[h3 continue] overlap={}", plan.overlap);
            } else {
                println!(
                    "[h3 continue] {} context_frames={} snap={} first_window={}",
                    plan.path, plan.context_frames, plan.context_snap, plan.first_window
                );
            }
            println!("[h3 continue] weights={}", plan.weights.display());
            println!("[h3 continue] video_vae={}", plan.video_vae);
            println!("[h3 continue] output={}", plan.output.display());
            println!("[h3 continue] work_dir={}", plan.work_dir.display());
            if let Some(p) = &plan.input {
                println!("[h3 continue] input={}", p.display());
            }
            if let Some(p) = &plan.start_image {
                println!("[h3 continue] start_image={}", p.display());
            }
            println!("[h3 continue] graph_mode={}", plan.graph_mode);
            if !plan.ref_mod.is_empty() {
                println!(
                    "[h3 continue] refmods={}",
                    plan.ref_mod
                        .iter()
                        .map(|m| m.spec())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            if let Some(p) = &plan.ref_image {
                println!("[h3 continue] ref_image={}", p.display());
            }
            if plan.legacy_fl2va {
                println!(
                    "[h3 continue] memory_frames={} anchor_frames={} te_memory={} dit_memory_refs={}",
                    plan.memory_frames,
                    plan.anchor_frames,
                    plan.te_memory && !plan.no_te_memory,
                    plan.dit_memory_refs
                );
            }
        }
        return Ok(());
    }

    if plan.legacy_fl2va {
        return run_legacy_fl2va(&args, plan);
    }
    run_comfy_continue(&args, plan)
}

fn run_comfy_continue(args: &H3ContinueArgs, plan: ContinuePlan) -> Result<()> {
    validate_runtime_ready()?;
    if !plan.weights.is_file() {
        bail!("DiT weights missing: {}", plan.weights.display());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.script.is_file() {
        bail!("continue worker missing: {}", plan.script.display());
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::create_dir_all(&plan.work_dir)?;

    let refs_json: Vec<_> = plan
        .ref_image
        .as_ref()
        .map(|p| vec![json!({"kind": "image", "path": p})])
        .unwrap_or_default();

    let request = json!({
        "prompt": plan.prompt,
        "compile_ir": plan.compile_ir,
        "width": plan.width,
        "height": plan.height,
        "window_frames": plan.window,
        "num_windows": plan.num_windows,
        "context_frames": plan.context_snap,
        "steps": plan.steps,
        "seed": plan.seed,
        "seed_inc": plan.seed_mode == "inc",
        "input": plan.input,
        "start_image": plan.start_image,
        "output": plan.output,
        "work_dir": plan.work_dir,
        "h3_root": plan.h3_root,
        "checkpoints_root": plan.checkpoints_root,
        "python": plan.python,
        "weights": plan.weights,
        "video_vae": plan.video_vae,
        "mode": plan.graph_mode,
        "refmods": plan.ref_mod,
        "refs": refs_json,
        "quality": plan.quality,
        "attn": plan.attn,
        "keep_segs": plan.keep_segs,
        "legacy_masked_av": plan.legacy_masked_av,
    });
    let request_path = plan.work_dir.join("request.json");
    std::fs::write(&request_path, serde_json::to_vec_pretty(&request)?)
        .with_context(|| format!("write {}", request_path.display()))?;

    println!(
        "[h3 continue] path={} {}x{} window={} snap={} windows={} attn={}",
        plan.path,
        plan.width, plan.height, plan.window, plan.context_snap, plan.num_windows, plan.attn
    );
    println!("[h3 continue] work_dir={}", plan.work_dir.display());
    println!("[h3 continue] output={}", plan.output.display());
    println!("[h3 continue] unloading Gemmy before H3 continue...");

    let verbose = args.verbose;
    let python = plan.python.clone();
    let script = plan.script.clone();
    let out_mp4 = plan.output.clone();
    let ckpt_root = plan.checkpoints_root.clone();
    let h3_root = plan.h3_root.clone();
    let req_path = request_path.clone();

    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", ckpt_root.as_os_str());
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        super::paths::forward_refmods_dir(&mut cmd);
        cmd.current_dir(&h3_root);
        cmd.arg(&script).arg("--request").arg(&req_path);
        WorkerSpec::new(cmd, "video.h3.continue")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(out_mp4.clone()).labeled("H3 continue mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    println!("wrote H3 continue video: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "engine": "h3",
                "command": "continue",
                "path": plan.path,
                "plan": plan,
                "output": plan.output,
                "work_dir": plan.work_dir,
            }))?
        );
    }
    Ok(())
}

fn run_legacy_fl2va(args: &H3ContinueArgs, plan: ContinuePlan) -> Result<()> {
    validate_runtime_ready()?;
    if !plan.weights.is_file() {
        bail!("DiT weights missing: {}", plan.weights.display());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.script.is_file() {
        bail!("continue script missing: {}", plan.script.display());
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::create_dir_all(&plan.work_dir)?;

    let prompt_path = plan.work_dir.join("prompt.txt");
    std::fs::write(&prompt_path, plan.prompt.as_bytes())
        .with_context(|| format!("write {}", prompt_path.display()))?;

    println!(
        "[h3 continue] path=legacy-fl2va {}x{} window={} overlap={} steps={} attn={}",
        plan.width, plan.height, plan.window, plan.overlap, plan.steps, plan.attn
    );
    println!("[h3 continue] work_dir={}", plan.work_dir.display());
    println!("[h3 continue] output={}", plan.output.display());
    println!("[h3 continue] unloading Gemmy before H3 continue...");

    let verbose = args.verbose;
    let python = plan.python.clone();
    let script = plan.script.clone();
    let out_mp4 = plan.output.clone();
    let work_dir = plan.work_dir.clone();
    let ckpt_root = plan.checkpoints_root.clone();
    let h3_root = plan.h3_root.clone();
    let weights = plan.weights.clone();
    let start_image = plan.start_image.clone();
    let keep_segs = plan.keep_segs;

    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", ckpt_root.as_os_str());
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        cmd.current_dir(&h3_root);
        cmd.arg(&script);
        cmd.arg(&weights);
        cmd.arg("--prompt-file").arg(&prompt_path);
        cmd.arg("--out").arg(&work_dir);
        cmd.arg("--width").arg(plan.width.to_string());
        cmd.arg("--height").arg(plan.height.to_string());
        cmd.arg("--window").arg(plan.window.to_string());
        cmd.arg("--overlap").arg(plan.overlap.to_string());
        cmd.arg("--steps").arg(plan.steps.to_string());
        cmd.arg("--seed").arg(plan.seed.to_string());
        cmd.arg("--seed-mode").arg(&plan.seed_mode);
        cmd.arg("--device").arg("cuda");
        cmd.arg("--attn").arg(&plan.attn);
        cmd.arg("--sage-backend").arg("auto");
        cmd.arg("--num-windows").arg(plan.num_windows.to_string());
        if let Some(t) = plan.total_frames {
            cmd.arg("--total-frames").arg(t.to_string());
        }
        if let Some(img) = &start_image {
            cmd.arg("--start-image").arg(img);
        }
        if plan.memory_frames > 0 {
            cmd.arg("--memory-frames")
                .arg(plan.memory_frames.to_string());
        }
        if plan.anchor_frames > 0 {
            cmd.arg("--anchor-frames")
                .arg(plan.anchor_frames.to_string());
        }
        if plan.te_memory {
            cmd.arg("--te-memory");
        }
        if plan.no_te_memory {
            cmd.arg("--no-te-memory");
        }
        if plan.dit_memory_refs {
            cmd.arg("--dit-memory-refs");
        }
        if keep_segs {
            cmd.arg("--keep-segs");
        }

        WorkerSpec::new(cmd, "video.h3.continue")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(work_dir.join("output.mp4")).labeled("H3 continue mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    let produced = work_dir.join("output.mp4");
    if !produced.is_file() {
        let silent = work_dir.join("final").join("video.mp4");
        if silent.is_file() {
            std::fs::copy(&silent, &out_mp4)
                .with_context(|| format!("copy {} → {}", silent.display(), out_mp4.display()))?;
        } else {
            bail!("continue produced no output.mp4 under {}", work_dir.display());
        }
    } else if produced != out_mp4 {
        std::fs::copy(&produced, &out_mp4)
            .with_context(|| format!("copy {} → {}", produced.display(), out_mp4.display()))?;
    }

    println!("wrote H3 continue video: {}", out_mp4.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "engine": "h3",
                "command": "continue",
                "path": "legacy-fl2va",
                "plan": plan,
                "output": out_mp4,
                "work_dir": work_dir,
            }))?
        );
    }
    Ok(())
}

fn build_plan(args: &H3ContinueArgs) -> Result<ContinuePlan> {
    let prompt = resolve_prompt(args)?;
    if prompt.trim().is_empty() {
        bail!("prompt is required (--prompt or --prompt-file)");
    }
    if args.steps == 0 || args.steps > 100 {
        bail!("--steps must be in 1..=100");
    }
    let canvas = resolve_sample_canvas(args.canvas, args.width, args.height)?;
    if args.verbose {
        eprintln!("[h3 continue] canvas={}", canvas.label());
    }
    if args.window < 5 {
        bail!("--window must be >= 5");
    }
    if args.legacy_fl2va && args.overlap == 0 {
        bail!("--overlap must be >= 1 (WanGP FL2VA uses 1)");
    }
    if let Some(n) = args.num_windows {
        if n == 0 {
            bail!("--num-windows must be >= 1");
        }
    }
    if let Some(t) = args.total_frames {
        if t == 0 {
            bail!("--total-frames must be >= 1");
        }
    }
    if args.dit_memory_refs {
        let ref_w = super::paths::stock_ref2va_backup_path();
        if args.weights.is_none() && !ref_w.is_file() {
            bail!(
                "--dit-memory-refs needs Ref2VA DiT at {} (or pass --weights)",
                ref_w.display()
            );
        }
    }

    let input = match &args.input {
        Some(p) => {
            let p = abs(p)?;
            if !args.dry_run_plan && !p.is_file() {
                bail!("--input not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };

    let start_image = match &args.start_image {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() {
                bail!("start-image not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };

    let refmods = parse_ref_mods(&args.ref_mod)?;
    if args.ref_image.len() > 1 {
        bail!(
            "native-guide continue supports one --ref-image (got {})",
            args.ref_image.len()
        );
    }
    let uses_ref2va = !refmods.is_empty() || !args.ref_image.is_empty();
    if uses_ref2va && args.legacy_fl2va {
        bail!("--ref-mod / --ref-image need native-guide continue (not --legacy-fl2va)");
    }
    if uses_ref2va && args.legacy_masked_av {
        bail!("--ref-mod / --ref-image need native-guide continue (not --legacy-masked-av)");
    }
    if uses_ref2va && start_image.is_some() {
        bail!("--ref-mod / --ref-image cannot stack with --start-image on continue");
    }
    if uses_ref2va && args.dit_memory_refs {
        bail!("--ref-mod / --ref-image cannot stack with --dit-memory-refs");
    }

    let ref_image = match args.ref_image.first() {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() {
                bail!("--ref-image not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };

    let weights = match &args.weights {
        Some(p) => abs(p)?,
        None if args.dit_memory_refs => super::paths::stock_ref2va_backup_path(),
        None if uses_ref2va => default_dit_weights_for_mode(H3Mode::Ref2va),
        None => default_dit_weights(),
    };

    let output = absolute_path(&args.output.clone().unwrap_or_else(|| {
        default_output_path(format!("h3_continue_{}.mp4", timestamp_slug()))
    }))?;

    let work_dir = match &args.work_dir {
        Some(p) => absolute_path(p)?,
        None => output
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(format!(".h3_continue_{}", timestamp_slug())),
    };

    let window = snap_h3_frames(args.window);
    let (num_windows, total_frames) = match (args.num_windows, args.total_frames) {
        (Some(n), t) => (n, t),
        (None, Some(t)) => {
            let target = snap_h3_frames(t);
            if target <= window {
                (1, Some(target))
            } else {
                let rest = target.saturating_sub(window);
                (1 + rest.div_ceil(window), Some(target))
            }
        }
        (None, None) => (2, None),
    };

    let same_shot = if args.legacy_masked_av {
        "masked_av"
    } else {
        "native_guide"
    };
    let first_window = if let Some(inp) = &input {
        let (st, _) = av_sidecar_paths(inp);
        if st.is_file() {
            same_shot
        } else {
            "vae_tail"
        }
    } else {
        "none"
    };
    if uses_ref2va && first_window == "vae_tail" {
        bail!(
            "--ref-mod / --ref-image need a .h3av sidecar next to --input \
             (VAE-tail import has no RefMod package yet)"
        );
    }
    let available = if input.is_some() { window.max(39) } else { window };
    let context_snap = if args.legacy_fl2va {
        0
    } else {
        snap_av_context_frames(args.context_frames, available)?
    };

    let seed_mode = if args.seed_inc { "inc" } else { "same" };
    let mut compile_ir = args.compile_ir && !args.no_compile_ir;
    if !refmods.is_empty() && ref_image.is_none() {
        // Text Encode presents `<Picture n>` / `<Video n>`. Do not rewrite those tags.
        compile_ir = false;
    }
    let mut steps = args.steps;
    if uses_ref2va && args.weights.is_none() && steps == 20 {
        steps = EROS_REF2VA_STEPS;
    }
    let graph_mode = if uses_ref2va {
        "ref2va".into()
    } else if start_image.is_some() && input.is_none() {
        "i2v".into()
    } else {
        "t2va".into()
    };
    let script = if args.legacy_fl2va {
        continue_script()
    } else {
        comfy_continue_worker_script()
    };

    let sidecar_vae = input
        .as_ref()
        .and_then(|p| read_sidecar_vae(p));
    let video_vae = resolve_video_vae(
        &args.vae_select,
        sidecar_vae.as_deref(),
        false,
        args.vae_select.video_vae.is_some() || !args.dry_run_plan,
    )?;

    Ok(ContinuePlan {
        path: if args.legacy_fl2va {
            "legacy-fl2va".into()
        } else {
            same_shot.into()
        },
        prompt,
        input,
        start_image,
        weights,
        canvas: canvas.preset.map(|p| p.as_str().into()),
        width: canvas.width,
        height: canvas.height,
        megapixels: canvas.megapixels,
        window,
        num_windows,
        total_frames,
        context_frames: args.context_frames,
        context_snap,
        first_window: first_window.into(),
        overlap: args.overlap,
        steps,
        seed: args.seed,
        seed_mode: seed_mode.into(),
        quality: args.quality.as_str().into(),
        attn: args.quality.attn().into(),
        compile_ir,
        legacy_masked_av: args.legacy_masked_av,
        legacy_fl2va: args.legacy_fl2va,
        memory_frames: args.memory_frames,
        anchor_frames: args.anchor_frames,
        te_memory: args.te_memory,
        no_te_memory: args.no_te_memory,
        dit_memory_refs: args.dit_memory_refs,
        keep_segs: args.keep_segs,
        output,
        work_dir,
        h3_root: h3_root(),
        checkpoints_root: checkpoints_root(),
        python: h3_python(),
        script,
        video_vae: video_vae.filename,
        graph_mode,
        ref_mod: refmods,
        ref_image,
    })
}

fn resolve_prompt(args: &H3ContinueArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        let path = abs(path)?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()))?;
        return Ok(text);
    }
    Ok(args.prompt.clone().unwrap_or_default())
}

/// Stream-concat already-trimmed H3 segments. Re-encodes if copy fails.
pub(crate) fn concat_h3_segments(segs: &[PathBuf], output: &Path) -> Result<()> {
    if segs.is_empty() {
        bail!("no accepted segments to assemble");
    }
    if segs.len() == 1 {
        if segs[0] != output {
            std::fs::copy(&segs[0], output)
                .with_context(|| format!("copy {} → {}", segs[0].display(), output.display()))?;
        }
        return Ok(());
    }
    let ffmpeg = find_ffmpeg()?;
    let parent = output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&parent)?;
    let list = parent.join(format!(
        ".h3_concat_{}.txt",
        output
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out")
    ));
    let mut f = std::fs::File::create(&list).with_context(|| format!("write {}", list.display()))?;
    for seg in segs {
        let p = seg.display().to_string().replace('\\', "/").replace('\'', r"'\''");
        writeln!(f, "file '{p}'")?;
    }
    drop(f);
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-f", "concat", "-safe", "0", "-i"])
        .arg(&list)
        .args(["-c", "copy"])
        .arg(output)
        .status()
        .with_context(|| format!("run {}", ffmpeg.display()))?;
    if !status.success() {
        let status = std::process::Command::new(&ffmpeg)
            .args(["-y", "-f", "concat", "-safe", "0", "-i"])
            .arg(&list)
            .args(["-c:v", "libx264", "-crf", "18", "-c:a", "aac"])
            .arg(output)
            .status()
            .with_context(|| format!("re-encode {}", ffmpeg.display()))?;
        if !status.success() {
            bail!("ffmpeg concat failed for {}", output.display());
        }
    }
    let _ = std::fs::remove_file(list);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> H3ContinueArgs {
        let mut argv = vec!["gemmy-h3-continue"];
        argv.extend_from_slice(args);
        H3ContinueArgs::try_parse_from(argv).unwrap()
    }

    #[test]
    fn refmod_only_switches_to_eros_native_guide() {
        let args = parse(&[
            "--prompt",
            "The woman in <Picture 1> walks on",
            "--ref-mod",
            "hero",
            "--window",
            "124",
            "--num-windows",
            "1",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.path, "native_guide");
        assert_eq!(plan.graph_mode, "ref2va");
        assert_eq!(plan.ref_mod.len(), 1);
        assert_eq!(plan.ref_mod[0].name, "hero");
        assert!(!plan.compile_ir);
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(
            name.contains("10Eros_Max_h3_TURBO_ref2va_beta2"),
            "dit={name}"
        );
        assert!(plan.prompt.contains("<Picture 1>"));
    }

    #[test]
    fn ref_image_switches_to_eros_and_keeps_ir() {
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = parse(&[
            "--prompt",
            "walks toward camera",
            "--ref-image",
            still.to_str().unwrap(),
            "--window",
            "124",
            "--num-windows",
            "1",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.graph_mode, "ref2va");
        assert!(plan.ref_image.is_some());
        assert!(plan.ref_mod.is_empty());
        assert!(plan.compile_ir);
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
    }

    #[test]
    fn default_continue_stays_fl2va() {
        let args = parse(&[
            "--prompt",
            "the same scene continues",
            "--window",
            "124",
            "--num-windows",
            "1",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.path, "native_guide");
        assert_eq!(plan.graph_mode, "t2va");
        assert!(plan.ref_mod.is_empty());
        assert_eq!(plan.steps, 20);
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(name.contains("fl2va"), "dit={name}");
    }

    #[test]
    fn refmod_rejected_on_legacy_masked_av() {
        let args = parse(&[
            "--prompt",
            "walks",
            "--ref-mod",
            "hero",
            "--legacy-masked-av",
            "--dry-run-plan",
        ]);
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--legacy-masked-av"), "{err}");
    }

    #[test]
    fn refmod_needs_sidecar_not_vae_tail() {
        let dir = tempfile::tempdir().unwrap();
        let mp4 = dir.path().join("imported.mp4");
        std::fs::write(&mp4, b"not-a-real-mp4").unwrap();
        let args = parse(&[
            "--prompt",
            "The woman in <Picture 1> walks on",
            "--input",
            mp4.to_str().unwrap(),
            "--ref-mod",
            "hero",
            "--dry-run-plan",
        ]);
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains(".h3av"), "{err}");
    }

    #[test]
    fn two_ref_images_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.png");
        let b = dir.path().join("b.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&a)
            .unwrap();
        image::RgbImage::from_pixel(32, 32, image::Rgb([40, 40, 200]))
            .save(&b)
            .unwrap();
        let args = parse(&[
            "--prompt",
            "walks",
            "--ref-image",
            a.to_str().unwrap(),
            "--ref-image",
            b.to_str().unwrap(),
            "--dry-run-plan",
        ]);
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("one --ref-image"), "{err}");
    }
}
