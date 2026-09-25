//! `h3 edit` — SAM3 track + Eros crop sample + uncrop.
//!
//! Source: ganloss `2026-08-31 minimax_h3_r2v_video_mask_edit.json` +
//! Veteran AI video `PJWfUAO1Oco`. Pixel mask from SAM3.1 (or `--mask`);
//! crop/uncrop is Gemmy-owned (not GPL MaskVidExperiments). Eros 8-step
//! euler/simple on the crop. Source audio is muxed back.
//!
//! Mask is a gated stage: track writes a tinted overlay, `gemmy analyze`
//! checks it (same 12B path as loop `--review`), then Eros runs only on
//! pass. `--stop-after-mask` exits after that gate. Resume with `--mask`.

use super::args::H3EditArgs;
use super::generate::snap_h3_frames;
use super::paths::{
    abs, checkpoints_root, comfy_root, default_dit_weights_for_mode, edit_worker_script,
    h3_python, h3_root, resolve_video_vae, sam3_checkpoint_path, EROS_REF2VA_STEPS,
};
use super::scene_ir::compile_edit_ir;
use crate::host::cli_output::CliError;
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::paths::default_output_path;
use crate::host::util::{absolute_path, timestamp_slug};
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const FPS: u32 = 24;
const SAM3_CKPT: &str = "sam3.1_multiplex_fp16.safetensors";
const EMPTY_COVERAGE: f64 = 0.002;

#[derive(Debug, Clone, Serialize)]
struct EditPlan {
    video: PathBuf,
    ref_images: Vec<PathBuf>,
    mask: Option<PathBuf>,
    mask_prompt: String,
    object_id: String,
    max_objects: u32,
    confidence: f64,
    crop_mode: String,
    crop_scale: f64,
    crop_mp: f64,
    feather: u32,
    mask_grow: u32,
    prompt: String,
    frames: u32,
    duration_s: f64,
    steps: u32,
    seed: u64,
    weights: PathBuf,
    sam3_ckpt: String,
    sam3_path: PathBuf,
    output: PathBuf,
    h3_root: PathBuf,
    checkpoints_root: PathBuf,
    comfy_root: PathBuf,
    python: PathBuf,
    worker: PathBuf,
    keep_work: bool,
    verify_mask: bool,
    stop_after_mask: bool,
    mask_out: Option<PathBuf>,
    mask_overlay_out: Option<PathBuf>,
    wants_sample: bool,
    video_vae: String,
}

#[derive(Debug, Clone, Serialize)]
struct MaskVerdict {
    pass: bool,
    notes: String,
    missing: Vec<String>,
    leaks: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TrackArtifacts {
    mask: PathBuf,
    overlay: PathBuf,
    overlay_mid: PathBuf,
    coverage_mean: f64,
    frames: u32,
}

pub(crate) fn run_edit(args: H3EditArgs, config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;
    if args.dry_run_plan {
        if args.json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("[h3 edit] dry-run plan");
            println!("[h3 edit] video={}", plan.video.display());
            println!(
                "[h3 edit] refs={} mask_prompt={:?} object={} crop={} crop_mp={}",
                plan.ref_images.len(),
                plan.mask_prompt,
                plan.object_id,
                plan.crop_mode,
                plan.crop_mp
            );
            println!(
                "[h3 edit] frames={} (~{:.2}s) steps={} seed={} verify={} stop_after_mask={}",
                plan.frames,
                plan.duration_s,
                plan.steps,
                plan.seed,
                plan.verify_mask,
                plan.stop_after_mask
            );
            println!("[h3 edit] weights={}", plan.weights.display());
            println!("[h3 edit] video_vae={}", plan.video_vae);
            println!("[h3 edit] sam3={}", plan.sam3_path.display());
            if plan.wants_sample {
                println!("[h3 edit] output={}", plan.output.display());
            } else {
                println!("[h3 edit] stage=track (no Eros)");
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
        bail!("edit worker missing: {}", plan.worker.display());
    }
    if plan.mask.is_none() && !plan.sam3_path.is_file() {
        bail!(
            "SAM3.1 checkpoint missing: {}\n\
             Download Comfy-Org/sam3.1 `checkpoints/sam3.1_multiplex_fp16.safetensors`\n\
             into that path (or F:\\Models\\sam3\\checkpoints\\). See docs\\MODEL_LOCATIONS.md.",
            plan.sam3_path.display()
        );
    }
    if plan.wants_sample {
        if let Some(parent) = plan.output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
    }

    let work = plan
        .output
        .parent()
        .map(|p| p.join(format!(".h3_edit_{}", timestamp_slug())))
        .unwrap_or_else(|| PathBuf::from(format!(".h3_edit_{}", timestamp_slug())));
    std::fs::create_dir_all(&work)?;

    println!(
        "[h3 edit] {} mask_prompt={:?} crop={} verify={} → {}",
        plan.video.display(),
        plan.mask_prompt,
        plan.crop_mode,
        plan.verify_mask,
        if plan.wants_sample {
            plan.output.display().to_string()
        } else {
            format!("{} (track only)", work.display())
        }
    );

    let track = run_track(&plan, &args, &work)?;
    copy_if_set(&track.mask, plan.mask_out.as_ref())?;
    copy_if_set(&track.overlay, plan.mask_overlay_out.as_ref())?;

    let verdict = if plan.verify_mask {
        Some(run_mask_verify(&plan, &track, &work, args.verbose)?)
    } else {
        println!("[h3 edit] mask verify skipped (--no-verify-mask)");
        None
    };

    if let Some(v) = &verdict {
        if !v.pass {
            println!("[h3 edit] overlay={}", track.overlay.display());
            println!("[h3 edit] mask={}", track.mask.display());
            println!("[h3 edit] kept work dir {}", work.display());
            return Err(mask_failed_error(&plan, &track, v, &work).into());
        }
        println!("[h3 edit] mask verify pass");
        if !v.notes.trim().is_empty() {
            println!("[h3 edit] {}", v.notes.trim());
        }
    }

    write_gate_note(&work, &track, verdict.as_ref(), true)?;

    if !plan.wants_sample {
        println!("[h3 edit] overlay={}", track.overlay.display());
        println!("[h3 edit] mask={}", track.mask.display());
        println!(
            "[h3 edit] stopped before Eros (--stop-after-mask). Resume with --mask {}",
            track.mask.display()
        );
        println!("[h3 edit] kept work dir {}", work.display());
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": true,
                    "command": "edit",
                    "stage": "track",
                    "mask": track.mask,
                    "overlay": track.overlay,
                    "overlay_mid": track.overlay_mid,
                    "verdict": verdict,
                    "work_dir": work,
                }))?
            );
        }
        return Ok(());
    }

    run_sample(&plan, &args, &work, &track)?;

    if !plan.keep_work {
        let _ = std::fs::remove_dir_all(&work);
    } else {
        println!("[h3 edit] kept work dir {}", work.display());
    }
    println!("wrote H3 edit video: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "edit",
                "plan": plan,
                "output": plan.output,
                "mask": track.mask,
                "overlay": track.overlay,
                "verdict": verdict,
            }))?
        );
    }
    let _ = config;
    Ok(())
}

fn run_track(plan: &EditPlan, args: &H3EditArgs, work: &Path) -> Result<TrackArtifacts> {
    let req_path = work.join("track_request.json");
    std::fs::write(
        &req_path,
        serde_json::to_vec_pretty(&worker_request(plan, work, "track", plan.mask.clone()))?,
    )?;
    println!("[h3 edit] track…");
    let mask_mp4 = work.join("mask.mp4");
    let overlay_mp4 = work.join("overlay.mp4");
    let overlay_mid = work.join("overlay_mid.png");
    let track_json = work.join("track.json");
    let needs_sam3 = plan.mask.is_none();
    let launch = || launch_worker(plan, args, work, &req_path, "video.h3.edit.track", vec![
        OutputExpectation::video(mask_mp4.clone()).labeled("H3 edit mask"),
        OutputExpectation::video(overlay_mp4.clone()).labeled("H3 edit overlay"),
        OutputExpectation::image(overlay_mid.clone()).labeled("H3 edit overlay mid"),
        OutputExpectation::json(track_json.clone()).labeled("H3 edit track note"),
    ]);
    if needs_sam3 {
        println!("[h3 edit] unloading Gemmy before SAM3 track...");
        with_gpu_handoff(args.verbose, launch)?;
    } else {
        launch()?;
    }
    let note: Value = serde_json::from_slice(
        &std::fs::read(&track_json).with_context(|| format!("read {}", track_json.display()))?,
    )?;
    let mean = note
        .pointer("/coverage/mean")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    if mean < EMPTY_COVERAGE {
        bail!(
            "SAM3 mask is empty (mean coverage {mean:.4}). Retry --mask-prompt / --object-id, or pass --mask."
        );
    }
    Ok(TrackArtifacts {
        mask: mask_mp4,
        overlay: overlay_mp4,
        overlay_mid,
        coverage_mean: mean,
        frames: note.get("frames").and_then(Value::as_u64).unwrap_or(0) as u32,
    })
}

fn run_sample(
    plan: &EditPlan,
    args: &H3EditArgs,
    work: &Path,
    track: &TrackArtifacts,
) -> Result<()> {
    let req_path = work.join("sample_request.json");
    let mut sample_plan = plan.clone();
    sample_plan.mask_grow = 0;
    std::fs::write(
        &req_path,
        serde_json::to_vec_pretty(&worker_request(
            &sample_plan,
            work,
            "sample",
            Some(track.mask.clone()),
        ))?,
    )?;
    println!("[h3 edit] Eros sample…");
    println!("[h3 edit] unloading Gemmy before H3 mask-edit...");
    let out = plan.output.clone();
    with_gpu_handoff(args.verbose, || {
        launch_worker(
            plan,
            args,
            work,
            &req_path,
            "video.h3.edit.sample",
            vec![OutputExpectation::video(out.clone()).labeled("H3 mask-edit mp4")],
        )
    })
}

fn launch_worker(
    plan: &EditPlan,
    args: &H3EditArgs,
    _work: &Path,
    req_path: &Path,
    label: &str,
    expect: Vec<OutputExpectation>,
) -> Result<()> {
    let mut cmd = crate::host::workers::env::native_tool_command(&plan.python);
    crate::host::paths::pin_huggingface_cache(&mut cmd);
    cmd.env("PYTHONUTF8", "1");
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
    super::paths::forward_live_preview(&mut cmd);
    super::paths::forward_extra_dit_roots(&mut cmd);
    if let Some(parent) = plan.sam3_path.parent().and_then(|p| p.parent()) {
        cmd.env("GEMMY_SAM3_CHECKPOINTS", parent.as_os_str());
    }
    cmd.current_dir(&plan.h3_root);
    cmd.arg(&plan.worker).arg("--request").arg(req_path);
    WorkerSpec::new(cmd, label)
        .verbose(args.verbose)
        .expect_outputs(expect)
        .run_inherited()
        .map(|_| ())
}

fn worker_request(
    plan: &EditPlan,
    work: &Path,
    stage: &str,
    mask: Option<PathBuf>,
) -> Value {
    json!({
        "stage": stage,
        "video": plan.video,
        "ref_images": plan.ref_images,
        "mask": mask,
        "mask_prompt": plan.mask_prompt,
        "object_indices": plan.object_id,
        "max_objects": plan.max_objects,
        "detection_threshold": plan.confidence,
        "crop_mode": plan.crop_mode,
        "crop_scale": plan.crop_scale,
        "crop_mp": plan.crop_mp,
        "feather": plan.feather,
        "mask_grow": plan.mask_grow,
        "prompt": plan.prompt,
        "frames": plan.frames,
        "steps": plan.steps,
        "seed": plan.seed,
        "weights": plan.weights,
        "video_vae": plan.video_vae,
        "sam3_ckpt": plan.sam3_ckpt,
        "output": plan.output,
        "work_dir": work,
        "h3_root": plan.h3_root,
        "checkpoints_root": plan.checkpoints_root,
        "comfy_root": plan.comfy_root,
        "python": plan.python,
        "attn": "sage",
        "shift_video": 12.0,
        "shift_audio": 3.0,
    })
}

fn run_mask_verify(
    plan: &EditPlan,
    track: &TrackArtifacts,
    work: &Path,
    verbose: bool,
) -> Result<MaskVerdict> {
    let gemmy = crate::host::gemmy::require_gemmy(
        "mask verify is `gemmy analyze`. Pass --no-verify-mask to sample without it",
    )?;
    let review_txt = work.join("mask_verify.txt");
    let subject = if plan.mask_prompt.is_empty() {
        "the intended subject".to_string()
    } else {
        format!("'{}'", plan.mask_prompt)
    };
    let prompt = format!(
        "This clip is a SAM3 tracking overlay. The red-tinted region is the mask that will be edited. \
The intended subject is {subject}. \
Judge whether the mask covers the WHOLE intended subject across frames — including hair volume, \
silhouette, and limbs that belong to that subject — and whether it leaks onto other people or \
background that must stay. Fail if hair is missing, the wrong person is selected, or the mask \
is empty/near-empty. \
Reply with JSON only: {{\"pass\": true|false, \"notes\": \"...\", \"missing\": [], \"leaks\": []}}"
    );
    let fps = if plan.duration_s <= 3.0 { "4" } else { "1" };
    let mut cmd: Command = crate::host::workers::env::gemmy_child_command(&gemmy);
    cmd.arg("analyze")
        .arg("--file")
        .arg(&track.overlay)
        .arg("--file")
        .arg(&track.overlay_mid)
        .arg("--video-fps")
        .arg(fps)
        .arg("--video-max-frames")
        .arg("60")
        .arg("--output")
        .arg(&review_txt)
        .arg(&prompt);
    println!(
        "[h3 edit] mask verify via gemmy analyze (12B) → {}",
        review_txt.display()
    );
    if verbose {
        println!("[h3 edit] overlay={}", track.overlay.display());
    }
    let status = cmd.status().context("run gemmy analyze for mask verify")?;
    if !status.success() {
        bail!(
            "gemmy analyze mask verify failed (exit {:?}). Overlay: {}",
            status.code(),
            track.overlay.display()
        );
    }
    let notes = std::fs::read_to_string(&review_txt).unwrap_or_default();
    let verdict = parse_mask_verdict(&notes);
    std::fs::write(
        work.join("mask_verify.json"),
        serde_json::to_vec_pretty(&json!({
            "overlay": track.overlay,
            "overlay_mid": track.overlay_mid,
            "coverage_mean": track.coverage_mean,
            "raw": notes,
            "verdict": verdict,
        }))?,
    )?;
    Ok(verdict)
}

fn parse_mask_verdict(text: &str) -> MaskVerdict {
    let parsed = extract_json_object(text).and_then(|v| {
        let pass = v.get("pass").and_then(Value::as_bool)?;
        Some(MaskVerdict {
            pass,
            notes: v
                .get("notes")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            missing: string_list(v.get("missing")),
            leaks: string_list(v.get("leaks")),
        })
    });
    parsed.unwrap_or_else(|| MaskVerdict {
        pass: false,
        notes: format!(
            "analyze did not return pass/fail JSON; treating as fail (fail closed).\n{}",
            text.trim()
        ),
        missing: vec!["unparsed-verdict".into()],
        leaks: vec![],
    })
}

fn extract_json_object(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    let fence = trimmed
        .find("```json")
        .or_else(|| trimmed.find("```"))
        .and_then(|start| {
            let after = trimmed.get(start..)?;
            let inner = after.split_once('\n').map(|(_, rest)| rest)?;
            let end = inner.find("```").unwrap_or(inner.len());
            inner.get(..end).map(str::trim)
        });
    let candidates = [fence, Some(trimmed), find_braced(trimmed)];
    for chunk in candidates.into_iter().flatten() {
        if let Ok(v) = serde_json::from_str::<Value>(chunk) {
            if v.get("pass").is_some() {
                return Some(v);
            }
        }
    }
    find_braced(trimmed).and_then(|s| serde_json::from_str(s).ok())
}

fn find_braced(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        text.get(start..=end)
    } else {
        None
    }
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn mask_failed_error(
    plan: &EditPlan,
    track: &TrackArtifacts,
    verdict: &MaskVerdict,
    work: &Path,
) -> CliError {
    let mut msg = format!(
        "mask verify failed — overlay {} (mean coverage {:.3})",
        track.overlay.display(),
        track.coverage_mean
    );
    if !verdict.notes.trim().is_empty() {
        msg.push_str(": ");
        msg.push_str(verdict.notes.trim());
    }
    let mut err = CliError::conflict(msg)
        .with_command("video-h3-edit")
        .with_detail("overlay", track.overlay.display().to_string())
        .with_detail("mask", track.mask.display().to_string())
        .with_detail("overlay_mid", track.overlay_mid.display().to_string())
        .with_detail("work_dir", work.display().to_string())
        .with_detail("pass", false)
        .suggest(format!(
            "look at the overlay: {}",
            track.overlay.display()
        ))
        .suggest(format!(
            "retry with more hair/silhouette: --mask-grow 12 --mask \"{}\"",
            track.mask.display()
        ));
    if !plan.mask_prompt.is_empty() {
        err = err.suggest(format!(
            "or re-track: --mask-prompt {:?} --object-id {}",
            plan.mask_prompt, plan.object_id
        ));
    }
    err
}

fn write_gate_note(
    work: &Path,
    track: &TrackArtifacts,
    verdict: Option<&MaskVerdict>,
    keep: bool,
) -> Result<()> {
    let _ = keep;
    std::fs::write(
        work.join("gate.json"),
        serde_json::to_vec_pretty(&json!({
            "mask": track.mask,
            "overlay": track.overlay,
            "overlay_mid": track.overlay_mid,
            "coverage_mean": track.coverage_mean,
            "frames": track.frames,
            "verdict": verdict,
        }))?,
    )?;
    Ok(())
}

fn copy_if_set(src: &Path, dest: Option<&PathBuf>) -> Result<()> {
    let Some(dest) = dest else {
        return Ok(());
    };
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(src, dest).with_context(|| format!("copy {} → {}", src.display(), dest.display()))?;
    println!("[h3 edit] copied {} → {}", src.display(), dest.display());
    Ok(())
}

fn build_plan(args: &H3EditArgs) -> Result<EditPlan> {
    let video = abs(&args.video)?;
    if !video.is_file() {
        bail!("--video not found: {}", video.display());
    }
    let wants_sample = !args.stop_after_mask;
    if wants_sample && args.ref_image.is_empty() {
        bail!("--ref-image is required (replacement identity still)");
    }
    let mut refs = Vec::new();
    for p in &args.ref_image {
        let path = abs(p)?;
        if !path.is_file() {
            bail!("--ref-image not found: {}", path.display());
        }
        refs.push(path);
    }
    let mask = match &args.mask {
        Some(p) => {
            let path = abs(p)?;
            if !path.is_file() {
                bail!("--mask not found: {}", path.display());
            }
            Some(path)
        }
        None => None,
    };
    let mask_prompt = args
        .mask_prompt
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default();
    if mask.is_none() && mask_prompt.is_empty() {
        bail!("pass --mask-prompt (e.g. head / female) or --mask PATH");
    }
    if args.steps == 0 || args.steps > 100 {
        bail!("--steps must be in 1..=100");
    }
    if !(0.0..=1.0).contains(&args.confidence) {
        bail!("--confidence must be in 0..=1");
    }
    if args.crop_scale <= 0.0 || !args.crop_scale.is_finite() {
        bail!("--crop-scale must be finite and > 0");
    }

    let mut prompt = resolve_prompt(args)?;
    if wants_sample && prompt.trim().is_empty() {
        bail!("prompt is required (--prompt or --prompt-file)");
    }

    let frames = resolve_frames(args, &video)?;
    let duration_s = f64::from(frames) / f64::from(FPS);
    let compile_ir = args.compile_ir && !args.no_compile_ir;
    if wants_sample && compile_ir && !prompt.contains("subject_definitions") {
        prompt = compile_edit_ir(&prompt, duration_s, &mask_prompt);
    }

    let output = absolute_path(&args.output.clone().unwrap_or_else(|| {
        default_output_path(format!("h3_edit_{}.mp4", timestamp_slug()))
    }))?;
    let sam3_ckpt = args
        .sam3_ckpt
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| SAM3_CKPT.to_string());
    let sam3_path = sam3_checkpoint_path(&sam3_ckpt);
    let verify_mask = args.verify_mask && !args.no_verify_mask;
    let mask_out = args.mask_out.as_ref().map(|p| abs(p)).transpose()?;
    let mask_overlay_out = args.mask_overlay_out.as_ref().map(|p| abs(p)).transpose()?;

    Ok(EditPlan {
        video,
        ref_images: refs,
        mask,
        mask_prompt,
        object_id: args.object_id.clone(),
        max_objects: args.max_objects,
        confidence: args.confidence,
        crop_mode: args.crop_mode.as_str().into(),
        crop_scale: args.crop_scale,
        crop_mp: args.crop_mp,
        feather: args.feather,
        mask_grow: args.mask_grow,
        prompt,
        frames,
        duration_s,
        steps: if args.steps == 20 {
            EROS_REF2VA_STEPS
        } else {
            args.steps
        },
        seed: args.seed,
        weights: default_dit_weights_for_mode(super::args::H3Mode::Ref2va),
        sam3_ckpt,
        sam3_path,
        output,
        h3_root: h3_root(),
        checkpoints_root: checkpoints_root(),
        comfy_root: comfy_root(),
        python: h3_python(),
        worker: edit_worker_script(),
        keep_work: args.keep_work || args.stop_after_mask,
        verify_mask,
        stop_after_mask: args.stop_after_mask,
        mask_out,
        mask_overlay_out,
        wants_sample,
        video_vae: resolve_video_vae(
            &args.vae_select,
            None,
            false,
            args.vae_select.video_vae.is_some() || !args.dry_run_plan,
        )?
        .filename,
    })
}

fn resolve_prompt(args: &H3EditArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        let path = abs(path)?;
        return std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()));
    }
    Ok(args.prompt.clone().unwrap_or_default())
}

fn resolve_frames(args: &H3EditArgs, video: &Path) -> Result<u32> {
    if args.frames > 0 {
        return Ok(snap_h3_frames(args.frames));
    }
    if args.duration.is_finite() && args.duration > 0.0 {
        let raw = (args.duration * f64::from(FPS)).round() as u32;
        return Ok(snap_h3_frames(raw.max(1)));
    }
    match probe_duration(video) {
        Ok(d) if d > 0.0 => {
            let raw = (d * f64::from(FPS)).round() as u32;
            Ok(snap_h3_frames(raw.max(1)))
        }
        Ok(_) | Err(_) => {
            bail!(
                "could not read duration from {} — pass --duration or --frames",
                video.display()
            )
        }
    }
}

fn probe_duration(path: &Path) -> Result<f64> {
    let ffmpeg = crate::host::paths::find_ffmpeg()?;
    let ffprobe = ffmpeg.with_file_name("ffprobe.exe");
    if !ffprobe.is_file() {
        bail!("ffprobe missing next to {}", ffmpeg.display());
    }
    let output = crate::host::workers::env::native_tool_command(&ffprobe)
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(path)
        .output()
        .with_context(|| format!("ffprobe {}", path.display()))?;
    if !output.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim()
        .parse::<f64>()
        .with_context(|| format!("parse duration {:?}", text.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn edit_plan_compiles_video_editing_ir() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("src.mp4");
        std::fs::write(&video, b"not-a-video").unwrap();
        let still = dir.path().join("sheet.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([20, 40, 80]))
            .save(&still)
            .unwrap();
        let args = H3EditArgs::try_parse_from([
            "gemmy-h3-edit",
            "--video",
            video.to_str().unwrap(),
            "--ref-image",
            still.to_str().unwrap(),
            "--mask-prompt",
            "head",
            "--prompt",
            "replace her head with the reference",
            "--duration",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.frames, 124);
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
        assert_eq!(plan.crop_mode, "combined");
        assert!(plan.verify_mask);
        assert!(!plan.stop_after_mask);
        assert!(plan.wants_sample);
        assert!(plan.prompt.contains("[video editing + reference generation]"));
        assert!(plan.prompt.contains("<Video 1>"));
        assert!(plan.prompt.contains("<Picture 1>"));
        assert!(plan.prompt.contains("head"));
        assert!(plan.prompt.contains("Do not keep the source hair"));
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(name.contains("10Eros_Max_h3_TURBO_ref2va_beta2"));
    }

    #[test]
    fn edit_requires_mask_or_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("src.mp4");
        std::fs::write(&video, b"x").unwrap();
        let still = dir.path().join("sheet.png");
        image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3]))
            .save(&still)
            .unwrap();
        let args = H3EditArgs::try_parse_from([
            "gemmy-h3-edit",
            "--video",
            video.to_str().unwrap(),
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "swap",
            "--duration",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("mask-prompt") || err.contains("--mask"));
    }

    #[test]
    fn stop_after_mask_does_not_need_ref_or_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("src.mp4");
        std::fs::write(&video, b"x").unwrap();
        let args = H3EditArgs::try_parse_from([
            "gemmy-h3-edit",
            "--video",
            video.to_str().unwrap(),
            "--mask-prompt",
            "female",
            "--stop-after-mask",
            "--duration",
            "2",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert!(plan.stop_after_mask);
        assert!(!plan.wants_sample);
        assert!(plan.verify_mask);
        assert!(plan.ref_images.is_empty());
        assert!(plan.prompt.is_empty());
    }

    #[test]
    fn no_verify_mask_disables_gate() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("src.mp4");
        std::fs::write(&video, b"x").unwrap();
        let still = dir.path().join("sheet.png");
        image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3]))
            .save(&still)
            .unwrap();
        let args = H3EditArgs::try_parse_from([
            "gemmy-h3-edit",
            "--video",
            video.to_str().unwrap(),
            "--ref-image",
            still.to_str().unwrap(),
            "--mask-prompt",
            "head",
            "--prompt",
            "swap",
            "--no-verify-mask",
            "--duration",
            "5",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert!(!plan.verify_mask);
        assert!(plan.wants_sample);
    }

    #[test]
    fn parse_mask_verdict_reads_pass_json() {
        let v = parse_mask_verdict(
            "sure\n```json\n{\"pass\": true, \"notes\": \"covers hair\", \"missing\": [], \"leaks\": []}\n```\n",
        );
        assert!(v.pass);
        assert!(v.notes.contains("covers hair"));
        let fail = parse_mask_verdict(
            "{\"pass\": false, \"notes\": \"missed bun volume\", \"missing\": [\"hair\"], \"leaks\": []}",
        );
        assert!(!fail.pass);
        assert_eq!(fail.missing, vec!["hair"]);
        let closed = parse_mask_verdict("the mask looks ok i guess");
        assert!(!closed.pass);
        assert!(closed.missing.contains(&"unparsed-verdict".into()));
    }
}
