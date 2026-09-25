//! `h3 loop` — multi-scene Context Loop production (CLI, not Comfy UI).

use super::args::H3LoopArgs;
use super::continue_cmd::concat_h3_segments;
use super::generate::{av_sidecar_paths, snap_h3_frames};
use super::models::validate_runtime_ready;
use super::paths::{
    abs, checkpoints_root, comfy_worker_script, default_dit_weights, h3_python, h3_root,
    read_sidecar_vae, resolve_video_vae,
};
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::paths::default_output_path;
use crate::host::util::{absolute_path, timestamp_slug};
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LoopPlanFile {
    #[serde(default)]
    schema: String,
    #[serde(default)]
    canvas: LoopCanvas,
    #[serde(default = "default_audio_mode")]
    audio_mode: String,
    /// Optional pipeline override (shortfilm sets `ref2va`). Empty = per-scene / inferred.
    #[serde(default)]
    mode: Option<String>,
    /// Optional DiT path (shortfilm sets Eros INT8). Empty = stock FL2VA default.
    #[serde(default)]
    weights: Option<PathBuf>,
    /// Optional cache (`off` / `spectrum` / `easy` / `fbc`). Empty = `off`.
    #[serde(default)]
    cache: Option<String>,
    /// Optional diffusion steps (shortfilm Eros writes 8). Empty = stock 20.
    #[serde(default)]
    steps: Option<u32>,
    /// Optional MiniMaxH3SigmaShift video (shortfilm writes 12). Empty = 12.
    #[serde(default)]
    shift_video: Option<f64>,
    /// Optional MiniMaxH3SigmaShift audio (shortfilm writes 3). Empty = 3.
    #[serde(default)]
    shift_audio: Option<f64>,
    /// Shortfilm already writes official six-section IR (`false`). Default
    /// `true` so existing loop plans still compile freeform prompts.
    #[serde(default = "default_compile_ir")]
    compile_ir: bool,
    /// Shortfilm Eros: 0.2 MP sample then 1.0 MP latent refine per scene.
    #[serde(default)]
    two_stage: bool,
    #[serde(default)]
    refine_canvas: Option<LoopCanvas>,
    scenes: Vec<LoopScene>,
}

fn default_compile_ir() -> bool {
    true
}

fn default_audio_mode() -> String {
    "generated_audio".into()
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct LoopCanvas {
    #[serde(default = "default_w")]
    width: u32,
    #[serde(default = "default_h")]
    height: u32,
}

fn default_w() -> u32 {
    864
}
fn default_h() -> u32 {
    480
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LoopScene {
    id: String,
    prompt: String,
    #[serde(default)]
    continuation_mode: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    frames: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    tags: BTreeMap<String, LoopTag>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LoopTag {
    kind: String,
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneStatus {
    index: u32,
    id: String,
    status: String,
    continuation_mode: String,
    seed: u64,
    segment: PathBuf,
    digest: Option<String>,
}

pub(crate) fn run_loop(args: H3LoopArgs, _config: &H3Config) -> Result<()> {
    let plan_path = abs(&args.plan)?;
    if !plan_path.is_file() && !args.dry_run_plan {
        bail!("--plan not found: {}", plan_path.display());
    }
    let raw = if plan_path.is_file() {
        std::fs::read_to_string(&plan_path)
            .with_context(|| format!("read {}", plan_path.display()))?
    } else {
        r#"{"schema":"gemmy-h3-loop-v1","scenes":[]}"#.into()
    };
    let plan: LoopPlanFile = serde_json::from_str(&raw)
        .with_context(|| format!("parse loop plan {}", plan_path.display()))?;
    if plan.scenes.is_empty() {
        bail!("loop plan has no scenes");
    }
    if !plan.schema.is_empty() && plan.schema != "gemmy-h3-loop-v1" {
        bail!("unsupported loop schema {:?} (expected gemmy-h3-loop-v1)", plan.schema);
    }

    let output = absolute_path(&args.output.clone().unwrap_or_else(|| {
        let stem = plan_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("loop");
        default_output_path(format!("{stem}.mp4"))
    }))?;
    let work_dir = output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(
            ".h3_loop_{}",
            plan_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&timestamp_slug())
        ));
    let (start, end) = scene_range(&args, plan.scenes.len())?;

    if args.dry_run_plan {
        print_dry_run(&plan, &output, &work_dir, start, end, &args)?;
        return Ok(());
    }

    validate_runtime_ready()?;
    std::fs::create_dir_all(&work_dir)?;
    std::fs::copy(&plan_path, work_dir.join("plan.json")).ok();

    let mut manifest = load_manifest(&work_dir, &plan);
    apply_control_flags(&args, &mut manifest, &plan)?;

    if args.stop {
        return assemble_accepted(&manifest, &work_dir, &output, args.json);
    }

    let python = h3_python();
    let worker = comfy_worker_script();
    if !python.is_file() {
        bail!("H3 python missing: {} — run: h3 install", python.display());
    }
    if !worker.is_file() {
        bail!("H3 worker missing: {}", worker.display());
    }

    let mut accepted: Vec<PathBuf> = Vec::new();
    for i in 0..plan.scenes.len() {
        let idx = (i as u32) + 1;
        let st = manifest
            .iter()
            .find(|s| s.index == idx)
            .map(|s| s.status.as_str())
            .unwrap_or("pending");
        if st == "accepted" {
            if let Some(s) = manifest.iter().find(|s| s.index == idx) {
                if s.segment.is_file() {
                    accepted.push(s.segment.clone());
                }
            }
        }
    }

    for i in (start - 1)..end {
        let idx = (i as u32) + 1;
        let scene = &plan.scenes[i];
        let scene_dir = work_dir.join("scenes").join(idx.to_string());
        std::fs::create_dir_all(&scene_dir)?;
        let current = manifest.iter().find(|s| s.index == idx);
        let status = current.map(|s| s.status.as_str()).unwrap_or("pending");

        if status == "accepted" && !args.retry && !args.reroll {
            if let Some(s) = current {
                if s.segment.is_file() {
                    continue;
                }
            }
        }
        if status == "reviewed" {
            match reviewed_scene_action(args.approve, args.retry, args.reroll) {
                ReviewedAction::Regenerate => {}
                ReviewedAction::Accept => {
                    let take = current.filter(|s| s.segment.is_file()).cloned();
                    if let Some(s) = take {
                        if let Some(slot) = manifest.iter_mut().find(|m| m.index == idx) {
                            slot.status = "accepted".into();
                        }
                        let accepted_rec = SceneStatus {
                            index: idx,
                            id: scene.id.clone(),
                            status: "accepted".into(),
                            continuation_mode: s.continuation_mode.clone(),
                            seed: s.seed,
                            segment: s.segment.clone(),
                            digest: s.digest.clone(),
                        };
                        write_scene_json(&scene_dir, &accepted_rec, scene)?;
                        accepted.push(s.segment);
                        write_manifest(&work_dir, &manifest)?;
                        println!(
                            "[h3 loop] scene {idx} ({}) approved without regenerating",
                            scene.id
                        );
                        continue;
                    }
                }
                ReviewedAction::Park => {
                    println!(
                        "[h3 loop] scene {idx} ({}) is reviewed. Pass --approve, --retry, --reroll, or --stop.",
                        scene.id
                    );
                    write_manifest(&work_dir, &manifest)?;
                    return Ok(());
                }
            }
        }

        let seed = if args.reroll {
            current.map(|s| s.seed.saturating_add(1)).unwrap_or(scene.seed.unwrap_or(42))
        } else {
            current.and_then(|s| if s.status == "reviewed" { None } else { Some(s.seed) })
                .unwrap_or(scene.seed.unwrap_or(42 + u64::from(idx)))
        };

        let continuation = resolve_continuation(scene, i, &accepted);
        let frames = generate_frames(scene_frames(scene), &continuation);
        let seg = scene_dir.join("segment.mp4");
        let prev_seg = accepted.last().cloned();
        let (prev_latent, prev_video) = match &prev_seg {
            Some(p) => {
                let (st, _) = av_sidecar_paths(p);
                (
                    if st.is_file() { Some(st) } else { None },
                    Some(p.clone()),
                )
            }
            None => (None, None),
        };

        let refs = resolve_tagged_refs(scene)?;
        let guides = resolve_guide_images(scene, &continuation)?;
        // ganloss JSON sampler on 16 GB: SplitSigmas@4 in process 1 (persist
        // latent), 3D SR + 3-step in a fresh process. One-graph refine dumps
        // the DiT to 0 MB VRAM (~24 min/step).
        let sampler_continuation = if plan.two_stage && continuation == "none" {
            "ganloss_two_stage".to_string()
        } else {
            continuation.clone()
        };
        let gen_out = if plan.two_stage {
            scene_dir.join("segment_stage1.mp4")
        } else {
            seg.clone()
        };
        run_scene_generate(
            args.verbose,
            &python,
            &worker,
            &scene_dir,
            &gen_out,
            scene,
            frames,
            plan.canvas.width,
            plan.canvas.height,
            seed,
            &sampler_continuation,
            prev_latent.as_deref(),
            prev_video.as_deref(),
            &plan.audio_mode,
            &refs,
            &guides,
            plan.mode.as_deref(),
            plan.weights.as_deref(),
            plan.cache.as_deref(),
            plan.steps,
            plan.shift_video,
            plan.shift_audio,
            plan.compile_ir,
            &args.vae_select,
        )?;
        if plan.two_stage {
            let img = scene.tags.values().find(|t| t.kind == "image").map(|t| t.path.clone());
            let weights = plan.weights.clone().unwrap_or_else(super::paths::default_dit_weights);
            super::upscale::run_eros_stage2(
                &gen_out,
                &seg,
                &scene.prompt,
                &weights,
                img.as_deref(),
                &[],
                seed,
                args.verbose,
                _config,
                608,
                352,
                false,
                0.75,
                super::paths::VSA_GATE_DEFAULT,
                &[],
                &args.vae_select,
            )?;
        }

        let digest = av_sidecar_paths(&seg)
            .1
            .exists()
            .then(|| load_digest(&av_sidecar_paths(&seg).1))
            .flatten();
        let status_rec = SceneStatus {
            index: idx,
            id: scene.id.clone(),
            status: if args.review { "reviewed".into() } else { "accepted".into() },
            continuation_mode: continuation.clone(),
            seed,
            segment: seg.clone(),
            digest,
        };
        write_scene_json(&scene_dir, &status_rec, scene)?;
        upsert_status(&mut manifest, status_rec);

        if args.review {
            run_review(&seg, scene, &scene_dir, args.verbose)?;
            write_manifest(&work_dir, &manifest)?;
            if !args.approve {
                println!(
                    "[h3 loop] scene {idx} ({}) reviewed. Next: --approve | --retry | --reroll | --stop",
                    scene.id
                );
                return Ok(());
            }
            if let Some(s) = manifest.iter_mut().find(|s| s.index == idx) {
                s.status = "accepted".into();
            }
        }

        accepted.push(seg);
        write_manifest(&work_dir, &manifest)?;
        if args.stop {
            break;
        }
    }

    assemble_accepted(&manifest, &work_dir, &output, args.json)
}

fn scene_range(args: &H3LoopArgs, n: usize) -> Result<(usize, usize)> {
    if let Some(raw) = &args.scene_range {
        let Some((a, b)) = raw.split_once(':') else {
            bail!("--scene-range expects A:B (1-based inclusive)");
        };
        let start: usize = a.parse().context("--scene-range start")?;
        let end: usize = b.parse().context("--scene-range end")?;
        if start == 0 || end == 0 || start > n || end > n || start > end {
            bail!("--scene-range {raw} is outside 1..={n}");
        }
        return Ok((start, end));
    }
    let start = args.start_scene.unwrap_or(1) as usize;
    if start == 0 || start > n {
        bail!("--start-scene {start} is outside 1..={n}");
    }
    Ok((start, n))
}

const DEFAULT_CONTEXT_FRAMES: u32 = 39;

fn scene_frames(scene: &LoopScene) -> u32 {
    if let Some(f) = scene.frames {
        return snap_h3_frames(f);
    }
    let secs = scene.duration.unwrap_or(5.0).max(0.5);
    snap_h3_frames((secs * 24.0).round() as u32)
}

/// Continued scenes sample context+scene, then trim the protected prefix.
/// A 39-frame `masked_av` scene must not generate only 39 frames (trim would be empty).
fn generate_frames(scene_len: u32, continuation: &str) -> u32 {
    if continuation == "masked_av" || continuation == "vae_tail" {
        snap_h3_frames(scene_len.saturating_add(DEFAULT_CONTEXT_FRAMES))
    } else {
        scene_len
    }
}

fn resolve_continuation(scene: &LoopScene, index: usize, accepted: &[PathBuf]) -> String {
    if let Some(mode) = scene.continuation_mode.as_deref() {
        return mode.to_string();
    }
    if index == 0 || accepted.is_empty() {
        return "none".into();
    }
    let prev = &accepted[accepted.len() - 1];
    let (st, _) = av_sidecar_paths(prev);
    if st.is_file() {
        "masked_av".into()
    } else {
        "vae_tail".into()
    }
}

fn resolve_guide_images(scene: &LoopScene, continuation: &str) -> Result<Vec<Value>> {
    if continuation != "guide" {
        return Ok(Vec::new());
    }
    let mut guides = Vec::new();
    for (tag, spec) in &scene.tags {
        if spec.kind != "image" {
            continue;
        }
        let path = abs(&spec.path)?;
        if !path.is_file() {
            bail!("@{} guide image not found: {}", tag, path.display());
        }
        guides.push(json!({ "path": path, "frame_idx": 0 }));
    }
    Ok(guides)
}

fn resolve_tagged_refs(scene: &LoopScene) -> Result<Vec<Value>> {
    let mut refs = Vec::new();
    for (tag, spec) in &scene.tags {
        let path = abs(&spec.path)?;
        if !path.is_file() {
            bail!("@{tag} path not found: {}", path.display());
        }
        refs.push(json!({
            "kind": spec.kind,
            "path": path,
            "tag": tag,
        }));
    }
    Ok(refs)
}

fn run_scene_generate(
    verbose: bool,
    python: &Path,
    worker: &Path,
    scene_dir: &Path,
    seg: &Path,
    scene: &LoopScene,
    frames: u32,
    width: u32,
    height: u32,
    seed: u64,
    continuation: &str,
    prev_latent: Option<&Path>,
    prev_video: Option<&Path>,
    audio_mode: &str,
    refs: &[Value],
    guides: &[Value],
    plan_mode: Option<&str>,
    plan_weights: Option<&Path>,
    plan_cache: Option<&str>,
    plan_steps: Option<u32>,
    plan_shift_video: Option<f64>,
    plan_shift_audio: Option<f64>,
    compile_ir: bool,
    vae_select: &super::args::H3VaeSelect,
) -> Result<()> {
    let work = scene_dir.join("work");
    std::fs::create_dir_all(&work)?;
    let mode = scene
        .mode
        .clone()
        .or_else(|| plan_mode.map(str::to_string))
        .unwrap_or_else(|| if refs.iter().any(|r| r["kind"] != "image") { "ref2va".into() } else if !refs.is_empty() { "ref2va".into() } else { "t2va".into() });
    let weights = plan_weights
        .map(Path::to_path_buf)
        .unwrap_or_else(default_dit_weights);
    let cache = plan_cache.unwrap_or("off");
    let steps = plan_steps.unwrap_or(20);
    let shift_video = plan_shift_video.unwrap_or(12.0);
    let shift_audio = plan_shift_audio.unwrap_or(3.0);
    let sidecar = prev_video.and_then(read_sidecar_vae);
    let video_vae = resolve_video_vae(
        vae_select,
        sidecar.as_deref(),
        false,
        true,
    )?;
    let request = json!({
        "prompt": scene.prompt,
        "output": seg,
        "work_dir": work,
        "h3_root": h3_root(),
        "checkpoints_root": checkpoints_root(),
        "python": python,
        "mode": mode,
        "quality": "fast",
        "attn": "sage",
        "engine": "comfy",
        "turbo": "off",
        "realism": false,
        "sol": false,
        "cache": cache,
        "dit_quant": "int8",
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "seed": seed,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "compile_ir": compile_ir,
        "refs": refs,
        "weights": weights,
        "video_vae": video_vae.filename,
        "persist_av_latent": true,
        "continuation_mode": continuation,
        "context_latent": prev_latent.filter(|_| continuation == "masked_av" || continuation == "native_guide"),
        "context_video": prev_video.filter(|_| continuation == "vae_tail"),
        "context_frames": DEFAULT_CONTEXT_FRAMES,
        "trim_prefix": continuation == "masked_av" || continuation == "vae_tail",
        "audio_mode": audio_mode,
        "guide_images": guides,
    });
    let request_path = scene_dir.join("request.json");
    std::fs::write(&request_path, serde_json::to_vec_pretty(&request)?)?;

    println!(
        "[h3 loop] scene {} mode={} continuation={} frames={} seed={}",
        scene.id, mode, continuation, frames, seed
    );
    println!("[h3 loop] unloading Gemmy before H3 scene...");

    let python = python.to_path_buf();
    let worker = worker.to_path_buf();
    let req = request_path.clone();
    let ckpt = checkpoints_root();
    let out = seg.to_path_buf();
    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", ckpt.as_os_str());
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        cmd.arg(&worker).arg("--request").arg(&req);
        WorkerSpec::new(cmd, "video.h3.loop")
            .verbose(verbose)
            .expect_outputs(vec![OutputExpectation::video(out.clone()).labeled("H3 loop scene")])
            .run_inherited()
            .map(|_| ())
    })
}

fn run_review(seg: &Path, scene: &LoopScene, scene_dir: &Path, verbose: bool) -> Result<()> {
    let gemmy = crate::host::gemmy::require_gemmy(
        "loop --review is `gemmy analyze`. Omit --review to assemble without it",
    )?;
    let review_txt = scene_dir.join("review.txt");
    let prompt = format!(
        "Review this MiniMax-H3 scene for production. Planned prompt:\n{}\n\nReply with JSON only: {{\"accept\": true|false, \"notes\": \"...\"}}",
        scene.prompt
    );
    let mut cmd = crate::host::workers::env::gemmy_child_command(&gemmy);
    cmd.env("GEMMY_QUEUE_ACTIVE", "1");
    cmd.arg("analyze")
        .arg("--file")
        .arg(seg)
        .arg("--video-fps")
        .arg("1")
        .arg("--video-max-frames")
        .arg("60")
        .arg("--output")
        .arg(&review_txt)
        .arg(&prompt);
    println!("[h3 loop] review via gemmy analyze (12B) → {}", review_txt.display());
    let status = cmd
        .status()
        .with_context(|| "run gemmy analyze")?;
    if !status.success() {
        bail!("gemmy analyze review failed for {}", seg.display());
    }
    let notes = std::fs::read_to_string(&review_txt).unwrap_or_default();
    let review = json!({
        "scene": scene.id,
        "segment": seg,
        "notes": notes,
        "verbose": verbose,
    });
    std::fs::write(scene_dir.join("review.json"), serde_json::to_vec_pretty(&review)?)?;
    Ok(())
}

fn assemble_accepted(
    manifest: &[SceneStatus],
    work_dir: &Path,
    output: &Path,
    json_out: bool,
) -> Result<()> {
    let segs: Vec<PathBuf> = manifest
        .iter()
        .filter(|s| s.status == "accepted" && s.segment.is_file())
        .map(|s| s.segment.clone())
        .collect();
    if segs.is_empty() {
        bail!("no accepted scenes to assemble");
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    concat_h3_segments(&segs, output)?;
    write_manifest(work_dir, manifest)?;
    println!("wrote H3 loop video: {}", output.display());
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "loop",
                "output": output,
                "accepted": segs,
                "manifest": manifest,
            }))?
        );
    }
    Ok(())
}

fn print_dry_run(
    plan: &LoopPlanFile,
    output: &Path,
    work_dir: &Path,
    start: usize,
    end: usize,
    args: &H3LoopArgs,
) -> Result<()> {
    if args.json {
        let scenes: Vec<Value> = plan
            .scenes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let idx = i + 1;
                let scene_dir = work_dir.join("scenes").join(idx.to_string());
                let digest = load_digest(&av_sidecar_paths(&scene_dir.join("segment.mp4")).1);
                json!({
                    "index": idx,
                    "id": s.id,
                    "continuation_mode": s.continuation_mode.clone().unwrap_or_else(|| if i == 0 { "none".into() } else { "masked_av".into() }),
                    "mode": s.mode,
                    "frames": scene_frames(s),
                    "generate_frames": generate_frames(
                        scene_frames(s),
                        &s.continuation_mode.clone().unwrap_or_else(|| {
                            if i == 0 {
                                "none".into()
                            } else {
                                "masked_av".into()
                            }
                        }),
                    ),
                    "resume_hash": digest,
                    "in_range": idx >= start && idx <= end,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "command": "loop",
                "audio_mode": plan.audio_mode,
                "mode": plan.mode,
                "cache": plan.cache,
                "steps": plan.steps,
                "shift_video": plan.shift_video,
                "shift_audio": plan.shift_audio,
                "compile_ir": plan.compile_ir,
                "two_stage": plan.two_stage,
                "refine_canvas": plan.refine_canvas,
                "canvas": plan.canvas,
                "output": output,
                "work_dir": work_dir,
                "range": [start, end],
                "scenes": scenes,
            }))?
        );
        return Ok(());
    }
    println!("[h3 loop] dry-run plan");
    println!(
        "[h3 loop] canvas={}x{} audio_mode={} scenes={} range={}:{} mode={:?} cache={:?} steps={:?} shift={:?}/{:?} compile_ir={} two_stage={}",
        plan.canvas.width,
        plan.canvas.height,
        plan.audio_mode,
        plan.scenes.len(),
        start,
        end,
        plan.mode,
        plan.cache,
        plan.steps,
        plan.shift_video,
        plan.shift_audio,
        plan.compile_ir,
        plan.two_stage
    );
    let vae = resolve_video_vae(
        &args.vae_select,
        None,
        false,
        args.vae_select.video_vae.is_some(),
    )?;
    println!("[h3 loop] video_vae={}", vae.filename);
    println!("[h3 loop] output={}", output.display());
    println!("[h3 loop] work_dir={}", work_dir.display());
    for (i, s) in plan.scenes.iter().enumerate() {
        let idx = i + 1;
        let scene_dir = work_dir.join("scenes").join(idx.to_string());
        let digest = load_digest(&av_sidecar_paths(&scene_dir.join("segment.mp4")).1)
            .unwrap_or_else(|| "none".into());
        let mode = s
            .continuation_mode
            .clone()
            .unwrap_or_else(|| if i == 0 { "none".into() } else { "masked_av".into() });
        println!(
            "[h3 loop] scene {idx} id={} continuation={} frames={} generate={} resume_hash={}",
            s.id,
            mode,
            scene_frames(s),
            generate_frames(scene_frames(s), &mode),
            digest
        );
    }
    Ok(())
}

fn load_manifest(work_dir: &Path, plan: &LoopPlanFile) -> Vec<SceneStatus> {
    let path = work_dir.join("manifest.json");
    if let Ok(raw) = std::fs::read_to_string(&path) {
        if let Ok(v) = serde_json::from_str::<Vec<SceneStatus>>(&raw) {
            return v;
        }
    }
    plan.scenes
        .iter()
        .enumerate()
        .map(|(i, s)| SceneStatus {
            index: (i as u32) + 1,
            id: s.id.clone(),
            status: "pending".into(),
            continuation_mode: s.continuation_mode.clone().unwrap_or_default(),
            seed: s.seed.unwrap_or(42 + i as u64),
            segment: work_dir
                .join("scenes")
                .join((i + 1).to_string())
                .join("segment.mp4"),
            digest: None,
        })
        .collect()
}

fn write_manifest(work_dir: &Path, manifest: &[SceneStatus]) -> Result<()> {
    std::fs::create_dir_all(work_dir)?;
    std::fs::write(
        work_dir.join("manifest.json"),
        serde_json::to_vec_pretty(manifest)?,
    )?;
    Ok(())
}

fn write_scene_json(scene_dir: &Path, status: &SceneStatus, scene: &LoopScene) -> Result<()> {
    std::fs::write(
        scene_dir.join("scene.json"),
        serde_json::to_vec_pretty(&json!({
            "id": scene.id,
            "prompt": scene.prompt,
            "status": status,
        }))?,
    )?;
    Ok(())
}

fn upsert_status(manifest: &mut Vec<SceneStatus>, next: SceneStatus) {
    if let Some(slot) = manifest.iter_mut().find(|s| s.index == next.index) {
        *slot = next;
    } else {
        manifest.push(next);
    }
}

fn apply_control_flags(
    args: &H3LoopArgs,
    manifest: &mut [SceneStatus],
    _plan: &LoopPlanFile,
) -> Result<()> {
    // --approve is handled in the scene loop so a reviewed take is accepted
    // without a second DiT pass. Only --retry / --reroll reset to pending.
    if args.retry || args.reroll {
        if let Some(s) = manifest
            .iter_mut()
            .rev()
            .find(|s| s.status == "reviewed" || s.status == "pending")
        {
            s.status = "pending".into();
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewedAction {
    Park,
    Accept,
    Regenerate,
}

fn reviewed_scene_action(approve: bool, retry: bool, reroll: bool) -> ReviewedAction {
    if retry || reroll {
        ReviewedAction::Regenerate
    } else if approve {
        ReviewedAction::Accept
    } else {
        ReviewedAction::Park
    }
}

fn load_digest(meta: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(meta).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    v.get("digest")?.as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::{generate_frames, reviewed_scene_action, ReviewedAction};

    #[test]
    fn approve_accepts_reviewed_without_regen() {
        assert_eq!(
            reviewed_scene_action(true, false, false),
            ReviewedAction::Accept
        );
    }

    #[test]
    fn park_reviewed_without_decision() {
        assert_eq!(
            reviewed_scene_action(false, false, false),
            ReviewedAction::Park
        );
    }

    #[test]
    fn retry_and_reroll_regenerate() {
        assert_eq!(
            reviewed_scene_action(false, true, false),
            ReviewedAction::Regenerate
        );
        assert_eq!(
            reviewed_scene_action(false, false, true),
            ReviewedAction::Regenerate
        );
        assert_eq!(
            reviewed_scene_action(true, true, false),
            ReviewedAction::Regenerate
        );
    }

    #[test]
    fn continued_scene_generates_past_the_protected_prefix() {
        assert_eq!(generate_frames(39, "none"), 39);
        assert_eq!(generate_frames(39, "native_guide"), 39);
        assert_eq!(generate_frames(39, "masked_av"), 90);
        assert_eq!(generate_frames(39, "vae_tail"), 90);
    }

    #[test]
    fn loop_plan_keeps_scheduled_tags_not_picture_slots() {
        let raw = r#"{
            "schema": "gemmy-h3-loop-v1",
            "mode": "ref2va",
            "cache": "off",
            "weights": "F:\\Models\\minimax-h3-eros\\diffusion_models\\10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors",
            "scenes": [{
                "id": "open",
                "prompt": "subject_definitions:\n<Subject 1> is the identity-locked character from @hero_face.\n@storyboard is the labeled storyboard Picture.\n",
                "mode": "ref2va",
                "tags": {
                    "hero_face": {"kind": "image", "path": "sheet.jpg"},
                    "storyboard": {"kind": "image", "path": "board.png"}
                }
            }]
        }"#;
        let plan: super::LoopPlanFile = serde_json::from_str(raw).unwrap();
        assert_eq!(plan.mode.as_deref(), Some("ref2va"));
        assert_eq!(plan.cache.as_deref(), Some("off"));
        assert!(plan
            .weights
            .as_ref()
            .unwrap()
            .to_string_lossy()
            .contains("10Eros_Max_h3_TURBO_ref2va_beta2"));
        let scene = &plan.scenes[0];
        assert!(scene.tags.contains_key("hero_face"));
        assert!(scene.tags.contains_key("storyboard"));
        assert!(!scene.prompt.contains("<Picture 1>"));
        assert!(scene.prompt.contains("@hero_face"));
        assert!(scene.prompt.contains("@storyboard"));
        assert!(plan.compile_ir);
    }

    #[test]
    fn loop_plan_honors_explicit_compile_ir_false() {
        let raw = r#"{
            "schema": "gemmy-h3-loop-v1",
            "compile_ir": false,
            "scenes": [{"id": "s1", "prompt": "subject_definitions:\n"}]
        }"#;
        let plan: super::LoopPlanFile = serde_json::from_str(raw).unwrap();
        assert!(!plan.compile_ir);
    }
}
