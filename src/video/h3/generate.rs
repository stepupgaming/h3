//! `h3 generate` — plan → optional Krea still → GPU handoff → worker → ffprobe.

use super::args::{H3Cache, H3Dit, H3DitQuant, H3Engine, H3GenerateArgs, H3Mode, H3Turbo};
use super::canvas::{resolve_sample_canvas, H3CanvasPreset};
use super::models::validate_runtime_ready;
use super::paths::{
    abs, checkpoints_root, comfy_root, comfy_runner_script, comfy_worker_script,
    default_dit_weights_for_mode, h3_python, h3_root, jev_native_sla_dir, jev_sdk_python,
    realism_people_lora, resolve_video_vae, singularity_ref2va_int8_path,
    stock_ref2va_backup_path, turbo_lora_v4, typesafe_api_key_present, vsa_gate_path,
    worker_script, EROS_REF2VA_STEPS, JEV_INITIAL_POLICY, JEV_STEPS, SINGULARITY_REF2VA_INT8,
    VIDEO_VAE_FP16_FILE,
};
#[cfg(test)]
use super::paths::VIDEO_VAE_INT8_FILE;
use super::scene_ir::compile_scene_ir;
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::paths::default_output_path;
use crate::host::util::{absolute_path, run_simple_command, timestamp_slug};
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// H3 frame grid: frame_count ≡ 5 (mod 17), min 5, 24 fps.
const FRAME_MOD: u32 = 17;
const FRAME_REM: u32 = 5;
const FPS: u32 = 24;

/// Official Ref2VA caps (package `conditioning.py` / sample CLI).
const MAX_REF_IMAGES: usize = 9;
const MAX_REF_VIDEOS: usize = 3;
const MAX_REF_AUDIOS: usize = 3;
const MAX_REF_FILES: usize = 12;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct H3RefModUse {
    pub(crate) name: String,
    pub(crate) strength: f64,
    pub(crate) copies: u32,
}

impl H3RefModUse {
    pub(crate) fn spec(&self) -> String {
        format!("{}:{}:{}", self.name, self.strength, self.copies)
    }
}

#[derive(Debug, Clone, Serialize)]
struct H3RefSpec {
    /// `image` | `video` | `audio` | `av`
    kind: String,
    /// Single media path (image/video/audio).
    path: Option<PathBuf>,
    /// Paired paths for `av` (video, audio).
    video: Option<PathBuf>,
    audio: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct H3Plan {
    pub(crate) mode: String,
    quality: String,
    attn: String,
    engine: String,
    turbo: String,
    turbo_strength: f64,
    realism: bool,
    realism_strength: f64,
    pub(crate) sol: bool,
    pub(crate) vsa: bool,
    pub(crate) vsa_sparsity: f64,
    pub(crate) vsa_gate: String,
    pub(crate) jev: bool,
    pub(crate) jev_sdk_python: String,
    pub(crate) jev_initial_policy: String,
    pub(crate) sla_fixed: Option<u8>,
    pub(crate) sla_table: Option<PathBuf>,
    pub(crate) jev_log_dataset: Option<PathBuf>,
    pub(crate) no_sla: bool,
    pub(crate) cache: String,
    dit_quant: String,
    prompt: String,
    still_prompt: Option<String>,
    pub(crate) compile_ir: bool,
    canvas: Option<String>,
    width: u32,
    height: u32,
    megapixels: f64,
    frames: u32,
    pub(crate) duration_s: f64,
    pub(crate) steps: u32,
    /// `res_multistep` for 4-step --jev/--no-sla; `euler` for stock HQ including N-step native SLA.
    pub(crate) sampler: String,
    seed: u64,
    pub(crate) shift_video: f64,
    pub(crate) shift_audio: f64,
    pub(crate) first_frame: Option<PathBuf>,
    last_frame: Option<PathBuf>,
    /// Ordered Ref2VA media specs (product paths; worker encodes to latents).
    refs: Vec<H3RefSpec>,
    /// Saved RefMods (`name`, strength, copies) injected via MiniMaxH3Mod.
    refmods: Vec<H3RefModUse>,
    /// Experimental Multishot-style DiT memory: keyframes + ref-images together.
    allow_keyframe_refs: bool,
    /// When true, Krea 2 will generate `first_frame` before H3 sample.
    auto_still: bool,
    auto_still_engine: Option<String>,
    pub(crate) weights: PathBuf,
    output: PathBuf,
    h3_root: PathBuf,
    checkpoints_root: PathBuf,
    comfy_root: Option<PathBuf>,
    python: PathBuf,
    worker: PathBuf,
    profile: bool,
    keep_work: bool,
    persist_av_latent: bool,
    av_latent: PathBuf,
    continuation_mode: String,
    two_stage: bool,
    pub(crate) video_vae: String,
}

fn validate_sla_table_file(path: &Path) -> Result<usize> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read --sla-table {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parse --sla-table JSON {}", path.display()))?;
    let table = v.get("keep_table").unwrap_or(&v);
    let rows = table.as_array().ok_or_else(|| {
        anyhow::anyhow!("--sla-table must be an N×50 JSON array or {{keep_table: ...}}")
    })?;
    if rows.is_empty() || rows.len() > 100 {
        bail!("--sla-table must have 1..=100 steps (got {})", rows.len());
    }
    for (si, row) in rows.iter().enumerate() {
        let cols = row.as_array().ok_or_else(|| {
            anyhow::anyhow!("--sla-table step {} must be an array of 50 keeps", si + 1)
        })?;
        if cols.len() != 50 {
            bail!(
                "--sla-table step {} must have 50 layers (got {})",
                si + 1,
                cols.len()
            );
        }
        for (li, cell) in cols.iter().enumerate() {
            let n = cell.as_u64().or_else(|| cell.as_i64().map(|x| x as u64));
            match n {
                Some(1 | 3 | 5 | 10) => {}
                Some(other) => bail!(
                    "--sla-table step {} layer {} keep {other} is not 1, 3, 5, or 10",
                    si + 1,
                    li
                ),
                None => bail!(
                    "--sla-table step {} layer {} is not a keep percent",
                    si + 1,
                    li
                ),
            }
        }
    }
    Ok(rows.len())
}

pub(crate) fn run_generate(args: H3GenerateArgs, config: &H3Config) -> Result<()> {
    let mut plan = build_plan(&args)?;

    if args.dry_run_plan {
        let v = serde_json::to_value(&plan)?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            println!("[h3 generate] dry-run plan");
            println!(
                "[h3 generate] mode={} engine={} quality={} attn={} turbo={} realism={} sol={} vsa={} jev={} cache={} {}x{} (~{:.2} MP{}) frames={} (~{:.2}s) steps={} sampler={} shift={}/{} seed={} two_stage={}",
                plan.mode,
                plan.engine,
                plan.quality,
                plan.attn,
                plan.turbo,
                plan.realism,
                plan.sol,
                plan.vsa,
                plan.jev,
                plan.cache,
                plan.width,
                plan.height,
                plan.megapixels,
                plan.canvas
                    .as_ref()
                    .map(|c| format!(", {c}"))
                    .unwrap_or_default(),
                plan.frames,
                plan.duration_s,
                plan.steps,
                plan.sampler,
                plan.shift_video,
                plan.shift_audio,
                plan.seed,
                plan.two_stage
            );
            if plan.two_stage {
                let (fw, fh) = H3CanvasPreset::Mp10.size();
                let (rw, rh) = if plan.height > plan.width {
                    (fh, fw)
                } else {
                    (fw, fh)
                };
                println!("[h3 generate] Eros two-stage refine {rw}x{rh} after SplitSigmas@4");
            }
            println!("[h3 generate] weights={}", plan.weights.display());
            if plan
                .weights
                .file_name()
                .and_then(|name| name.to_str())
                == Some(SINGULARITY_REF2VA_INT8)
            {
                println!(
                    "[h3 generate] dit=singularity dual-sample (scheduler 6, extend 2, split 2, 12 denoise steps, full INT8)"
                );
            }
            if plan.no_sla {
                println!(
                    "[h3 generate] no-sla 4-step res_multistep (no H3JevNativeSLAPatch)"
                );
            }
            if plan.jev {
                println!(
                    "[h3 generate] 009jev native SLA initial_policy={} sampler={} sdk_python={}",
                    plan.jev_initial_policy, plan.sampler, plan.jev_sdk_python
                );
            }
            if let Some(keep) = plan.sla_fixed {
                println!(
                    "[h3 generate] sla-fixed {keep}% const keep (no Jev worker)"
                );
            }
            if let Some(table) = &plan.sla_table {
                println!(
                    "[h3 generate] sla-table {} per-(step,layer) keep (no Jev worker)",
                    table.display()
                );
            }
            println!("[h3 generate] video_vae={}", plan.video_vae);
            println!("[h3 generate] output={}", plan.output.display());
            if plan.persist_av_latent {
                println!("[h3 generate] av_latent={}", plan.av_latent.display());
            }
            println!("[h3 generate] h3_root={}", plan.h3_root.display());
            println!(
                "[h3 generate] checkpoints_root={}",
                plan.checkpoints_root.display()
            );
            if let Some(c) = &plan.comfy_root {
                println!("[h3 generate] comfy_root={}", c.display());
            }
            println!("[h3 generate] worker={}", plan.worker.display());
            if plan.auto_still {
                println!(
                    "[h3 generate] first_frame=auto via {} (still_prompt={})",
                    plan.auto_still_engine.as_deref().unwrap_or("krea2"),
                    plan.still_prompt.as_deref().unwrap_or(&plan.prompt)
                );
            } else if let Some(p) = &plan.first_frame {
                println!("[h3 generate] first_frame={}", p.display());
            }
            if let Some(p) = &plan.last_frame {
                println!("[h3 generate] last_frame={}", p.display());
            }
            for (i, r) in plan.refs.iter().enumerate() {
                match r.kind.as_str() {
                    "av" => println!(
                        "[h3 generate] ref[{i}]=av video={} audio={}",
                        r.video
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_default(),
                        r.audio
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_default()
                    ),
                    _ => println!(
                        "[h3 generate] ref[{i}]={} {}",
                        r.kind,
                        r.path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_default()
                    ),
                }
            }
            for (i, m) in plan.refmods.iter().enumerate() {
                println!(
                    "[h3 generate] refmod[{i}]={} strength={} copies={}",
                    m.name, m.strength, m.copies
                );
            }
        }
        return Ok(());
    }

    validate_runtime_ready()?;
    if plan.mode == "ref2va" {
        if !plan.weights.is_file() {
            let name = plan
                .weights
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if name == SINGULARITY_REF2VA_INT8 {
                bail!(
                    "Singularity DiT missing: {}\n\
                     Opt-in only (--dit singularity). File {} from \
                     Hugging Face WarmBloodAban/Minimax-h3_Singularity, under \
                     F:\\Models\\minimax-h3-singularity\\diffusion_models\\.\n\
                     This does not replace Eros.",
                    plan.weights.display(),
                    SINGULARITY_REF2VA_INT8
                );
            }
            bail!(
                "Ref2VA DiT weights missing: {}\n\
                 Product default is Eros INT8 at F:\\Models\\minimax-h3-eros \
                 (h3 shortfilm --phase download). Stock Comfy-Org Ref2VA is a backup \
                 on G:\\Models\\minimax-h3-backup — pass --weights PATH.",
                plan.weights.display()
            );
        }
    } else if !plan.weights.is_file() {
        bail!("DiT weights missing: {}", plan.weights.display());
    }
    if !plan.python.is_file() {
        bail!(
            "H3 python missing: {} — run: h3 install",
            plan.python.display()
        );
    }
    if !plan.worker.is_file() {
        bail!("H3 worker missing: {}", plan.worker.display());
    }
    if plan.engine == "comfy" {
        let runner = comfy_runner_script();
        if !runner.is_file() {
            bail!(
                "Comfy runner missing: {}\n\
                 Expected internalized pack at runtimes\\minimax-h3\\ComfyUI\\run_h3_workflow.py \
                 (dev override only: GEMMY_H3_COMFY).",
                runner.display()
            );
        }
        if plan.turbo != "off" && !turbo_lora_v4().is_file() && plan.turbo == "v4" {
            bail!(
                "Turbo v4 LoRA missing: {}\n\
                 Download larryvrh MiniMax-H3-Turbo-Lora v4 ema into checkpoints/loras/ \
                 (or use --turbo ema_wan if the WanGP lab file is hardlinked).",
                turbo_lora_v4().display()
            );
        }
        if plan.realism && !realism_people_lora().is_file() {
            bail!(
                "Realism People LoRA missing: {}\n\
                 Place fal MiniMax-H3-Realism-People file \
                 h3-realism-people-t2v-i2v-r2v.safetensors under checkpoints/loras/.",
                realism_people_lora().display()
            );
        }
        if matches!(plan.turbo.as_str(), "v4" | "v1" | "ema_wan")
            && matches!(plan.cache.as_str(), "easy" | "fbc")
            && plan.cache != "off"
        {
            // spectrum is allowed with turbo per xmarre; easy/fbc must not stack with spectrum only —
            // single cache value already enforces mutex among caches.
        }
    }

    if plan.output.exists() {
        // Allow overwrite of prior gemmy outputs; still warn.
        eprintln!(
            "[h3 generate] overwriting existing output {}",
            plan.output.display()
        );
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }

    let run_id = format!("h3-{}", timestamp_slug());
    let work_dir = plan
        .output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!(".h3_work_{run_id}"));
    std::fs::create_dir_all(&work_dir)?;

    // Default product path: Krea 2 hero still → H3 I2V. Krea does its own GPU
    // handoff and exits before H3 loads, keeping 16 GB sequencing clean.
    if plan.auto_still {
        let still_path = work_dir.join("first_frame_krea2.png");
        let still_prompt = plan
            .still_prompt
            .clone()
            .unwrap_or_else(|| plan.prompt.clone());
        generate_krea_still(
            &still_prompt,
            &still_path,
            plan.width,
            plan.height,
            plan.seed,
            args.verbose,
        )?;
        plan.first_frame = Some(still_path);
        plan.auto_still = false;
    }

    if plan.mode == "i2v" && plan.first_frame.is_none() {
        bail!("i2v mode ended without a first-frame still");
    }

    let refs_json: Vec<_> = plan
        .refs
        .iter()
        .map(|r| match r.kind.as_str() {
            "av" => json!({
                "kind": "av",
                "video": r.video,
                "audio": r.audio,
            }),
            _ => json!({
                "kind": r.kind,
                "path": r.path,
            }),
        })
        .collect();

    let worker_out = if plan.two_stage {
        let mut p = plan.output.clone();
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "h3".into());
        p.set_file_name(format!("{stem}_stage1.mp4"));
        p
    } else {
        plan.output.clone()
    };

    let mut request = json!({
        "prompt": plan.prompt,
        "output": worker_out,
        "work_dir": work_dir,
        "h3_root": plan.h3_root,
        "checkpoints_root": plan.checkpoints_root,
        "comfy_root": plan.comfy_root,
        "python": plan.python,
        "mode": plan.mode,
        "quality": plan.quality,
        "attn": plan.attn,
        "engine": plan.engine,
        "turbo": plan.turbo,
        "turbo_strength": plan.turbo_strength,
        "realism": plan.realism,
        "realism_strength": plan.realism_strength,
        "sol": plan.sol,
        "vsa": plan.vsa,
        "vsa_sparsity": plan.vsa_sparsity,
        "vsa_gate": plan.vsa_gate,
        "cache": plan.cache,
        "dit_quant": plan.dit_quant,
        "width": plan.width,
        "height": plan.height,
        "frames": plan.frames,
        "steps": plan.steps,
        "sampler": plan.sampler,
        "model_family": plan.mode,
        "seed": plan.seed,
        "shift_video": plan.shift_video,
        "shift_audio": plan.shift_audio,
        "compile_ir": plan.compile_ir,
        "first_image": plan.first_frame,
        "last_image": plan.last_frame,
        "refs": refs_json,
        "refmods": plan.refmods,
        "allow_keyframe_refs": plan.allow_keyframe_refs,
        "weights": plan.weights,
        "video_vae": plan.video_vae,
        "profile": plan.profile,
        "persist_av_latent": plan.persist_av_latent,
        "continuation_mode": plan.continuation_mode,
    });
    if let Some(obj) = request.as_object_mut() {
        obj.insert("jev".into(), json!(plan.jev));
        obj.insert("jev_sdk_python".into(), json!(plan.jev_sdk_python));
        obj.insert("jev_initial_policy".into(), json!(plan.jev_initial_policy));
        obj.insert("sla_fixed".into(), json!(plan.sla_fixed));
        obj.insert("sla_table".into(), json!(plan.sla_table));
        obj.insert("jev_log_dataset".into(), json!(plan.jev_log_dataset));
        obj.insert("no_sla".into(), json!(plan.no_sla));
    }
    let request_path = work_dir.join("request.json");
    std::fs::write(&request_path, serde_json::to_vec_pretty(&request)?)
        .with_context(|| format!("write {}", request_path.display()))?;

    println!(
        "[h3 generate] mode={} engine={} {}x{} frames={} steps={} sampler={} shift={}/{} attn={} turbo={} realism={} sol={} vsa={} jev={} cache={}",
        plan.mode,
        plan.engine,
        plan.width,
        plan.height,
        plan.frames,
        plan.steps,
        plan.sampler,
        plan.shift_video,
        plan.shift_audio,
        plan.attn,
        plan.turbo,
        plan.realism,
        plan.sol,
        plan.vsa,
        plan.jev,
        plan.cache
    );
    if plan.no_sla {
        println!("[h3 generate] no-sla 4-step res_multistep (no H3JevNativeSLAPatch)");
    }
    if let Some(keep) = plan.sla_fixed {
        println!("[h3 generate] sla-fixed {keep}% const keep (no Jev worker)");
    }
    if let Some(table) = &plan.sla_table {
        println!(
            "[h3 generate] sla-table {} per-(step,layer) keep (no Jev worker)",
            table.display()
        );
    }
    if let Some(p) = &plan.first_frame {
        println!("[h3 generate] first_frame={}", p.display());
    }
    if !plan.refs.is_empty() {
        println!("[h3 generate] refs={}", plan.refs.len());
    }
    if !plan.refmods.is_empty() {
        println!(
            "[h3 generate] refmods={}",
            plan.refmods
                .iter()
                .map(|m| m.spec())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!("[h3 generate] output={}", plan.output.display());
    if plan.two_stage {
        println!(
            "[h3 generate] Eros stage-1 {}x{} SplitSigmas@4 → {}",
            plan.width,
            plan.height,
            worker_out.display()
        );
    }
    println!("[h3 generate] unloading Gemmy before H3...");

    let verbose = args.verbose;
    let python = plan.python.clone();
    let worker = plan.worker.clone();
    let out_mp4 = worker_out.clone();
    let req_path = request_path.clone();
    let ckpt_root = plan.checkpoints_root.clone();
    let keep_work = plan.keep_work;
    let work_for_cleanup = work_dir.clone();
    let jev_api_key = if plan.jev && plan.jev_initial_policy == JEV_INITIAL_POLICY {
        super::paths::typesafe_api_key()
    } else {
        None
    };

    with_gpu_handoff(verbose, || {
        // Do not use python_worker_command (-I): H3 needs site-packages from the
        // runtime venv (torch, sageattention, minimax_h3 editable, etc.).
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        // Scripts resolve TE/VAE weights via this env (code root ≠ checkpoints root).
        cmd.env("GEMMY_H3_CHECKPOINTS", ckpt_root.as_os_str());
        if let Some(key) = &jev_api_key {
            cmd.env("TYPESAFE_API_KEY", key);
        }
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        super::paths::forward_refmods_dir(&mut cmd);
        cmd.arg(&worker).arg("--request").arg(&req_path);
        WorkerSpec::new(cmd, "video.h3.generate")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(out_mp4.clone()).labeled("H3 output mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    if plan.two_stage {
        let img = plan.refs.iter().find_map(|r| {
            if r.kind == "image" {
                r.path.clone()
            } else {
                None
            }
        });
        let audios: Vec<PathBuf> = plan
            .refs
            .iter()
            .filter(|r| r.kind == "audio")
            .filter_map(|r| r.path.clone())
            .collect();
        super::upscale::run_eros_stage2(
            &worker_out,
            &plan.output,
            &plan.prompt,
            &plan.weights,
            img.as_deref(),
            &audios,
            plan.seed,
            verbose,
            config,
            plan.width,
            plan.height,
            plan.vsa,
            plan.vsa_sparsity,
            &plan.vsa_gate,
            &plan.refmods,
            &args.vae_select,
        )?;
    }

    if !keep_work {
        let _ = std::fs::remove_dir_all(&work_for_cleanup);
    } else {
        println!("[h3 generate] kept work dir {}", work_for_cleanup.display());
    }

    println!("wrote H3 video: {}", plan.output.display());
    if args.json {
        let summary = json!({
            "ok": true,
            "engine": plan.engine,
            "plan": plan,
            "output": plan.output,
            "work_dir": if keep_work { Some(work_for_cleanup) } else { None },
        });
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }
    Ok(())
}

fn generate_krea_still(
    prompt: &str,
    output: &Path,
    width: u32,
    height: u32,
    seed: u64,
    verbose: bool,
) -> Result<()> {
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let gemmy_exe = crate::host::gemmy::require_gemmy(
        "I2V auto still is `gemmy image --engine krea2`",
    )?;
    let seed_i = i64::try_from(seed).unwrap_or(i64::MAX);
    println!(
        "[h3 generate] auto still via Krea 2 {}x{} → {}",
        width,
        height,
        output.display()
    );
    let mut cmd = crate::host::workers::env::gemmy_child_command(&gemmy_exe);
    cmd.arg("image")
        .arg("--engine")
        .arg("krea2")
        .arg("--width")
        .arg(width.to_string())
        .arg("--height")
        .arg(height.to_string())
        .arg("--seed")
        .arg(seed_i.to_string())
        .arg("-o")
        .arg(output)
        .arg(prompt);
    if verbose {
        cmd.arg("-v");
    }
    run_simple_command(cmd, "h3 krea2 still", verbose)
        .with_context(|| format!("Krea 2 still generation for {}", output.display()))?;
    if !output.is_file() {
        bail!(
            "Krea 2 still missing after generate: {}\n\
             Pass --first-frame PATH to skip auto-still, or run `gemmy image --engine krea2` alone to diagnose.",
            output.display()
        );
    }
    println!("[h3 generate] wrote Krea still {}", output.display());
    Ok(())
}

pub(crate) fn build_plan(args: &H3GenerateArgs) -> Result<H3Plan> {
    let mut prompt = resolve_prompt(args)?;
    if prompt.trim().is_empty() {
        bail!("prompt is required (--prompt or --prompt-file)");
    }

    let still_prompt = args
        .still_prompt
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let has_refs = !args.ref_image.is_empty()
        || !args.ref_video.is_empty()
        || !args.ref_audio.is_empty()
        || !args.ref_av.is_empty();

    let mut auto_still = false;
    let allow_keyframe_refs = args.allow_keyframe_refs;
    match args.mode {
        H3Mode::T2va => {
            if args.first_frame.is_some() || args.last_frame.is_some() {
                bail!(
                    "t2va mode does not take --first-frame/--last-frame \
                     (use --mode i2v, fl2va, or ref2va)"
                );
            }
            if has_refs {
                bail!("t2va mode does not take --ref-* (use --mode ref2va)");
            }
            if allow_keyframe_refs {
                bail!("--allow-keyframe-refs needs FL2VA/I2V keyframes + --ref-*");
            }
            if args.still_prompt.is_some() {
                bail!("--still-prompt is only used with default I2V auto-still");
            }
        }
        H3Mode::I2v => {
            if args.last_frame.is_some() {
                bail!("i2v mode does not take --last-frame (use --mode fl2va)");
            }
            if has_refs && !allow_keyframe_refs {
                bail!(
                    "i2v mode does not take --ref-* unless --allow-keyframe-refs \
                     (experimental Multishot DiT memory; prefer --mode ref2va)"
                );
            }
            if allow_keyframe_refs && !has_refs {
                bail!("--allow-keyframe-refs requires at least one --ref-image (or other --ref-*)");
            }
            if args.first_frame.is_none() {
                if args.no_auto_still {
                    bail!(
                        "i2v mode needs --first-frame (or omit --no-auto-still to generate one with Krea 2)"
                    );
                }
                if crate::host::gemmy::gemmy_exe().is_none() {
                    bail!(
                        "i2v mode needs --first-frame. This CLI does not generate stills. \
                         Install `gemmy` on PATH to delegate the Krea still, or pass --first-frame."
                    );
                }
                auto_still = true;
            }
        }
        H3Mode::Fl2va => {
            if args.first_frame.is_none() || args.last_frame.is_none() {
                bail!(
                    "fl2va mode requires both --first-frame and --last-frame \
                     (auto Krea still is I2V-only)"
                );
            }
            if has_refs && !allow_keyframe_refs {
                bail!(
                    "fl2va mode does not take --ref-* unless --allow-keyframe-refs \
                     (experimental Multishot DiT memory; prefer --mode ref2va)"
                );
            }
            if allow_keyframe_refs && !has_refs {
                bail!("--allow-keyframe-refs requires at least one --ref-image (or other --ref-*)");
            }
            if args.still_prompt.is_some() {
                bail!("--still-prompt is only used with I2V auto-still");
            }
        }
        H3Mode::Ref2va => {
            if args.first_frame.is_some() || args.last_frame.is_some() {
                if !allow_keyframe_refs {
                    bail!(
                        "ref2va mode does not take --first-frame/--last-frame without \
                         --allow-keyframe-refs (use --ref-* only, or enable experimental mix)"
                    );
                }
            }
            if args.still_prompt.is_some() || args.no_auto_still {
                bail!("--still-prompt / --no-auto-still are I2V-only");
            }
            if !has_refs && args.ref_mod.is_empty() {
                bail!(
                    "ref2va mode needs at least one --ref-image, --ref-video, --ref-av, \
                     or --ref-mod (--ref-audio alone is not enough)"
                );
            }
        }
    }

    if !args.ref_mod.is_empty()
        && matches!(args.mode, H3Mode::I2v | H3Mode::Fl2va)
    {
        bail!("--ref-mod needs --mode t2va or --mode ref2va (not I2V / first-last)");
    }

    if let Some(s) = args.steps {
        if s == 0 || s > 100 {
            bail!("--steps must be in 1..=100");
        }
    }
    if !args.shift_video.is_finite()
        || !args.shift_audio.is_finite()
        || args.shift_video < 0.0
        || args.shift_audio < 0.0
    {
        bail!("--shift-video / --shift-audio must be finite and >= 0 (0 with the other 0 skips MiniMaxH3SigmaShift)");
    }
    let skip_shift = args.shift_video == 0.0 && args.shift_audio == 0.0;
    if !skip_shift && (args.shift_video == 0.0 || args.shift_audio == 0.0) {
        bail!(
            "pass both --shift-video 0 --shift-audio 0 to skip MiniMaxH3SigmaShift, or both > 0 (stock and both source graphs: 12 / 3)"
        );
    }

    let singularity = args.dit == H3Dit::Singularity;
    if singularity && args.weights.is_some() {
        bail!(
            "--dit singularity selects {SINGULARITY_REF2VA_INT8}; do not also pass --weights"
        );
    }
    if singularity && !matches!(args.mode, H3Mode::Ref2va) {
        bail!(
            "--dit singularity is the WarmBloodAban Ref2VA fusion. \
             Pass --mode ref2va with --ref-image or --ref-mod. \
             It does not replace default I2V or Eros."
        );
    }
    if singularity && args.dit_quant != H3DitQuant::Int8 {
        bail!("--dit singularity is the INT8 file; do not pass --dit-quant w4a8");
    }

    let eros_default = matches!(args.mode, H3Mode::Ref2va)
        && args.weights.is_none()
        && !allow_keyframe_refs
        && !singularity;
    // Two-stage 0.2 MP canvas is for live Picture 1. Saved RefMods alone use
    // Text Encode at the normal native canvas (480p unless --canvas/--width).
    // 009jev is the author's 4-step res_multistep recipe, not Eros two-stage.
    let eros_two_stage =
        eros_default && has_refs && !args.jev && args.sla_fixed.is_none() && args.sla_table.is_none() && !args.no_sla;
    let canvas = if singularity && args.canvas.is_none() && args.width.is_none() && args.height.is_none()
    {
        // ResolutionSelector 9:16, 0.5 MP, multiple 32 from the AI Brief graph.
        super::canvas::ResolvedCanvas {
            width: 544,
            height: 960,
            preset: None,
            megapixels: (544.0 * 960.0) / 1_000_000.0,
        }
    } else if eros_two_stage && args.canvas.is_none() && args.width.is_none() && args.height.is_none()
    {
        let p = H3CanvasPreset::Mp02;
        let (w, h) = p.size();
        super::canvas::ResolvedCanvas {
            width: w,
            height: h,
            preset: Some(p),
            megapixels: p.megapixels(),
        }
    } else {
        resolve_sample_canvas(args.canvas, args.width, args.height)?
    };
    if args.verbose {
        eprintln!("[h3 generate] canvas={}", canvas.label());
    }

    let frames = if args.frames > 0 {
        snap_frames(args.frames)
    } else {
        if !(args.duration.is_finite() && args.duration > 0.0) {
            bail!("--duration must be finite and > 0 (or pass --frames)");
        }
        let raw = (args.duration * f64::from(FPS)).round() as u32;
        snap_frames(raw.max(1))
    };
    let duration_s = f64::from(frames) / f64::from(FPS);

    let mut compile_ir = args.compile_ir && !args.no_compile_ir;
    if !args.ref_mod.is_empty() && !has_refs {
        // Text Encode presents `<Picture n>` / `<Video n>`. Do not rewrite those tags.
        compile_ir = false;
    } else if eros_default && compile_ir && !prompt.contains("subject_definitions:") {
        prompt = compile_scene_ir(&prompt, duration_s, 1, 1);
        compile_ir = false;
    }
    if singularity {
        compile_ir = false;
    }

    let first_frame = match &args.first_frame {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() {
                bail!("first-frame not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };
    let last_frame = match &args.last_frame {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() {
                bail!("last-frame not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };

    let refs = resolve_refs(args)?;
    if singularity {
        let images = refs.iter().filter(|r| r.kind == "image").count();
        let other = refs.iter().any(|r| r.kind != "image");
        if images != 1 || other {
            bail!(
                "--dit singularity needs exactly one --ref-image. \
                 The dual-sample graph does not take --ref-video, --ref-audio, or extra stills."
            );
        }
    }
    let mut weights = match &args.weights {
        Some(p) => abs(p)?,
        // Keyframe+ref mix is Multishot DiT memory — prefer Ref2VA partition.
        None if allow_keyframe_refs => stock_ref2va_backup_path(),
        None if singularity => singularity_ref2va_int8_path(),
        None => default_dit_weights_for_mode(args.mode),
    };
    if args.weights.is_none()
        && args.dit_quant == H3DitQuant::W4a8
        && !matches!(args.mode, H3Mode::Ref2va)
        && !allow_keyframe_refs
    {
        weights = checkpoints_root()
            .join("diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors");
    }

    let output = absolute_path(&args.output.clone().unwrap_or_else(|| {
        default_output_path(format!("h3_{}.mp4", timestamp_slug()))
    }))?;

    let engine = args.engine;
    let mut video_vae = resolve_video_vae(
        &args.vae_select,
        None,
        engine == H3Engine::Python,
        args.vae_select.video_vae.is_some() || !args.dry_run_plan,
    )?;
    if singularity && args.vae_select.vae.is_none() && args.vae_select.video_vae.is_none() {
        video_vae.filename = VIDEO_VAE_FP16_FILE.to_string();
        video_vae.fingerprint_id = super::paths::video_vae_fingerprint_id(VIDEO_VAE_FP16_FILE);
        video_vae.path = checkpoints_root().join("vae").join(VIDEO_VAE_FP16_FILE);
    }
    if !args.ref_mod.is_empty() && engine == H3Engine::Python {
        bail!("--ref-mod requires --engine comfy (MiniMaxH3Mod)");
    }
    if eros_two_stage && engine == H3Engine::Python {
        bail!("--mode ref2va Eros two-stage requires --engine comfy (python engine cannot run SplitSigmas@4 + stage-2)");
    }
    if singularity && engine == H3Engine::Python {
        bail!("--dit singularity requires --engine comfy");
    }
    let turbo = if singularity && args.turbo != H3Turbo::Off {
        eprintln!(
            "[h3 generate] Singularity does not load the Larryvrh turbo LoRA — ignoring --turbo"
        );
        H3Turbo::Off
    } else if eros_default && args.turbo != H3Turbo::Off {
        eprintln!(
            "[h3 generate] Eros Ref2VA UNET is already the turbo merge — ignoring --turbo (Larryvrh LoRA does not stack)"
        );
        H3Turbo::Off
    } else {
        args.turbo
    };
    let sol = args.sol && !args.no_sol;
    let cache = args.cache;
    let vsa = args.vsa;
    let vsa_sparsity = args.vsa_sparsity;
    let vsa_gate = {
        let name = args.vsa_gate.trim();
        if name.is_empty() {
            super::paths::VSA_GATE_DEFAULT.to_string()
        } else {
            Path::new(name)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| name.to_string())
        }
    };
    if vsa && !matches!(args.mode, H3Mode::Ref2va) {
        bail!("--vsa is Ref2VA-only (Kablex gate transplant on MiniMaxH3ReferenceToVideo)");
    }
    if vsa && sol {
        bail!("--vsa cannot stack with --sol (Saganaki Sol-Attn). Pick one attention owner.");
    }
    if vsa && cache != H3Cache::Off {
        bail!(
            "--vsa cannot stack with --cache {}. Leave cache off.",
            cache.as_str()
        );
    }
    if vsa && !args.ref_mod.is_empty() {
        bail!("--vsa cannot stack with --ref-mod (no compiled combo package)");
    }
    let sla_fixed = args.sla_fixed;
    let sla_table = args.sla_table.clone();
    if args.jev && sla_fixed.is_some() {
        bail!("--jev cannot stack with --sla-fixed (adaptive Jev vs true fixed keep). Pick one.");
    }
    if args.jev && sla_table.is_some() {
        bail!("--jev cannot stack with --sla-table (adaptive Jev vs per-cell keep table). Pick one.");
    }
    if sla_fixed.is_some() && sla_table.is_some() {
        bail!("--sla-fixed cannot stack with --sla-table (const keep vs per-cell table). Pick one.");
    }
    if args.no_sla && (args.jev || sla_fixed.is_some() || sla_table.is_some()) {
        bail!("--no-sla cannot stack with --jev, --sla-fixed, or --sla-table (dense 4-step vs native SLA). Pick one.");
    }
    let native_sla = args.jev || sla_fixed.is_some() || sla_table.is_some();
    let sla_flag = if args.jev {
        "--jev"
    } else if sla_table.is_some() {
        "--sla-table"
    } else {
        "--sla-fixed"
    };
    let res4 = native_sla || args.no_sla;
    let res4_flag = if args.no_sla {
        "--no-sla"
    } else {
        sla_flag
    };
    if res4 && vsa {
        bail!("{res4_flag} cannot stack with --vsa (native SLA vs Kablex VSA). Pick one.");
    }
    if res4 && sol {
        bail!("{res4_flag} cannot stack with --sol (Saganaki Sol-Attn). Pick one attention owner.");
    }
    if res4 && cache != H3Cache::Off {
        bail!(
            "{res4_flag} cannot stack with --cache {}. Leave cache off.",
            cache.as_str()
        );
    }
    if res4 && turbo != H3Turbo::Off {
        bail!("{res4_flag} cannot stack with --turbo (native SLA / --no-sla, not Larryvrh turbo)");
    }
    if res4 && args.realism {
        bail!("{res4_flag} cannot stack with --realism (no compiled combo package)");
    }
    if res4 && !args.ref_mod.is_empty() {
        bail!("{res4_flag} cannot stack with --ref-mod (no compiled combo package)");
    }
    if res4 && allow_keyframe_refs {
        bail!("{res4_flag} cannot stack with --allow-keyframe-refs");
    }
    if res4 && engine == H3Engine::Python {
        bail!("{res4_flag} requires --engine comfy");
    }
    if args.jev_log_dataset.is_some() && !native_sla {
        bail!("--jev-log-dataset requires --jev, --sla-fixed, or --sla-table");
    }
    if args.jev && !typesafe_api_key_present() {
        bail!(
            "--jev requires TYPESAFE_API_KEY in the process environment, config env_overrides, or repo `.env`. \
             Default generate and --sla-fixed do not set or need this key. Do not put the key in workflows or git."
        );
    }
    let node = jev_native_sla_dir().join("native_sla.py");
    if native_sla && !node.is_file() {
        bail!(
            "{sla_flag} needs H3JevNativeSLAPatch (009jev) at {}",
            node.display()
        );
    }
    let (jev_sdk_python, jev_initial_policy) = if args.jev {
        let sdk = jev_sdk_python();
        if !sdk.is_file() {
            bail!(
                "--jev needs the dedicated SDK Python with typesafe-sdk==0.7.0: {}\n\
                 From runtimes/minimax-h3/jev-sdk run: uv sync",
                sdk.display()
            );
        }
        (sdk.to_string_lossy().into_owned(), JEV_INITIAL_POLICY.to_string())
    } else if let Some(keep) = sla_fixed {
        (String::new(), format!("const{keep}"))
    } else if sla_table.is_some() {
        (String::new(), "table".to_string())
    } else {
        (String::new(), String::new())
    };
    let table_n = if let Some(path) = &sla_table {
        if !path.is_file() {
            bail!("--sla-table file not found: {}", path.display());
        }
        Some(validate_sla_table_file(path)?)
    } else {
        None
    };
    let jev = native_sla;
    if !(0.0..1.0).contains(&vsa_sparsity) {
        bail!("--vsa-sparsity must be in [0, 1) (got {vsa_sparsity})");
    }
    if engine == H3Engine::Python
        && (turbo != H3Turbo::Off
            || args.realism
            || sol
            || vsa
            || jev
            || sla_fixed.is_some()
            || sla_table.is_some()
            || args.no_sla
            || cache != H3Cache::Off)
    {
        bail!(
            "--turbo / --realism / --sol / --vsa / --jev / --sla-fixed / --sla-table / --no-sla / --cache require --engine comfy \
             (python engine is the streamed run_sample path only)"
        );
    }
    if vsa {
        let gate = vsa_gate_path(&vsa_gate);
        if !gate.is_file() {
            bail!(
                "--vsa needs the FastH3 gate file: {}\n\
                 Download: https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo/resolve/main/fasth3_vsa_gate.safetensors\n\
                 Place it under checkpoints/loras/ (see docs/MODEL_LOCATIONS.md).",
                gate.display()
            );
        }
    }
    if singularity
        && (args.jev
            || args.sla_fixed.is_some()
            || args.sla_table.is_some()
            || args.no_sla
            || vsa
            || sol
            || cache != H3Cache::Off
            || !args.ref_mod.is_empty())
    {
        bail!(
            "--dit singularity is the AI Brief dual-sample graph. \
             Do not stack --jev, --sla-fixed, --sla-table, --no-sla, --vsa, --sol, --cache, or --ref-mod."
        );
    }
    if args.realism && !(0.0..=2.0).contains(&args.realism_strength) {
        bail!(
            "--realism-strength must be in 0.0..=2.0 (got {})",
            args.realism_strength
        );
    }

    let mut steps = match args.steps {
        Some(s) => s,
        None if singularity => 12,
        None if table_n.is_some() => table_n.unwrap() as u32,
        None if native_sla || args.no_sla => JEV_STEPS,
        None if eros_default => EROS_REF2VA_STEPS,
        None => 20,
    };
    if let Some(n) = table_n {
        if steps != n as u32 {
            bail!("--sla-table has {n} steps but --steps is {steps}; they must match");
        }
    }
    if args.no_sla && steps != JEV_STEPS {
        bail!("--no-sla is the 4-step dense control; omit --steps or pass --steps 4");
    }
    if args.no_sla && last_frame.is_some() {
        bail!("--no-sla has no first+last FL2VA package");
    }
    if native_sla && last_frame.is_some() && steps == JEV_STEPS {
        bail!(
            "{sla_flag} has no first+last 4-step package; pass --steps 20 (HQ euler) for --mode fl2va --last-frame"
        );
    }
    if singularity && steps != 12 {
        bail!(
            "--dit singularity keeps the AI Brief schedule: BasicScheduler 6, \
             ExtendIntermediateSigmas 2, SplitSigmas at step 2 (12 denoise steps). Omit --steps."
        );
    }
    if eros_default && !native_sla && !args.no_sla && steps == 20 {
        steps = EROS_REF2VA_STEPS;
    }
    if turbo != H3Turbo::Off {
        if let Some(ts) = args.turbo_steps {
            steps = ts;
        } else if steps == 20 {
            // Product default for turbo when caller left stock 20-step HQ default.
            steps = 4;
        }
        if !(4..=8).contains(&steps) {
            eprintln!(
                "[h3 generate] warn: turbo steps={steps} outside Larryvrh 4–8 band"
            );
        }
    }
    let sampler = if (native_sla || args.no_sla) && steps == JEV_STEPS {
        "res_multistep".to_string()
    } else {
        "euler".to_string()
    };

    let worker = match engine {
        H3Engine::Comfy => comfy_worker_script(),
        H3Engine::Python => worker_script(),
    };
    let comfy = match engine {
        H3Engine::Comfy => Some(comfy_root()),
        H3Engine::Python => None,
    };

    Ok(H3Plan {
        mode: args.mode.as_str().into(),
        quality: args.quality.as_str().into(),
        attn: args.quality.attn().into(),
        engine: engine.as_str().into(),
        turbo: turbo.as_str().into(),
        turbo_strength: args.turbo_strength,
        realism: args.realism || singularity,
        realism_strength: args.realism_strength,
        sol,
        vsa,
        vsa_sparsity,
        vsa_gate,
        jev,
        jev_sdk_python,
        jev_initial_policy,
        sla_fixed,
        sla_table,
        jev_log_dataset: args.jev_log_dataset.clone(),
        no_sla: args.no_sla,
        cache: cache.as_str().into(),
        dit_quant: args.dit_quant.as_str().into(),
        prompt,
        still_prompt,
        compile_ir,
        canvas: canvas.preset.map(|p| p.as_str().into()),
        width: canvas.width,
        height: canvas.height,
        megapixels: canvas.megapixels,
        frames,
        duration_s,
        steps,
        sampler,
        seed: args.seed,
        shift_video: args.shift_video,
        shift_audio: args.shift_audio,
        first_frame,
        last_frame,
        refs,
        refmods: parse_ref_mods(&args.ref_mod)?,
        allow_keyframe_refs,
        auto_still,
        auto_still_engine: if auto_still {
            Some("krea2".into())
        } else {
            None
        },
        weights,
        output: output.clone(),
        h3_root: h3_root(),
        checkpoints_root: checkpoints_root(),
        comfy_root: comfy,
        python: h3_python(),
        worker,
        profile: args.profile,
        keep_work: args.keep_work,
        persist_av_latent: engine == H3Engine::Comfy,
        av_latent: {
            let mut p = output.clone();
            p.set_extension("h3av.safetensors");
            p
        },
        continuation_mode: if singularity && args.continuation_mode == "none" {
            "singularity_dual".into()
        } else if eros_two_stage && args.continuation_mode == "none" {
            "ganloss_two_stage".into()
        } else {
            args.continuation_mode.clone()
        },
        two_stage: eros_two_stage || args.continuation_mode == "ganloss_two_stage",
        video_vae: video_vae.filename,
    })
}

pub(crate) fn parse_ref_mods(raw: &[String]) -> Result<Vec<H3RefModUse>> {
    if raw.len() > 4 {
        bail!("--ref-mod cap is 4 (got {})", raw.len());
    }
    let mut out = Vec::new();
    for spec in raw {
        let spec = spec.trim();
        if spec.is_empty() {
            bail!("--ref-mod value is empty");
        }
        let mut parts = spec.split(':');
        let name = parts
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches(".safetensors")
            .to_string();
        if name.is_empty() || name == "(none)" {
            bail!("--ref-mod name is empty");
        }
        let strength = match parts.next() {
            Some(s) if !s.trim().is_empty() => s
                .trim()
                .parse::<f64>()
                .with_context(|| format!("--ref-mod strength in {spec}"))?,
            _ => 1.0,
        };
        if !(0.0..=1.0).contains(&strength) {
            bail!("--ref-mod strength must be 0..=1 (got {strength} in {spec})");
        }
        let copies = match parts.next() {
            Some(s) if !s.trim().is_empty() => s
                .trim()
                .parse::<u32>()
                .with_context(|| format!("--ref-mod copies in {spec}"))?,
            _ => 1,
        };
        if copies == 0 || copies > 10 {
            bail!("--ref-mod copies must be 1..=10 (got {copies} in {spec})");
        }
        if parts.next().is_some() {
            bail!("--ref-mod expected name[:strength[:copies]] (got {spec})");
        }
        out.push(H3RefModUse {
            name,
            strength,
            copies,
        });
    }
    Ok(out)
}

fn resolve_refs(args: &H3GenerateArgs) -> Result<Vec<H3RefSpec>> {
    let mut refs = Vec::new();
    // Clap preserves per-flag order but not interleaved cross-flag order.
    // Collect image → video → audio → av (package still validates caps).
    for p in &args.ref_image {
        let path = abs(p)?;
        if !path.is_file() {
            bail!("--ref-image not found: {}", path.display());
        }
        refs.push(H3RefSpec {
            kind: "image".into(),
            path: Some(path),
            video: None,
            audio: None,
        });
    }
    for p in &args.ref_video {
        let path = abs(p)?;
        if !path.is_file() {
            bail!("--ref-video not found: {}", path.display());
        }
        refs.push(H3RefSpec {
            kind: "video".into(),
            path: Some(path),
            video: None,
            audio: None,
        });
    }
    for p in &args.ref_audio {
        let path = abs(p)?;
        if !path.is_file() {
            bail!("--ref-audio not found: {}", path.display());
        }
        refs.push(H3RefSpec {
            kind: "audio".into(),
            path: Some(path),
            video: None,
            audio: None,
        });
    }
    for raw in &args.ref_av {
        let (v, a) = parse_ref_av(raw)?;
        let video = abs(&v)?;
        let audio = abs(&a)?;
        if !video.is_file() {
            bail!("--ref-av video not found: {}", video.display());
        }
        if !audio.is_file() {
            bail!("--ref-av audio not found: {}", audio.display());
        }
        refs.push(H3RefSpec {
            kind: "av".into(),
            path: None,
            video: Some(video),
            audio: Some(audio),
        });
    }

    if refs.is_empty() {
        return Ok(refs);
    }

    let n_img = refs.iter().filter(|r| r.kind == "image").count();
    let n_vid = refs
        .iter()
        .filter(|r| r.kind == "video" || r.kind == "av")
        .count();
    let n_aud = refs
        .iter()
        .filter(|r| r.kind == "audio" || r.kind == "av")
        .count();
    // Each av pair is 2 files; image/video/audio are 1.
    let n_files: usize = refs
        .iter()
        .map(|r| if r.kind == "av" { 2 } else { 1 })
        .sum();

    if n_img > MAX_REF_IMAGES {
        bail!("--ref-image cap is {MAX_REF_IMAGES} (got {n_img})");
    }
    if n_vid > MAX_REF_VIDEOS {
        bail!("--ref-video + --ref-av video slots cap is {MAX_REF_VIDEOS} (got {n_vid})");
    }
    if n_aud > MAX_REF_AUDIOS {
        bail!("--ref-audio + --ref-av audio slots cap is {MAX_REF_AUDIOS} (got {n_aud})");
    }
    if n_files > MAX_REF_FILES {
        bail!("Ref2VA total file cap is {MAX_REF_FILES} (got {n_files})");
    }
    let has_visual = n_img + n_vid > 0;
    if !has_visual {
        bail!("Ref2VA cannot use audio as the sole input — add --ref-image, --ref-video, or --ref-av");
    }
    Ok(refs)
}

fn parse_ref_av(raw: &str) -> Result<(PathBuf, PathBuf)> {
    // Prefer comma split (matches package CLI VIDEO.safetensors,AUDIO.safetensors).
    let parts: Vec<&str> = if raw.contains(',') {
        raw.splitn(2, ',').collect()
    } else if raw.contains(';') {
        raw.splitn(2, ';').collect()
    } else {
        bail!(
            "--ref-av expects VIDEO_PATH,AUDIO_PATH (got {raw:?})"
        );
    };
    if parts.len() != 2 {
        bail!("--ref-av expects VIDEO_PATH,AUDIO_PATH (got {raw:?})");
    }
    let v = parts[0].trim();
    let a = parts[1].trim();
    if v.is_empty() || a.is_empty() {
        bail!("--ref-av expects non-empty VIDEO_PATH,AUDIO_PATH (got {raw:?})");
    }
    Ok((PathBuf::from(v), PathBuf::from(a)))
}

fn resolve_prompt(args: &H3GenerateArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        let path = abs(path)?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()))?;
        return Ok(text);
    }
    Ok(args.prompt.clone().unwrap_or_default())
}

pub(crate) const AV_SAFE_FRAMES: &[u32] = &[39, 90, 141, 192, 243, 294, 345, 396];

fn snap_frames(n: u32) -> u32 {
    let mut fc = n.max(FRAME_REM);
    while fc % FRAME_MOD != FRAME_REM {
        fc += 1;
    }
    fc
}

pub(crate) fn snap_h3_frames(n: u32) -> u32 {
    snap_frames(n)
}

/// Largest AV-safe context length that fits in both the request and the predecessor.
pub(crate) fn snap_av_context_frames(requested: u32, available: u32) -> Result<u32> {
    let cap = requested.min(available);
    let mut chosen = 0u32;
    for &n in AV_SAFE_FRAMES {
        if n <= cap {
            chosen = n;
        } else {
            break;
        }
    }
    if chosen > 0 {
        return Ok(chosen);
    }
    let mut n = cap;
    while n >= FRAME_REM && n % FRAME_MOD != FRAME_REM {
        n -= 1;
    }
    // 24 fps × 40 Hz audio: audio_steps * 3 == frames * 5
    if n >= FRAME_REM && u64::from(audio_steps_for_frames(n)) * 3 == u64::from(n) * 5 {
        return Ok(n);
    }
    bail!("no AV-safe context length fits requested={requested} available={available}");
}

fn audio_steps_for_frames(frame_count: u32) -> u32 {
    ((f64::from(frame_count) / f64::from(FPS)) * 40.0).round() as u32
}

pub(crate) fn av_sidecar_paths(mp4: &Path) -> (PathBuf, PathBuf) {
    let mut st = mp4.to_path_buf();
    st.set_extension("h3av.safetensors");
    let mut meta = mp4.to_path_buf();
    meta.set_extension("h3av.json");
    (st, meta)
}

#[cfg(test)]
mod tests {
    use super::super::args::{H3GenerateArgs, H3Quality};
    use super::*;

    #[test]
    fn snap_frames_grid() {
        assert_eq!(snap_frames(1), 5);
        assert_eq!(snap_frames(5), 5);
        assert_eq!(snap_frames(6), 22);
        assert_eq!(snap_frames(120), 124);
        assert_eq!(snap_frames(124), 124);
    }

    #[test]
    fn snap_context_default_39() {
        assert_eq!(snap_av_context_frames(39, 362).unwrap(), 39);
        assert_eq!(snap_av_context_frames(100, 200).unwrap(), 90);
        assert_eq!(snap_av_context_frames(141, 141).unwrap(), 141);
    }

    #[test]
    fn quality_attn_mapping() {
        assert_eq!(H3Quality::Fast.attn(), "sage");
        assert_eq!(H3Quality::Hq.attn(), "fa2");
    }

    #[test]
    fn parse_ref_av_comma() {
        let (v, a) = parse_ref_av(r"C:\clips\v.mp4,C:\clips\a.wav").unwrap();
        assert!(v.to_string_lossy().ends_with("v.mp4"));
        assert!(a.to_string_lossy().ends_with("a.wav"));
    }

    #[test]
    fn default_i2v_resolves_stock_fl2va_not_eros() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "i2v");
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(
            name.contains("minimax_h3_fl2va_pruned_int8_convrot"),
            "default I2V dit={name}"
        );
        assert!(!name.contains("10Eros"));
        assert!(!name.contains("ref2va"));
        assert_eq!(plan.steps, 20);
        assert!(!plan.two_stage);
        assert!(!plan.jev);
        assert!(plan.jev_sdk_python.is_empty());
        assert_eq!(plan.video_vae, VIDEO_VAE_INT8_FILE);
    }

    #[test]
    fn dry_run_vae_fp16_override() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--vae",
            "fp16",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.video_vae, VIDEO_VAE_FP16_FILE);
    }

    #[test]
    fn dry_run_missing_video_vae_fail_closes() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--video-vae",
            "not_a_real_vae.safetensors",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err();
        assert!(err.to_string().contains("not_a_real_vae"));
    }

    #[test]
    fn python_engine_int8_fail_closes() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--engine",
            "python",
            "--vae",
            "int8",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("python"), "{msg}");
        assert!(msg.contains("fp16"), "{msg}");
    }

    #[test]
    fn default_ref2va_uses_eros_two_stage() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "ref2va");
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(name.contains("10Eros_Max_h3_TURBO_ref2va_beta2"), "dit={name}");
        assert!(name.contains("int8_convrot"), "dit={name}");
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
        assert_eq!(plan.width, 608);
        assert_eq!(plan.height, 352);
        assert_eq!(plan.continuation_mode, "ganloss_two_stage");
        assert!(plan.two_stage);
        assert!(!plan.compile_ir);
        assert!(plan.prompt.contains("subject_definitions:"));
        assert!(!plan.prompt.contains("[Shot"));
        assert!(!plan.prompt.to_ascii_lowercase().contains("storyboard"));
        assert_eq!(plan.turbo, "off");
        assert!(!plan.vsa);
        assert!(!plan.jev);
        assert!(plan.jev_sdk_python.is_empty());
    }

    #[test]
    fn singularity_ref2va_is_dual_sample() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--dit",
            "singularity",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert_eq!(name, SINGULARITY_REF2VA_INT8);
        assert!(!name.contains("Pruned"), "{name}");
        assert!(!plan.two_stage);
        assert_eq!(plan.continuation_mode, "singularity_dual");
        assert_eq!(plan.steps, 12);
        assert_eq!(plan.sampler, "euler");
        assert_eq!(plan.width, 544);
        assert_eq!(plan.height, 960);
        assert!(!plan.compile_ir);
        assert!(plan.realism);
        assert_eq!(plan.video_vae, VIDEO_VAE_FP16_FILE);
    }

    #[test]
    fn singularity_rejects_i2v_and_weights() {
        use clap::Parser;
        let i2v = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--dit",
            "singularity",
            "--prompt",
            "a red fox",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&i2v).unwrap_err().to_string();
        assert!(err.contains("ref2va"), "{err}");

        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let weights = dir.path().join("other.safetensors");
        std::fs::write(&weights, b"fake").unwrap();
        let both = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--dit",
            "singularity",
            "--ref-image",
            still.to_str().unwrap(),
            "--weights",
            weights.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&both).unwrap_err().to_string();
        assert!(err.contains("--weights"), "{err}");
    }

    #[test]
    fn eros_ref2va_keeps_live_audio_refs_on_two_stage() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let wav = dir.path().join("voice.wav");
        std::fs::write(&wav, b"RIFF").unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--ref-audio",
            wav.to_str().unwrap(),
            "--prompt",
            "the woman in <Picture 1> speaks with the voice of <Audio 1>",
            "--no-compile-ir",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "ref2va");
        assert!(plan.two_stage);
        assert_eq!(plan.continuation_mode, "ganloss_two_stage");
        assert_eq!(plan.steps, 8);
        let kinds: Vec<_> = plan.refs.iter().map(|r| r.kind.as_str()).collect();
        assert!(kinds.contains(&"image"), "{kinds:?}");
        assert!(kinds.contains(&"audio"), "{kinds:?}");
    }

    #[test]
    fn stock_weights_ref2va_keeps_two_audio_refs() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let a = dir.path().join("a.wav");
        let b = dir.path().join("b.wav");
        std::fs::write(&a, b"RIFF").unwrap();
        std::fs::write(&b, b"RIFF").unwrap();
        let weights = dir.path().join("minimax_h3_ref2va_pruned_int8_convrot.safetensors");
        std::fs::write(&weights, b"fake").unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--ref-audio",
            a.to_str().unwrap(),
            "--ref-audio",
            b.to_str().unwrap(),
            "--weights",
            weights.to_str().unwrap(),
            "--prompt",
            "the woman in <Picture 1> speaks; timbre mixes <Audio 1> and <Audio 2>",
            "--no-compile-ir",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "ref2va");
        assert!(!plan.two_stage);
        assert_eq!(plan.continuation_mode, "none");
        assert!(!plan.compile_ir);
        assert!(plan.prompt.contains("<Audio 1>"));
        assert!(plan.prompt.contains("<Audio 2>"));
        let kinds: Vec<&str> = plan.refs.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(kinds, ["image", "audio", "audio"]);
    }

    #[test]
    fn ref2va_vsa_mutex_sol() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--vsa",
            "--sol",
            "--dry-run-plan",
        ])
        .unwrap();
        assert!(build_plan(&args).is_err());
    }

    #[test]
    fn stock_ref2va_vsa_is_allowed_with_weights() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let weights = dir.path().join("minimax_h3_ref2va_pruned_int8_convrot.safetensors");
        std::fs::write(&weights, b"stub").unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--weights",
            weights.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--vsa",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args);
        // Gate file may be missing in unit tests; either a valid stock plan or the
        // gate-missing error. Eros-only reject must not fire.
        match plan {
            Ok(p) => {
                assert_eq!(p.mode, "ref2va");
                assert!(p.vsa);
                assert!(!p.two_stage);
            }
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("fasth3_vsa_gate") || msg.contains("--vsa needs"),
                    "unexpected error: {msg}"
                );
                assert!(!msg.contains("Eros two-stage / ganloss graphs only"));
            }
        }
    }

    #[test]
    fn eros_ref2va_ignores_larryvrh_turbo_lora() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--turbo",
            "v4",
            "--turbo-steps",
            "4",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.turbo, "off");
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
        assert!(plan.two_stage);
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(name.contains("10Eros_Max_h3_TURBO_ref2va_beta2"), "dit={name}");
    }

    #[test]
    fn portrait_ref2va_keeps_two_stage_and_swaps_axes() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--width",
            "352",
            "--height",
            "608",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.width, 352);
        assert_eq!(plan.height, 608);
        assert!(plan.two_stage);
    }

    #[test]
    fn refmod_only_ref2va_skips_two_stage_canvas() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-mod",
            "hero",
            "--prompt",
            "The woman in <Picture 1> walks through a neon market",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "ref2va");
        assert_eq!(plan.refmods.len(), 1);
        assert_eq!(plan.refmods[0].name, "hero");
        assert!(!plan.two_stage);
        assert_eq!(plan.continuation_mode, "none");
        assert_eq!(plan.width, 864);
        assert_eq!(plan.height, 480);
        assert_eq!(plan.steps, EROS_REF2VA_STEPS);
        assert!(!plan.compile_ir);
        assert!(plan.prompt.contains("<Picture 1>"));
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(name.contains("10Eros_Max_h3_TURBO_ref2va_beta2"), "dit={name}");
    }

    #[test]
    fn refmod_with_live_still_keeps_eros_two_stage() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--ref-mod",
            "hero:0.8:2",
            "--prompt",
            "walks toward camera",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert!(plan.two_stage);
        assert_eq!(plan.width, 608);
        assert_eq!(plan.height, 352);
        assert_eq!(plan.refmods[0].strength, 0.8);
        assert_eq!(plan.refmods[0].copies, 2);
    }

    #[test]
    fn refmod_rejected_on_i2v() {
        use clap::Parser;
        let dir = tempfile::tempdir().unwrap();
        let still = dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "i2v",
            "--first-frame",
            still.to_str().unwrap(),
            "--ref-mod",
            "hero",
            "--prompt",
            "walks",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--ref-mod"), "{err}");
    }

    #[test]
    fn refmod_mutex_vsa() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-mod",
            "hero",
            "--vsa",
            "--prompt",
            "The woman in <Picture 1> walks",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--vsa") && err.contains("--ref-mod"), "{err}");
    }

    struct JevOverrideGuard;
    impl Drop for JevOverrideGuard {
        fn drop(&mut self) {
            super::super::paths::set_typesafe_api_key_override(None);
            super::super::paths::set_jev_sdk_python_override(None);
        }
    }

    fn jev_sdk_stub() -> (tempfile::TempDir, std::path::PathBuf, JevOverrideGuard) {
        let dir = tempfile::tempdir().unwrap();
        let py = dir.path().join("python.exe");
        std::fs::write(&py, b"stub").unwrap();
        super::super::paths::set_jev_sdk_python_override(Some(py.clone()));
        (dir, py, JevOverrideGuard)
    }

    #[test]
    fn default_parse_does_not_enable_jev() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--dry-run-plan",
        ])
        .unwrap();
        assert!(!args.jev);
        assert!(!args.no_sla);
        let plan = build_plan(&args).unwrap();
        assert!(!plan.jev);
        assert!(!plan.no_sla);
        assert!(plan.sla_fixed.is_none());
        assert!(!plan.vsa);
        assert_eq!(plan.steps, 20);
        assert!(!plan.two_stage);
    }

    #[test]
    fn jev_opt_in_requires_typesafe_key() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        assert!(args.jev);
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("TYPESAFE_API_KEY"), "{err}");
        assert!(err.contains("--jev"), "{err}");
    }

    #[test]
    fn jev_opt_in_plan_is_4step_not_eros_two_stage() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert!(plan.jev);
        assert!(!plan.vsa);
        assert_eq!(plan.steps, JEV_STEPS);
        assert_eq!(plan.sampler, "res_multistep");
        assert_eq!(plan.jev_initial_policy, JEV_INITIAL_POLICY);
        assert_eq!(plan.jev_sdk_python, py.to_string_lossy());
        assert!(!plan.two_stage);
        assert_eq!(plan.continuation_mode, "none");
        assert_eq!(plan.mode, "i2v");
        assert_eq!(plan.engine, "comfy");
    }

    #[test]
    fn jev_ref2va_skips_eros_two_stage() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let still_dir = tempfile::tempdir().unwrap();
        let still = still_dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert!(plan.jev);
        assert!(!plan.two_stage);
        assert_eq!(plan.continuation_mode, "none");
        assert_eq!(plan.steps, JEV_STEPS);
        assert_ne!(plan.width, 608);
        assert!(!plan.vsa);
    }

    #[test]
    fn jev_mutex_vsa() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let still_dir = tempfile::tempdir().unwrap();
        let still = still_dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--jev",
            "--vsa",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--jev") && err.contains("--vsa"), "{err}");
    }

    #[test]
    fn jev_mutex_sol() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--sol",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--jev") && err.contains("--sol"), "{err}");
    }

    #[test]
    fn jev_without_steps_is_still_4step_res_multistep() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.steps, JEV_STEPS);
        assert_eq!(plan.sampler, "res_multistep");
    }

    #[test]
    fn jev_steps_20_is_hq_euler_not_remapped_to_4() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--steps",
            "20",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.steps, 20);
        assert_eq!(plan.sampler, "euler");
        assert!(plan.jev);
        assert!(!plan.two_stage);
        assert_ne!(plan.steps, JEV_STEPS);
    }

    #[test]
    fn jev_steps_15_and_32_plan_hq_euler() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        for n in ["15", "32"] {
            let args = H3GenerateArgs::try_parse_from([
                "gemmy-h3-generate",
                "--prompt",
                "a red fox trots through fresh snow",
                "--jev",
                "--steps",
                n,
                "--dry-run-plan",
            ])
            .unwrap();
            let plan = build_plan(&args).unwrap();
            assert_eq!(plan.steps, n.parse::<u32>().unwrap(), "steps {n}");
            assert_eq!(plan.sampler, "euler", "sampler {n}");
        }
    }

    #[test]
    fn jev_fl2va_last_steps_20_is_allowed() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.png");
        let last = dir.path().join("last.png");
        std::fs::write(&first, b"x").unwrap();
        std::fs::write(&last, b"y").unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "fl2va",
            "--first-frame",
            first.to_str().unwrap(),
            "--last-frame",
            last.to_str().unwrap(),
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--steps",
            "20",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.mode, "fl2va");
        assert_eq!(plan.steps, 20);
        assert_eq!(plan.sampler, "euler");
        assert!(plan.last_frame.is_some());
        assert!(!plan.two_stage);
    }

    #[test]
    fn jev_fl2va_last_without_steps_still_rejected() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.png");
        let last = dir.path().join("last.png");
        std::fs::write(&first, b"x").unwrap();
        std::fs::write(&last, b"y").unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "fl2va",
            "--first-frame",
            first.to_str().unwrap(),
            "--last-frame",
            last.to_str().unwrap(),
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(
            err.contains("first+last") || err.contains("last-frame") || err.contains("--last-frame"),
            "{err}"
        );
    }

    #[test]
    fn sla_fixed_5_skips_typesafe_key_and_is_const5() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-fixed",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        assert_eq!(args.sla_fixed, Some(5));
        assert!(!args.jev);
        let plan = build_plan(&args).unwrap();
        assert!(plan.jev, "fixed SLA still uses 009jev packages");
        assert_eq!(plan.jev_initial_policy, "const5");
        assert_eq!(plan.sla_fixed, Some(5));
        assert!(plan.jev_sdk_python.is_empty());
        assert_eq!(plan.steps, JEV_STEPS);
        assert!(!plan.two_stage);
    }

    #[test]
    fn sla_fixed_each_keep_is_const_policy() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        for keep in [1_u8, 3, 5, 10] {
            let args = H3GenerateArgs::try_parse_from([
                "gemmy-h3-generate",
                "--prompt",
                "a red fox trots through fresh snow",
                "--sla-fixed",
                &keep.to_string(),
                "--dry-run-plan",
            ])
            .unwrap();
            let plan = build_plan(&args).unwrap();
            assert_eq!(plan.sla_fixed, Some(keep));
            assert_eq!(plan.jev_initial_policy, format!("const{keep}"));
            assert_eq!(plan.steps, JEV_STEPS);
        }
    }

    #[test]
    fn sla_table_skips_typesafe_key_and_is_table_policy() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let dir = tempfile::tempdir().unwrap();
        let table_path = dir.path().join("table.json");
        let mut table = vec![vec![10_u8; 50]; 4];
        table[1][3] = 5;
        table[2][3] = 3;
        table[3][3] = 1;
        std::fs::write(
            &table_path,
            serde_json::to_vec(&serde_json::json!({ "keep_table": table })).unwrap(),
        )
        .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-table",
            table_path.to_str().unwrap(),
            "--dry-run-plan",
        ])
        .unwrap();
        assert!(args.sla_table.is_some());
        assert!(!args.jev);
        let plan = build_plan(&args).unwrap();
        assert!(plan.jev, "table SLA still uses 009jev packages");
        assert_eq!(plan.jev_initial_policy, "table");
        assert!(plan.sla_fixed.is_none());
        assert_eq!(plan.steps, JEV_STEPS);
        assert!(plan.jev_sdk_python.is_empty());
        assert!(!plan.two_stage);
    }

    #[test]
    fn sla_table_mutex_jev() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let dir = tempfile::tempdir().unwrap();
        let table_path = dir.path().join("table.json");
        std::fs::write(
            &table_path,
            serde_json::to_vec(&serde_json::json!({ "keep_table": vec![vec![10_u8; 50]; 4] }))
                .unwrap(),
        )
        .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--sla-table",
            table_path.to_str().unwrap(),
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--jev") && err.contains("--sla-table"), "{err}");
    }

    #[test]
    fn sla_table_is_not_const10() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let dir = tempfile::tempdir().unwrap();
        let table_path = dir.path().join("table.json");
        let mut table = vec![vec![10_u8; 50]; 4];
        table[2][11] = 5;
        std::fs::write(
            &table_path,
            serde_json::to_vec(&serde_json::json!({ "keep_table": table })).unwrap(),
        )
        .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-table",
            table_path.to_str().unwrap(),
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.jev_initial_policy, "table");
        assert_ne!(plan.jev_initial_policy, "const10");
        assert_eq!(plan.steps, JEV_STEPS);
        assert_eq!(plan.sampler, "res_multistep");
    }

    #[test]
    fn sla_table_20x50_plans_hq_euler() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let dir = tempfile::tempdir().unwrap();
        let table_path = dir.path().join("table20.json");
        let table = vec![vec![10_u8; 50]; 20];
        std::fs::write(
            &table_path,
            serde_json::to_vec(&serde_json::json!({ "keep_table": table })).unwrap(),
        )
        .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-table",
            table_path.to_str().unwrap(),
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.steps, 20);
        assert_eq!(plan.sampler, "euler");
        assert_eq!(plan.jev_initial_policy, "table");
        assert!(!plan.two_stage);
    }

    #[test]
    fn sla_table_4x50_with_steps_20_mismatches() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let dir = tempfile::tempdir().unwrap();
        let table_path = dir.path().join("table.json");
        std::fs::write(
            &table_path,
            serde_json::to_vec(&serde_json::json!({ "keep_table": vec![vec![10_u8; 50]; 4] }))
                .unwrap(),
        )
        .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-table",
            table_path.to_str().unwrap(),
            "--steps",
            "20",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("steps"), "{err}");
    }

    #[test]
    fn sla_fixed_mutex_jev() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--sla-fixed",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--jev") && err.contains("--sla-fixed"), "{err}");
    }

    #[test]
    fn sla_fixed_mutex_vsa() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(None));
        let _guard = JevOverrideGuard;
        let still_dir = tempfile::tempdir().unwrap();
        let still = still_dir.path().join("still.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([200, 40, 40]))
            .save(&still)
            .unwrap();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--mode",
            "ref2va",
            "--ref-image",
            still.to_str().unwrap(),
            "--prompt",
            "walks toward camera",
            "--sla-fixed",
            "3",
            "--vsa",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--sla-fixed") && err.contains("--vsa"), "{err}");
    }

    #[test]
    fn jev_log_dataset_requires_native_sla() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev-log-dataset",
            "datasets/h3_jev_teacher",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--jev-log-dataset"), "{err}");
    }

    #[test]
    fn jev_adaptive_policy_unchanged_with_dataset_flag() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--jev",
            "--jev-log-dataset",
            "datasets/h3_jev_teacher",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_plan(&args).unwrap();
        assert_eq!(plan.jev_initial_policy, JEV_INITIAL_POLICY);
        assert_eq!(plan.steps, JEV_STEPS);
        assert_eq!(plan.jev_sdk_python, py.to_string_lossy());
        assert!(plan.jev_log_dataset.is_some());
        assert!(plan.sla_fixed.is_none());
    }

    #[test]
    fn sla_fixed_rejects_invalid_percent() {
        use clap::Parser;
        let err = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--sla-fixed",
            "7",
            "--dry-run-plan",
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("1") && err.contains("10"), "{err}");
    }

    #[test]
    fn no_sla_plan_is_4step_not_jev_not_euler_default() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--no-sla",
            "--no-compile-ir",
            "--width",
            "864",
            "--height",
            "864",
            "--seed",
            "42",
            "--duration",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        assert!(args.no_sla);
        assert!(!args.jev);
        let plan = build_plan(&args).unwrap();
        assert!(plan.no_sla);
        assert!(!plan.jev);
        assert!(plan.sla_fixed.is_none());
        assert!(plan.jev_sdk_python.is_empty());
        assert_eq!(plan.steps, JEV_STEPS);
        assert!(!plan.compile_ir);
        assert!(!plan.two_stage);
        assert_eq!(plan.mode, "i2v");
        assert_eq!(plan.engine, "comfy");
        assert_eq!(plan.width, 864);
        assert_eq!(plan.height, 864);
        assert_eq!(plan.seed, 42);
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["no_sla"], true);
        assert_eq!(v["jev"], false);
        assert_eq!(v["steps"], 4);
        assert_eq!(v["compile_ir"], false);
    }

    #[test]
    fn no_sla_mutex_jev() {
        use clap::Parser;
        super::super::paths::set_typesafe_api_key_override(Some(Some("test-key")));
        let (_dir, _py, _guard) = jev_sdk_stub();
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--no-sla",
            "--jev",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--no-sla") && err.contains("--jev"), "{err}");
    }

    #[test]
    fn no_sla_mutex_sla_fixed() {
        use clap::Parser;
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--no-sla",
            "--sla-fixed",
            "5",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--no-sla") && err.contains("--sla-fixed"), "{err}");
    }
}
