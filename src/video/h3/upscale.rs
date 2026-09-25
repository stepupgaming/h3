//! `h3 upscale` — RTX / LBH latent refine / Video2X / ffmpeg.

use super::args::{H3Mode, H3UpscaleArgs};
use super::canvas::H3CanvasPreset;
use super::canvas::{resolve_latent_refine_canvas, resolve_upscale_canvas};
use super::paths::{
    abs, checkpoints_root, default_dit_weights_for_mode, h3_python, h3_root,
    latent_upscale_worker_script, resolve_video_vae, upscale_script,
};
use crate::host::gpu::with_gpu_handoff;
use crate::host::config::H3Config;
use crate::host::util::absolute_path;
use crate::host::workers::process::WorkerSpec;
use crate::host::workers::validation::OutputExpectation;
use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Ganloss Eros stage-2: 3D latent enlarge to ~1.0 MP + 3-step ManualSigmas.
/// Landscape 608×352 → 1344×768. Portrait 352×608 → 768×1344.
pub(crate) fn run_eros_stage2(
    input: &Path,
    output: &Path,
    prompt: &str,
    weights: &Path,
    ref_image: Option<&Path>,
    ref_audios: &[PathBuf],
    seed: u64,
    verbose: bool,
    config: &H3Config,
    stage1_width: u32,
    stage1_height: u32,
    vsa: bool,
    vsa_sparsity: f64,
    vsa_gate: &str,
    refmods: &[super::generate::H3RefModUse],
    vae_select: &super::args::H3VaeSelect,
) -> Result<()> {
    let (fw, fh) = H3CanvasPreset::Mp10.size();
    let (tw, th) = if stage1_height > stage1_width {
        (fh, fw)
    } else {
        (fw, fh)
    };
    let mut argv = vec![
        "gemmy-h3-upscale".to_string(),
        "--input".into(),
        input.display().to_string(),
        "--output".into(),
        output.display().to_string(),
        "--backend".into(),
        "latent".into(),
        "--width".into(),
        tw.to_string(),
        "--height".into(),
        th.to_string(),
        "--mode".into(),
        if ref_image.is_some() {
            "ref2va".into()
        } else {
            "t2va".into()
        },
        "--weights".into(),
        weights.display().to_string(),
        "--prompt".into(),
        prompt.to_string(),
        "--seed".into(),
        seed.to_string(),
        "--no-compile-ir".into(),
        "--cache".into(),
        "off".into(),
        "--no-sol".into(),
    ];
    if vsa {
        argv.push("--vsa".into());
        argv.push("--vsa-sparsity".into());
        argv.push(vsa_sparsity.to_string());
        argv.push("--vsa-gate".into());
        argv.push(vsa_gate.to_string());
    }
    if let Some(img) = ref_image {
        argv.push("--ref-image".into());
        argv.push(img.display().to_string());
    }
    for audio in ref_audios {
        argv.push("--ref-audio".into());
        argv.push(audio.display().to_string());
    }
    for m in refmods {
        argv.push("--ref-mod".into());
        argv.push(m.spec());
    }
    vae_select.push_argv(&mut argv);
    if verbose {
        argv.push("-v".into());
    }
    let args = H3UpscaleArgs::try_parse_from(&argv).map_err(|err| anyhow::anyhow!("{err}"))?;
    println!(
        "[h3 shortfilm] Eros stage-2 latent refine {tw}x{th} → {}",
        output.display()
    );
    run_upscale(args, config)
}

const LBH_REFINE_SIGMAS: &str = "0.9035, 0.6316, 0.3158, 0.0000";

#[derive(Debug, Clone, Serialize)]
struct UpscalePlan {
    input: PathBuf,
    output: PathBuf,
    backend: String,
    refine: bool,
    scale: u32,
    canvas: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    prompt: Option<String>,
    mode: Option<String>,
    seed: Option<u64>,
    steps: Option<u32>,
    denoise: Option<f64>,
    refine_sigmas: Option<String>,
    turbo: Option<String>,
    realism: bool,
    sol: bool,
    vsa: bool,
    vsa_sparsity: f64,
    vsa_gate: String,
    cache: Option<String>,
    quality: Option<String>,
    compile_ir: bool,
    first_frame: Option<PathBuf>,
    last_frame: Option<PathBuf>,
    refs: Vec<PathBuf>,
    ref_audios: Vec<PathBuf>,
    refmods: Vec<super::generate::H3RefModUse>,
    weights: Option<PathBuf>,
    rtx_quality: String,
    video2x_model: String,
    audio: Option<PathBuf>,
    check_only: bool,
    h3_root: PathBuf,
    python: PathBuf,
    script: PathBuf,
    video_vae: String,
}

pub(crate) fn run_upscale(args: H3UpscaleArgs, _config: &H3Config) -> Result<()> {
    let plan = build_plan(&args)?;

    if args.dry_run_plan {
        let v = serde_json::to_value(&plan)?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            println!("[h3 upscale] dry-run plan");
            println!("[h3 upscale] input={}", plan.input.display());
            println!("[h3 upscale] output={}", plan.output.display());
            println!(
                "[h3 upscale] backend={} refine={} scale={} rtx_quality={}",
                plan.backend, plan.refine, plan.scale, plan.rtx_quality
            );
            if let Some(c) = &plan.canvas {
                println!("[h3 upscale] canvas={c}");
            }
            if let Some(h) = plan.height {
                println!("[h3 upscale] height={h}");
            }
            if let Some(w) = plan.width {
                println!("[h3 upscale] width={w}");
            }
            if plan.refine {
                println!(
                    "[h3 upscale] latent=refine (LBH enlarge + second H3 sample)"
                );
                if let Some(m) = &plan.mode {
                    println!("[h3 upscale] mode={m}");
                }
                if let Some(s) = &plan.refine_sigmas {
                    println!("[h3 upscale] refine_sigmas={s}");
                }
                if let Some(d) = plan.denoise {
                    println!("[h3 upscale] denoise={d} steps={}", plan.steps.unwrap_or(8));
                }
                if let Some(p) = &plan.prompt {
                    let one: String = p.chars().take(120).collect();
                    println!("[h3 upscale] prompt={one}");
                }
                if let Some(w) = &plan.weights {
                    println!("[h3 upscale] weights={}", w.display());
                }
                println!("[h3 upscale] video_vae={}", plan.video_vae);
            } else if plan.backend == "latent-preview" {
                println!("[h3 upscale] latent=preview (enlarge + decode only)");
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
    if !plan.script.is_file() {
        bail!("upscale script missing: {}", plan.script.display());
    }

    if plan.check_only {
        if args.backend.is_latent() {
            return check_latent_backend(&plan);
        }
        let mut cmd = crate::host::workers::env::native_tool_command(&plan.python);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.current_dir(&plan.h3_root);
        cmd.arg(&plan.script).arg("--check");
        let status = cmd
            .status()
            .with_context(|| format!("run {}", plan.script.display()))?;
        if !status.success() {
            bail!("h3 upscale --check failed (no usable backend)");
        }
        return Ok(());
    }

    if !plan.input.is_file() {
        bail!("input video not found: {}", plan.input.display());
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }

    if let Some(c) = &plan.canvas {
        println!(
            "[h3 upscale] backend={} canvas={} → {}",
            plan.backend,
            c,
            plan.output.display()
        );
    } else {
        println!(
            "[h3 upscale] backend={} scale={} → {}",
            plan.backend,
            plan.scale,
            plan.output.display()
        );
    }
    println!("[h3 upscale] unloading Gemmy before upscale...");

    if args.backend.is_latent() {
        return run_latent_upscale(&args, &plan);
    }

    let verbose = args.verbose;
    let python = plan.python.clone();
    let script = plan.script.clone();
    let h3_root = plan.h3_root.clone();
    let input = plan.input.clone();
    let output = plan.output.clone();
    let audio = plan.audio.clone();
    let backend = plan.backend.clone();
    let rtx_quality = plan.rtx_quality.clone();
    let video2x_model = plan.video2x_model.clone();
    let scale = plan.scale;
    let width = plan.width;
    let height = plan.height;
    let ckpt = checkpoints_root();

    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", ckpt.as_os_str());
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        cmd.current_dir(&h3_root);
        cmd.arg(&script);
        cmd.arg(&input);
        cmd.arg("--out").arg(&output);
        cmd.arg("--backend").arg(&backend);
        cmd.arg("--scale").arg(scale.to_string());
        cmd.arg("--rtx-quality").arg(&rtx_quality);
        cmd.arg("--model").arg(&video2x_model);
        if let Some(w) = width {
            cmd.arg("--width").arg(w.to_string());
        }
        if let Some(h) = height {
            cmd.arg("--height").arg(h.to_string());
        }
        if let Some(a) = &audio {
            cmd.arg("--audio").arg(a);
        }
        WorkerSpec::new(cmd, "video.h3.upscale")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(output.clone()).labeled("H3 upscale mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    println!("wrote H3 upscale video: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "engine": "h3",
                "command": "upscale",
                "plan": plan,
                "output": plan.output,
            }))?
        );
    }
    Ok(())
}

fn resolve_prompt(args: &H3UpscaleArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        let path = abs(path)?;
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()))?;
        return Ok(text);
    }
    Ok(args.prompt.clone().unwrap_or_default())
}

fn build_plan(args: &H3UpscaleArgs) -> Result<UpscalePlan> {
    let check_only = args.check;
    let input = if check_only {
        PathBuf::from(".")
    } else {
        let Some(p) = &args.input else {
            bail!("input video required (or pass --check)");
        };
        let p = abs(p)?;
        if !args.dry_run_plan && !p.is_file() {
            bail!("input video not found: {}", p.display());
        }
        p
    };

    let refine = args.backend.refine();
    let (out_w, out_h, canvas_name) = if refine {
        let c = resolve_latent_refine_canvas(args.canvas, args.width, args.height)?;
        (
            Some(c.width),
            Some(c.height),
            c.preset.map(|p| p.as_str().to_string()),
        )
    } else {
        match resolve_upscale_canvas(args.canvas, args.width, args.height)? {
            Some(c) => {
                let w = if c.width > 0 { Some(c.width) } else { None };
                let h = if c.height > 0 { Some(c.height) } else { None };
                (w, h, c.preset.map(|p| p.as_str().to_string()))
            }
            None => (None, None, None),
        }
    };

    let output = match &args.output {
        Some(p) => absolute_path(p)?,
        None if check_only => PathBuf::from("."),
        None => {
            let tag = if let Some(name) = &canvas_name {
                name.clone()
            } else if let Some(h) = out_h {
                format!("{h}p")
            } else if let Some(w) = out_w {
                format!("{w}w")
            } else {
                format!("x{}", args.scale)
            };
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            let parent = input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            absolute_path(&parent.join(format!("{stem}_up_{tag}.mp4")))?
        }
    };

    if args.scale == 0 || args.scale > 8 {
        bail!("--scale must be in 1..=8");
    }

    let audio = match &args.audio {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() {
                bail!("--audio not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };

    let prompt = if refine && !check_only {
        let text = resolve_prompt(args)?;
        if text.trim().is_empty() {
            bail!(
                "--backend latent refine needs --prompt (same scene as the source clip).\n\
                 Use --backend latent-preview to enlarge and decode without a second H3 pass."
            );
        }
        Some(text)
    } else {
        None
    };

    let first_frame = match &args.first_frame {
        Some(p) => Some(abs(p)?),
        None => None,
    };
    let last_frame = match &args.last_frame {
        Some(p) => Some(abs(p)?),
        None => None,
    };
    if args.mode == H3Mode::Fl2va && refine && (first_frame.is_none() || last_frame.is_none()) {
        bail!("--mode fl2va refine needs both --first-frame and --last-frame");
    }

    let compile_ir = args.compile_ir && !args.no_compile_ir;
    let sol = args.sol && !args.no_sol;
    if refine && args.vsa && sol {
        bail!("--vsa cannot stack with --sol (Saganaki Sol-Attn). Pick one attention owner.");
    }
    if refine && args.vsa && args.cache != super::args::H3Cache::Off {
        bail!("--vsa cannot stack with --cache. Leave cache off.");
    }
    if refine && args.vsa && !args.ref_mod.is_empty() {
        bail!("--vsa cannot stack with --ref-mod (no compiled combo package)");
    }
    if refine && args.vsa && !(0.0..1.0).contains(&args.vsa_sparsity) {
        bail!(
            "--vsa-sparsity must be in [0, 1) (got {})",
            args.vsa_sparsity
        );
    }
    let weights = if refine {
        Some(match &args.weights {
            Some(p) => abs(p)?,
            None => default_dit_weights_for_mode(args.mode),
        })
    } else {
        None
    };

    let (refine_sigmas, denoise) = if refine {
        if let Some(d) = args.denoise {
            if !(0.0..=1.0).contains(&d) {
                bail!("--denoise must be in 0..=1");
            }
            (None, Some(d))
        } else {
            (
                Some(
                    args.refine_sigmas
                        .clone()
                        .unwrap_or_else(|| LBH_REFINE_SIGMAS.to_string()),
                ),
                None,
            )
        }
    } else {
        (None, None)
    };

    Ok(UpscalePlan {
        input,
        output,
        backend: args.backend.as_str().to_string(),
        refine,
        scale: args.scale,
        canvas: canvas_name,
        width: out_w,
        height: out_h,
        prompt,
        mode: refine.then(|| args.mode.as_str().to_string()),
        seed: refine.then_some(args.seed),
        steps: refine.then_some(args.steps),
        denoise,
        refine_sigmas,
        turbo: refine.then(|| args.turbo.as_str().to_string()),
        realism: refine && args.realism,
        sol: refine && sol,
        vsa: refine && args.vsa,
        vsa_sparsity: args.vsa_sparsity,
        vsa_gate: args.vsa_gate.clone(),
        cache: refine.then(|| args.cache.as_str().to_string()),
        quality: refine.then(|| args.quality.as_str().to_string()),
        compile_ir: refine && compile_ir,
        first_frame,
        last_frame,
        refs: {
            let mut out = Vec::new();
            for p in &args.ref_image {
                let p = abs(p)?;
                if !check_only && !args.dry_run_plan && !p.is_file() {
                    bail!("--ref-image not found: {}", p.display());
                }
                out.push(p);
            }
            out
        },
        ref_audios: {
            let mut out = Vec::new();
            for p in &args.ref_audio {
                let p = abs(p)?;
                if !check_only && !args.dry_run_plan && !p.is_file() {
                    bail!("--ref-audio not found: {}", p.display());
                }
                out.push(p);
            }
            out
        },
        refmods: super::generate::parse_ref_mods(&args.ref_mod)?,
        weights,
        rtx_quality: args.rtx_quality.clone(),
        video2x_model: args.video2x_model.clone(),
        audio,
        check_only,
        h3_root: h3_root(),
        python: h3_python(),
        script: if args.backend.is_latent() {
            latent_upscale_worker_script()
        } else {
            upscale_script()
        },
        video_vae: resolve_video_vae(
            &args.vae_select,
            None,
            false,
            args.vae_select.video_vae.is_some() || !args.dry_run_plan,
        )?
        .filename,
    })
}

fn latent_pack_dirs(h3_root: &std::path::Path) -> Vec<PathBuf> {
    let nodes = h3_root.join("ComfyUI").join("custom_nodes");
    vec![
        nodes.join("Comfyui_Minimax_h3_latent_Upscaler"),
        nodes.join("ComfyUI_Minimax_h3_latent_Upscaler"),
        nodes.join("Comfyui-Minimax-h3-latent-Upscaler"),
    ]
}

fn check_latent_backend(plan: &UpscalePlan) -> Result<()> {
    let found = latent_pack_dirs(&plan.h3_root)
        .into_iter()
        .find(|p| p.is_dir());
    match found {
        Some(p) => {
            println!("[h3 upscale] latent pack present: {}", p.display());
            Ok(())
        }
        None => bail!(
            "latent upscaler pack missing under {}\\ComfyUI\\custom_nodes\\Comfyui_Minimax_h3_latent_Upscaler",
            plan.h3_root.display()
        ),
    }
}

fn run_latent_upscale(args: &H3UpscaleArgs, plan: &UpscalePlan) -> Result<()> {
    if !plan.input.is_file() {
        bail!("input video not found: {}", plan.input.display());
    }
    if let Some(parent) = plan.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let work = plan
        .output
        .parent()
        .map(|p| p.join(".h3_latent_up"))
        .unwrap_or_else(|| PathBuf::from(".h3_latent_up"));
    std::fs::create_dir_all(&work)?;
    let attn = if plan.refine {
        args.quality.attn()
    } else {
        "sage"
    };
    let mut refs_json = plan
        .refs
        .iter()
        .map(|p| json!({"kind": "image", "path": p}))
        .collect::<Vec<_>>();
    refs_json.extend(
        plan.ref_audios
            .iter()
            .map(|p| json!({"kind": "audio", "path": p})),
    );
    let request = json!({
        "input": plan.input,
        "output": plan.output,
        "h3_root": plan.h3_root,
        "checkpoints_root": checkpoints_root(),
        "python": plan.python,
        "scale": plan.scale,
        "width": plan.width,
        "height": plan.height,
        "work_dir": work,
        "refine": plan.refine,
        "prompt": plan.prompt,
        "mode": plan.mode,
        "seed": plan.seed,
        "steps": plan.steps,
        "denoise": plan.denoise,
        "refine_sigmas": plan.refine_sigmas,
        "turbo": plan.turbo,
        "turbo_strength": args.turbo_strength,
        "realism": plan.realism,
        "realism_strength": args.realism_strength,
        "sol": plan.sol,
        "vsa": plan.vsa,
        "vsa_sparsity": plan.vsa_sparsity,
        "vsa_gate": plan.vsa_gate,
        "cache": plan.cache,
        "dit_quant": args.dit_quant.as_str(),
        "compile_ir": plan.compile_ir,
        "first_image": plan.first_frame,
        "last_image": plan.last_frame,
        "refs": refs_json,
        "refmods": plan.refmods,
        "weights": plan.weights,
        "video_vae": plan.video_vae,
        "attn": attn,
    });
    let req_path = work.join("request.json");
    std::fs::write(&req_path, serde_json::to_vec_pretty(&request)?)?;

    let verbose = args.verbose;
    let python = plan.python.clone();
    let script = plan.script.clone();
    let h3_root = plan.h3_root.clone();
    let output = plan.output.clone();
    with_gpu_handoff(verbose, || {
        let mut cmd = crate::host::workers::env::native_tool_command(&python);
        crate::host::paths::pin_huggingface_cache(&mut cmd);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("GEMMY_H3_CHECKPOINTS", checkpoints_root().as_os_str());
        super::paths::forward_live_preview(&mut cmd);
        super::paths::forward_extra_dit_roots(&mut cmd);
        super::paths::forward_refmods_dir(&mut cmd);
        cmd.current_dir(&h3_root);
        cmd.arg(&script).arg("--request").arg(&req_path);
        WorkerSpec::new(cmd, "video.h3.upscale.latent")
            .verbose(verbose)
            .expect_outputs(vec![
                OutputExpectation::video(output.clone()).labeled("H3 latent upscale mp4"),
            ])
            .run_inherited()
            .map(|_| ())
    })?;

    println!("wrote H3 latent upscale video: {}", plan.output.display());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "engine": "h3",
                "command": "upscale",
                "backend": plan.backend,
                "refine": plan.refine,
                "plan": plan,
                "output": plan.output,
            }))?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> H3UpscaleArgs {
        H3UpscaleArgs::try_parse_from(
            std::iter::once("upscale").chain(args.iter().copied()),
        )
        .expect("parse")
    }

    #[test]
    fn latent_refine_plan_defaults_720p_and_lbh_sigmas() {
        let args = parse(&[
            "--input",
            "clip.mp4",
            "--backend",
            "latent",
            "--prompt",
            "the courier keeps walking",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).expect("plan");
        assert!(plan.refine);
        assert_eq!(plan.backend, "latent");
        assert_eq!(plan.width, Some(1280));
        assert_eq!(plan.height, Some(736));
        assert_eq!(plan.canvas.as_deref(), Some("720p"));
        assert_eq!(plan.refine_sigmas.as_deref(), Some(LBH_REFINE_SIGMAS));
        assert_eq!(plan.mode.as_deref(), Some("t2va"));
        assert!(plan.prompt.as_deref().unwrap().contains("courier"));
    }

    #[test]
    fn latent_refine_1080p_fails() {
        let args = parse(&[
            "--input",
            "clip.mp4",
            "--backend",
            "latent",
            "--canvas",
            "1080p",
            "--prompt",
            "walk",
            "--dry-run-plan",
        ]);
        assert!(build_plan(&args).is_err());
    }

    #[test]
    fn latent_preview_allows_1080p() {
        let args = parse(&[
            "--input",
            "clip.mp4",
            "--backend",
            "latent-preview",
            "--canvas",
            "1080p",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).expect("plan");
        assert!(!plan.refine);
        assert_eq!(plan.backend, "latent-preview");
        assert_eq!(plan.width, Some(1920));
        assert_eq!(plan.height, Some(1088));
    }

    #[test]
    fn latent_refine_ganloss_1mp_with_ref() {
        let args = parse(&[
            "--input",
            "clip.mp4",
            "--backend",
            "latent",
            "--canvas",
            "1.0",
            "--mode",
            "ref2va",
            "--ref-image",
            "board.png",
            "--prompt",
            "subject_definitions:",
            "--no-compile-ir",
            "--dry-run-plan",
        ]);
        let plan = build_plan(&args).expect("plan");
        assert!(plan.refine);
        assert_eq!(plan.width, Some(1344));
        assert_eq!(plan.height, Some(768));
        assert_eq!(plan.mode.as_deref(), Some("ref2va"));
        assert_eq!(plan.refs.len(), 1);
        assert!(!plan.compile_ir);
    }

    #[test]
    fn latent_refine_requires_prompt() {
        let args = parse(&[
            "--input",
            "clip.mp4",
            "--backend",
            "latent",
            "--dry-run-plan",
        ]);
        let err = build_plan(&args).unwrap_err().to_string();
        assert!(err.contains("--prompt"), "{err}");
    }
}
