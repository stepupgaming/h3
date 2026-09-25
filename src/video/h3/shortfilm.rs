//! Ganloss 6-grid Ref2VA Eros pipeline (gate).
//!
//! Video matches `2026-08-26 minimax_h3_r2v_story_board.json`: one 6-panel
//! storyboard as the sole `MiniMaxH3ReferenceToVideo` Picture, Eros TURBO
//! (INT8 ConvRot on 16 GB), euler / simple / 8 / SigmaShift 12/3 / cache off.
//! Do not pass a character sheet as a second DiT ref — that split-screens.
//! `--scenes N` (N>1): pack path — Krea identity-edit stills, one Picture per
//! scene, hard-cut `h3 loop`. Undersized character/beat sheets
//! SeedVR2 first. No Studio preview. Quality lock (2026-08-27):
//! `outputs/locked_h3_shortfilm_parlor_20260827/` — one still, one 10s take.

use super::args::{H3GenerateArgs, H3LoopArgs, H3ShortfilmArgs, ShortfilmPhase};
use super::canvas::H3CanvasPreset;
use super::generate::build_plan as build_h3_generate_plan;
use super::paths::{
    abs, eros_checkpoints_root, eros_dit_dir, eros_ref2va_bf16_path, eros_ref2va_int8_path,
    EROS_BF16_REPO, EROS_INT8_REPO, EROS_REF2VA_BF16, EROS_REF2VA_INT8,
};
use crate::host::config::H3Config;
use crate::host::paths::default_output_path;
use crate::host::util::{absolute_path, timestamp_slug};
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::{Rgb, RgbImage};
use serde::Serialize;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const TAG_HERO: &str = "hero_face";
const TAG_BOARD: &str = "storyboard";
/// Beat / board stills fed to Eros as Picture 1. Pack Comfy EmptySD3 is 768×432
/// for fast 16-panel batches. The pack video (AykQHPVmG1w) and sample panels
/// are much larger (samples are 1536×1024); ganloss boards are ChatGPT Image.
/// Weak stills cap Ref2VA. 1536×864 is 16:9, 4× that latent, inside Krea 2048.
const PANEL_W: u32 = 1536;
const PANEL_H: u32 = 864;
/// Pack MultiShot V2 `BasicScheduler` steps widget (not Eros 8-step video).
const STILL_STEPS: u32 = 12;
/// Character sheet is the identity-edit source. Short-edge below this is
/// SeedVR2'd before Krea. A mushy or 512px sheet caps every beat.
const SHEET_MIN_SHORT_EDGE: u32 = 1536;
/// Per-scene Picture 1 / identity-edit panel. Pack 768×432 is below this.
const STILL_MIN_SHORT_EDGE: u32 = 864;
/// Ganloss 6-grid as one Picture 1 — each cell needs pixels after encode.
const BOARD_MIN_SHORT_EDGE: u32 = 1536;

/// ganloss `2026-08-26 minimax_h3_r2v_story_board.json`: UNET
/// `10Eros_Max_h3_TURBO_ref2va_beta2.safetensors`, `KSamplerSelect` euler,
/// `BasicScheduler` simple / 8 / denoise 1, `MiniMaxH3SigmaShift` 12/3, no cache.
/// Pack JSON is stock FL2VA (per-shot 15, Spectrum+EasyCache, turbo 4-step LoRA)
/// — that recipe is not loaded on this DiT.
const GANLOSS_EROS_STEPS: u32 = 8;
const GANLOSS_SHIFT_VIDEO: f64 = 12.0;
const GANLOSS_SHIFT_AUDIO: f64 = 3.0;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShortfilmPlan {
    pub pipeline: String,
    pub phase: String,
    pub grid: u32,
    pub scenes: u32,
    /// `ganloss-one-window` (default) or `pack-per-scene` (`--scenes N>1`).
    pub layout: String,
    pub stills: ShortfilmStillsPlan,
    pub video: ShortfilmVideoPlan,
    pub loop_plan: Value,
    pub download: ErosDownloadPlan,
    pub output: PathBuf,
    pub out_dir: PathBuf,
    pub plan_out: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShortfilmStillsPlan {
    pub character_sheet: Option<PathBuf>,
    pub identity_edit: bool,
    pub identity_lora: String,
    pub grid: u32,
    pub cols: u32,
    pub rows: u32,
    pub panel_size: [u32; 2],
    pub panel_prompts: Vec<String>,
    pub panels: Vec<PathBuf>,
    pub storyboard: Option<PathBuf>,
    pub board_is_first_frame: bool,
    pub skip_stills: bool,
    pub identity_edit_steps: u32,
    pub sheet_min_short_edge: u32,
    pub still_min_short_edge: u32,
    pub board_min_short_edge: u32,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShortfilmVideoPlan {
    pub mode: String,
    pub cache: String,
    pub spectrum: bool,
    pub fbc: bool,
    pub sol: bool,
    pub compile_ir: bool,
    pub sampler: String,
    pub scheduler: String,
    pub steps: u32,
    pub shift_video: f64,
    pub shift_audio: f64,
    pub prompt_ir: String,
    pub weights: PathBuf,
    pub weights_named_bf16: PathBuf,
    pub character_sheet: Option<PathBuf>,
    pub storyboard: Option<PathBuf>,
    pub refs: Vec<ShortfilmRef>,
    pub first_frame: Option<PathBuf>,
    pub duration_s: f64,
    pub seed: u64,
    /// ganloss Eros two-stage (video LByGCGzu67o): 0.2 MP sample then 1.0 MP refine.
    pub two_stage: bool,
    pub stage1: [u32; 2],
    pub stage2: [u32; 2],
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ShortfilmRef {
    pub tag: String,
    pub kind: String,
    pub path: PathBuf,
    pub role: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ErosDownloadPlan {
    pub dest_root: PathBuf,
    pub named: ErosFilePlan,
    pub int8: ErosFilePlan,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ErosFilePlan {
    pub repo: String,
    pub filename: String,
    pub url: String,
    pub dest: PathBuf,
    pub present: bool,
    pub bytes: Option<u64>,
}

pub(crate) fn run_shortfilm(args: H3ShortfilmArgs, _config: &H3Config) -> Result<()> {
    let plan = build_shortfilm_plan(&args)?;

    if args.dry_run_plan {
        print_dry_run(&plan, args.json)?;
        return Ok(());
    }

    if args.phase == ShortfilmPhase::Download {
        println!(
            "[h3 shortfilm] downloading Eros weights under {}",
            plan.download.dest_root.display()
        );
        download_eros_weights(&plan.download, args.verbose)?;
        return Ok(());
    }

    std::fs::create_dir_all(&plan.out_dir)
        .with_context(|| format!("create {}", plan.out_dir.display()))?;
    std::fs::write(
        &plan.plan_out,
        serde_json::to_vec_pretty(&plan.loop_plan)?,
    )
    .with_context(|| format!("write {}", plan.plan_out.display()))?;

    let mut board = plan.stills.storyboard.clone();

    if matches!(args.phase, ShortfilmPhase::All | ShortfilmPhase::Stills)
        && !plan.stills.skip_stills
    {
        let gemmy = crate::host::gemmy::require_gemmy(
            "shortfilm stills use `gemmy image` for the Krea identity-edit",
        )?;
        board = Some(run_stills_phase(&args, &plan, &gemmy)?);
    }

    if matches!(args.phase, ShortfilmPhase::Stills) {
        println!(
            "[h3 shortfilm] stills done. board={}",
            board
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        );
        return Ok(());
    }

    if plan.stills.character_sheet.is_some() {
        eprintln!(
            "[h3 shortfilm] --character-sheet is stills-only (not a DiT ref)"
        );
    }

    if args.scenes > 1 {
        let boards = scene_boards_on_disk(&plan)?;
        if boards.len() < args.scenes as usize {
            bail!(
                "--scenes {} needs that many stills (got {}). Pass --panel or run stills from --character-sheet.",
                args.scenes,
                boards.len()
            );
        }
        let hq = ensure_hq_panel_list(&boards, &plan.out_dir, args.verbose)?;
        let mut loop_plan = plan.loop_plan.clone();
        patch_loop_plan_still_paths(&mut loop_plan, &hq);
        std::fs::write(
            &plan.plan_out,
            serde_json::to_vec_pretty(&loop_plan)?,
        )
        .with_context(|| format!("rewrite {}", plan.plan_out.display()))?;
        println!(
            "[h3 shortfilm] pack director loop + Eros two-stage: {} scenes, one Picture each, continuation=none, {}x{} then {}x{}",
            args.scenes,
            plan.video.stage1[0],
            plan.video.stage1[1],
            plan.video.stage2[0],
            plan.video.stage2[1]
        );
        return dispatch_loop(&plan.plan_out, &plan.output, args.verbose, args.json, _config);
    }

    let board = board.context(
        "one-window video needs --storyboard, or --panel stills to compose a 6-grid",
    )?;
    if !board.is_file() && !plan.stills.panels.is_empty() {
        let (cols, rows) = grid_shape(plan.grid);
        compose_plain_board(&plan.stills.panels, cols, rows, &board)?;
        println!(
            "[h3 shortfilm] composed unlabeled {}×{} ganloss board → {}",
            cols,
            rows,
            board.display()
        );
    }
    if !board.is_file() {
        bail!("storyboard not found: {}", board.display());
    }
    refuse_board_as_first_frame(&board)?;
    let board = ensure_hq_still(
        &board,
        &plan.out_dir.join("scene_board_hq.png"),
        BOARD_MIN_SHORT_EDGE,
        args.verbose,
        "storyboard",
    )?;

    println!(
        "[h3 shortfilm] ganloss one-window: 1 ref-image (6-panel Picture 1) euler/simple steps={} shift={}/{} cache=off SplitSigmas@4 then fresh 3D+3-step {}x{}→{}x{}",
        args.steps,
        GANLOSS_SHIFT_VIDEO,
        GANLOSS_SHIFT_AUDIO,
        plan.video.stage1[0],
        plan.video.stage1[1],
        plan.video.stage2[0],
        plan.video.stage2[1]
    );
    let stage1_mp4 = plan.out_dir.join("stage1_0p2.mp4");
    let gen_args = generate_args_for_shortfilm(
        &plan.video.prompt_ir,
        &board,
        &plan.video.weights,
        &stage1_mp4,
        &args,
        false,
    )?;
    super::generate::run_generate(gen_args, _config)?;
    super::upscale::run_eros_stage2(
        &stage1_mp4,
        &plan.output,
        &plan.video.prompt_ir,
        &plan.video.weights,
        Some(&board),
        &[],
        args.seed,
        args.verbose,
        _config,
        608,
        352,
        args.vsa,
        args.vsa_sparsity,
        &args.vsa_gate,
        &[],
        &args.vae_select,
    )
}

fn scene_boards_on_disk(plan: &ShortfilmPlan) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in &plan.stills.panels {
        if p.is_file() {
            out.push(p.clone());
        }
    }
    Ok(out)
}

fn still_short_edge(width: u32, height: u32) -> u32 {
    width.min(height)
}

fn still_needs_upscale(width: u32, height: u32, min_short_edge: u32) -> bool {
    still_short_edge(width, height) < min_short_edge
}

fn read_still_size(path: &Path) -> Result<(u32, u32)> {
    image::image_dimensions(path).with_context(|| format!("read image size {}", path.display()))
}

fn print_still_size_note(path: &Path, min_short_edge: u32, label: &str) {
    if !path.is_file() {
        return;
    }
    let Ok((w, h)) = read_still_size(path) else {
        return;
    };
    if still_needs_upscale(w, h, min_short_edge) {
        println!(
            "[h3 shortfilm] {label} {}x{} short-edge {} < {min_short_edge} → SeedVR2 before Eros",
            w,
            h,
            still_short_edge(w, h)
        );
    } else {
        println!(
            "[h3 shortfilm] {label} {}x{} short-edge {} (ok)",
            w,
            h,
            still_short_edge(w, h)
        );
    }
}

fn patch_loop_plan_still_paths(loop_plan: &mut Value, boards: &[PathBuf]) {
    let Some(scenes) = loop_plan.get_mut("scenes").and_then(|s| s.as_array_mut()) else {
        return;
    };
    for (i, scene) in scenes.iter_mut().enumerate() {
        let Some(path) = boards.get(i) else {
            continue;
        };
        scene["tags"][TAG_BOARD]["path"] = json!(path);
    }
}

fn ensure_hq_panel_list(
    panels: &[PathBuf],
    out_dir: &Path,
    verbose: bool,
) -> Result<Vec<PathBuf>> {
    let mut out = Vec::with_capacity(panels.len());
    for (i, src) in panels.iter().enumerate() {
        let dest = out_dir.join(format!("panel_{}_hq.png", i + 1));
        out.push(ensure_hq_still(
            src,
            &dest,
            STILL_MIN_SHORT_EDGE,
            verbose,
            &format!("beat still {}", i + 1),
        )?);
    }
    Ok(out)
}

fn ensure_hq_still(
    src: &Path,
    dest: &Path,
    min_short_edge: u32,
    verbose: bool,
    label: &str,
) -> Result<PathBuf> {
    if !src.is_file() {
        bail!("{label} not found: {}", src.display());
    }
    let (w, h) = read_still_size(src)?;
    if !still_needs_upscale(w, h, min_short_edge) {
        println!(
            "[h3 shortfilm] {label} {}x{} (short-edge {} >= {min_short_edge}, keep)",
            w,
            h,
            still_short_edge(w, h)
        );
        return Ok(src.to_path_buf());
    }
    if dest == src {
        bail!(
            "SeedVR2 refuses to overwrite {label} {}",
            src.display()
        );
    }
    if dest.is_file() {
        if let Ok((dw, dh)) = read_still_size(dest) {
            if !still_needs_upscale(dw, dh, min_short_edge) {
                println!(
                    "[h3 shortfilm] {label} reusing {} ({}x{})",
                    dest.display(),
                    dw,
                    dh
                );
                return Ok(dest.to_path_buf());
            }
        }
    }
    println!(
        "[h3 shortfilm] {label} {}x{} is too small for Eros (short-edge {} < {min_short_edge}). SeedVR2 → {}",
        w,
        h,
        still_short_edge(w, h),
        dest.display()
    );
    let gemmy = crate::host::gemmy::require_gemmy(
        "this still is below the Eros short-edge and needs `gemmy image upscale` (SeedVR2)",
    )?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut cmd = crate::host::workers::env::gemmy_child_command(&gemmy);
    cmd.arg("image")
        .arg("upscale")
        .arg("--input")
        .arg(src)
        .arg("--output")
        .arg(dest)
        .arg("--resolution")
        .arg(min_short_edge.to_string())
        .arg("--overwrite");
    if verbose {
        cmd.arg("-v");
    }
    crate::host::util::run_simple_command(cmd, "h3 shortfilm seedvr2 still", verbose)
        .with_context(|| format!("SeedVR2 {label} {}", src.display()))?;
    if !dest.is_file() {
        bail!("SeedVR2 did not write {}", dest.display());
    }
    let (ow, oh) = read_still_size(dest)?;
    if still_needs_upscale(ow, oh, min_short_edge) {
        bail!(
            "SeedVR2 {label} still too small: {}x{} (short-edge {} < {min_short_edge})",
            ow,
            oh,
            still_short_edge(ow, oh)
        );
    }
    println!(
        "[h3 shortfilm] {label} SeedVR2 {}x{} → {}x{}",
        w, h, ow, oh
    );
    Ok(dest.to_path_buf())
}

fn dispatch_loop(
    plan_out: &Path,
    output: &Path,
    verbose: bool,
    json_out: bool,
    config: &H3Config,
) -> Result<()> {
    let mut argv = vec![
        "gemmy-h3-loop".to_string(),
        "--plan".into(),
        plan_out.display().to_string(),
        "--output".into(),
        output.display().to_string(),
    ];
    if verbose {
        argv.push("-v".into());
    }
    if json_out {
        argv.push("--json".into());
    }
    let loop_args = H3LoopArgs::try_parse_from(&argv).map_err(|err| anyhow::anyhow!("{err}"))?;
    super::loop_cmd::run_loop(loop_args, config)
}

pub(crate) fn build_shortfilm_plan(args: &H3ShortfilmArgs) -> Result<ShortfilmPlan> {
    if !matches!(args.grid, 6 | 9 | 16) {
        bail!("--grid must be 6 (3×2), 9 (3×3), or 16 (4×4)");
    }
    if args.scenes == 0 || args.scenes > 64 {
        bail!("--scenes must be in 1..=64");
    }
    if args.steps == 0 || args.steps > 100 {
        bail!("--steps must be in 1..=100 (ganloss Eros graph is 8)");
    }
    if args.duration.is_finite() && args.duration <= 0.0 && args.phase != ShortfilmPhase::Download
    {
        bail!("--duration must be > 0");
    }

    let download = eros_download_plan();
    let (cols, rows) = grid_shape(args.grid);

    let out_dir = absolute_path(&args.out_dir.clone().unwrap_or_else(|| {
        default_output_path(format!("h3_shortfilm_{}", timestamp_slug()))
    }))?;
    let output = absolute_path(&args.output.clone().unwrap_or_else(|| {
        out_dir.join("shortfilm.mp4")
    }))?;
    let plan_out = absolute_path(&args.plan_out.clone().unwrap_or_else(|| {
        out_dir.join("plan.json")
    }))?;

    if args.phase == ShortfilmPhase::Download {
        let empty_video = ShortfilmVideoPlan {
            mode: "ref2va".into(),
            cache: "off".into(),
            spectrum: false,
            fbc: false,
            sol: false,
            compile_ir: false,
            sampler: "euler".into(),
            scheduler: "simple".into(),
            steps: args.steps,
            shift_video: GANLOSS_SHIFT_VIDEO,
            shift_audio: GANLOSS_SHIFT_AUDIO,
            prompt_ir: String::new(),
            weights: download.int8.dest.clone(),
            weights_named_bf16: download.named.dest.clone(),
            character_sheet: None,
            storyboard: None,
            refs: Vec::new(),
            first_frame: None,
            duration_s: args.duration,
            seed: args.seed,
            two_stage: true,
            stage1: {
                let (w, h) = H3CanvasPreset::Mp02.size();
                [w, h]
            },
            stage2: {
                let (w, h) = H3CanvasPreset::Mp10.size();
                [w, h]
            },
        };
        return Ok(ShortfilmPlan {
            pipeline: "shortfilm-director-ref2va".into(),
            phase: args.phase.as_str().into(),
            grid: args.grid,
            scenes: args.scenes.max(1),
            layout: if args.scenes > 1 {
                "pack-per-scene".into()
            } else {
                "ganloss-one-window".into()
            },
            stills: ShortfilmStillsPlan {
                character_sheet: None,
                identity_edit: true,
                identity_lora: "krea2_identity_edit_v1_2.safetensors".into(),
                grid: args.grid,
                cols,
                rows,
                panel_size: [PANEL_W, PANEL_H],
                panel_prompts: Vec::new(),
                panels: Vec::new(),
                storyboard: None,
                board_is_first_frame: false,
                skip_stills: true,
                identity_edit_steps: STILL_STEPS,
                sheet_min_short_edge: SHEET_MIN_SHORT_EDGE,
                still_min_short_edge: STILL_MIN_SHORT_EDGE,
                board_min_short_edge: BOARD_MIN_SHORT_EDGE,
            },
            video: empty_video,
            loop_plan: json!({"schema": "gemmy-h3-loop-v1", "scenes": []}),
            download,
            output,
            out_dir,
            plan_out,
        });
    }

    let sheet = match &args.character_sheet {
        Some(p) => {
            let p = abs(p)?;
            if !p.is_file() && !args.dry_run_plan {
                bail!("--character-sheet not found: {}", p.display());
            }
            Some(p)
        }
        None => None,
    };
    if matches!(args.phase, ShortfilmPhase::Stills)
        && sheet.is_none()
        && args.panel.is_empty()
    {
        bail!("--phase stills needs --character-sheet or --panel");
    }

    let storyboard = match &args.storyboard {
        Some(p) => Some(abs(p)?),
        None => None,
    };
    if let Some(board) = &storyboard {
        refuse_board_as_first_frame(board)?;
        if !board.is_file() && !args.dry_run_plan && args.phase == ShortfilmPhase::Video {
            bail!("--storyboard not found: {}", board.display());
        }
    }

    let mut panels = Vec::new();
    for p in &args.panel {
        let p = abs(p)?;
        if !p.is_file() && !args.dry_run_plan {
            bail!("--panel not found: {}", p.display());
        }
        panels.push(p);
    }

    let per_scene = args.scenes > 1;
    let n_stills = if per_scene { args.scenes } else { args.grid };
    let skip_stills = if per_scene {
        args.phase == ShortfilmPhase::Video || panels.len() as u32 >= args.scenes
    } else {
        storyboard.is_some()
            || args.phase == ShortfilmPhase::Video
            || (!panels.is_empty() && sheet.is_none())
    };

    let brief = resolve_prompt(args)?;
    if args.phase != ShortfilmPhase::Stills && brief.trim().is_empty() && args.panel_prompt.is_empty()
    {
        bail!("--prompt or --prompt-file is required for the video phase");
    }
    if per_scene && storyboard.is_some() && panels.is_empty() && sheet.is_none() {
        bail!(
            "--scenes {} is one still per scene. Do not pass a composite --storyboard; pass --panel x {} or --character-sheet.",
            args.scenes,
            args.scenes
        );
    }
    if matches!(args.phase, ShortfilmPhase::Video | ShortfilmPhase::All)
        && !per_scene
        && storyboard.is_none()
        && panels.is_empty()
        && !args.dry_run_plan
    {
        bail!("one-window video needs --storyboard or --panel stills to compose a 6-grid");
    }
    if per_scene
        && matches!(args.phase, ShortfilmPhase::Video | ShortfilmPhase::All)
        && panels.is_empty()
        && sheet.is_none()
        && !args.dry_run_plan
    {
        bail!("--scenes {} needs --panel stills or --character-sheet", args.scenes);
    }

    let panel_prompts = expand_panel_prompts(n_stills, &args.panel_prompt, &brief);
    let planned_board = storyboard.clone().unwrap_or_else(|| out_dir.join("scene_board.png"));
    let planned_panels: Vec<PathBuf> = if panels.is_empty() {
        (1..=n_stills)
            .map(|i| out_dir.join(format!("panel_{i}.png")))
            .collect()
    } else {
        panels.clone()
    };

    let ir = if per_scene {
        compile_scene_ir(&brief, args.duration.max(0.5), 1, args.scenes)
    } else {
        compile_ganloss_ir(&brief, args.duration.max(0.5), args.grid)
    };
    let weights = eros_ref2va_int8_path();
    let named = eros_ref2va_bf16_path();

    let refs: Vec<ShortfilmRef> = if per_scene {
        planned_panels
            .iter()
            .take(args.scenes as usize)
            .enumerate()
            .map(|(i, p)| ShortfilmRef {
                tag: format!("scene_{}", i + 1),
                kind: "image".into(),
                path: p.clone(),
                role: "scene_picture_1".into(),
            })
            .collect()
    } else {
        vec![ShortfilmRef {
            tag: TAG_BOARD.into(),
            kind: "image".into(),
            path: planned_board.clone(),
            role: "ganloss_picture_1".into(),
        }]
    };

    let mut video = ShortfilmVideoPlan {
        mode: "ref2va".into(),
        cache: "off".into(),
        spectrum: false,
        fbc: false,
        sol: false,
        compile_ir: false,
        sampler: "euler".into(),
        scheduler: "simple".into(),
        steps: args.steps,
        shift_video: GANLOSS_SHIFT_VIDEO,
        shift_audio: GANLOSS_SHIFT_AUDIO,
        prompt_ir: ir.clone(),
        weights: weights.clone(),
        weights_named_bf16: named,
        character_sheet: sheet.clone(),
        storyboard: Some(planned_board.clone()),
        refs,
        first_frame: None,
        duration_s: args.duration,
        seed: args.seed,
        two_stage: true,
        stage1: {
            let (w, h) = H3CanvasPreset::Mp02.size();
            [w, h]
        },
        stage2: {
            let (w, h) = H3CanvasPreset::Mp10.size();
            [w, h]
        },
    };

    // Drive the shipped generate planner when the sole storyboard exists on disk.
    if planned_board.is_file() {
        if let Ok(gen_args) = generate_args_for_shortfilm(
            &ir,
            &planned_board,
            &weights,
            &output,
            args,
            true,
        ) {
            if let Ok(gp) = build_h3_generate_plan(&gen_args) {
                video.mode = gp.mode;
                video.cache = gp.cache;
                video.sol = gp.sol;
                video.weights = gp.weights;
                video.compile_ir = gp.compile_ir;
                video.duration_s = gp.duration_s;
                video.steps = gp.steps;
                video.shift_video = gp.shift_video;
                video.shift_audio = gp.shift_audio;
                video.first_frame = gp.first_frame;
            }
        }
    }

    let loop_plan = if per_scene {
        let scene_irs: Vec<String> = (0..args.scenes)
            .map(|i| {
                // Video IR is the film/scene action. `--panel-prompt` is the
                // Krea still wrapper only — stuffing it here made Eros freeze
                // on "photograph" / "still" language.
                let action = if !brief.trim().is_empty() {
                    brief.as_str()
                } else {
                    args.panel_prompt
                        .get(i as usize)
                        .or_else(|| args.panel_prompt.first())
                        .map(|s| s.as_str())
                        .unwrap_or("")
                };
                compile_scene_ir(action, args.duration.max(0.5), i + 1, args.scenes)
            })
            .collect();
        build_per_scene_loop_plan(
            &scene_irs,
            &planned_panels
                .iter()
                .take(args.scenes as usize)
                .cloned()
                .collect::<Vec<_>>(),
            &weights,
            args.duration,
            args.seed,
            args.steps,
        )
    } else {
        build_loop_plan_json(
            &ir,
            None,
            &planned_board,
            &weights,
            args.duration,
            1,
            args.seed,
            args.steps,
        )
    };

    Ok(ShortfilmPlan {
        pipeline: "shortfilm-director-ref2va".into(),
        phase: args.phase.as_str().into(),
        grid: n_stills,
        scenes: args.scenes.max(1),
        layout: if per_scene {
            "pack-per-scene".into()
        } else {
            "ganloss-one-window".into()
        },
        stills: ShortfilmStillsPlan {
            character_sheet: sheet,
            identity_edit: true,
            identity_lora: "krea2_identity_edit_v1_2.safetensors".into(),
            grid: n_stills,
            cols: if per_scene { grid_shape(n_stills).0 } else { cols },
            rows: if per_scene { grid_shape(n_stills).1 } else { rows },
            panel_size: [PANEL_W, PANEL_H],
            panel_prompts,
            panels: planned_panels,
            storyboard: Some(planned_board),
            board_is_first_frame: false,
            skip_stills,
            identity_edit_steps: STILL_STEPS,
            sheet_min_short_edge: SHEET_MIN_SHORT_EDGE,
            still_min_short_edge: STILL_MIN_SHORT_EDGE,
            board_min_short_edge: BOARD_MIN_SHORT_EDGE,
        },
        video,
        loop_plan,
        download,
        output,
        out_dir,
        plan_out,
    })
}

pub(crate) fn eros_download_plan() -> ErosDownloadPlan {
    let named_dest = eros_ref2va_bf16_path();
    let int8_dest = eros_ref2va_int8_path();
    ErosDownloadPlan {
        dest_root: eros_checkpoints_root(),
        named: file_plan(EROS_BF16_REPO, EROS_REF2VA_BF16, named_dest),
        int8: file_plan(EROS_INT8_REPO, EROS_REF2VA_INT8, int8_dest),
    }
}

fn file_plan(repo: &str, filename: &str, dest: PathBuf) -> ErosFilePlan {
    let present = dest.is_file();
    let bytes = if present {
        std::fs::metadata(&dest).ok().map(|m| m.len())
    } else {
        None
    };
    ErosFilePlan {
        repo: repo.into(),
        filename: filename.into(),
        url: hf_resolve_url(repo, filename),
        dest,
        present,
        bytes,
    }
}

fn hf_resolve_url(repo: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}?download=1",
        repo,
        filename.replace('\\', "/")
    )
}

fn grid_shape(grid: u32) -> (u32, u32) {
    match grid {
        6 => (3, 2),
        9 => (3, 3),
        16 => (4, 4),
        n => {
            let cols = (n as f64).sqrt().ceil() as u32;
            let rows = n.div_ceil(cols.max(1));
            (cols.max(1), rows.max(1))
        }
    }
}

fn resolve_prompt(args: &H3ShortfilmArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        let path = abs(path)?;
        return std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()));
    }
    Ok(args.prompt.clone().unwrap_or_default())
}

fn expand_panel_prompts(grid: u32, explicit: &[String], brief: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(grid as usize);
    for i in 0..grid {
        if let Some(p) = explicit.get(i as usize) {
            out.push(p.clone());
        } else if let Some(p) = explicit.first() {
            // Reuse the operator prompt as-is. Do not stamp "shot N of M" —
            // Krea will draw a numbered board.
            out.push(p.clone());
        } else {
            let b = brief.trim();
            let action = if b.is_empty() {
                "photoreal cinematic still of the locked character in one room".to_string()
            } else {
                b.to_string()
            };
            out.push(format!(
                "Keep the same person, face, hair, and outfit as the character sheet. Photoreal cinematic still: {action}. One photograph of one room. Bare walls: no framed art, no pictures hanging, no second landscape inside the room."
            ));
        }
    }
    out
}

/// Ganloss single-window Ref2VA IR. One 6-panel is `<Picture 1>`.
/// Identity lives in those panels. Do not add a second sheet as Picture 2.
pub(crate) fn compile_ganloss_ir(brief: &str, duration_s: f64, n_shots: u32) -> String {
    let n_shots = n_shots.max(1);
    let duration_s = duration_s.max(0.5);
    let brief = brief.trim();
    let intent = {
        let mut t = if brief.is_empty() {
            "the character performs the storyboard action in chronological shot order".to_string()
        } else {
            brief.to_string()
        };
        if !t.ends_with('.') && !t.ends_with('!') && !t.ends_with('?') {
            t.push('.');
        }
        t
    };
    let mut shots = String::new();
    for i in 1..=n_shots {
        let t = if i == 1 {
            0.0
        } else {
            duration_s * (f64::from(i - 1) / f64::from(n_shots))
        };
        let stamp = format_timestamp(t);
        if i == 1 {
            shots.push_str(&format!(
                "[Shot {i}] <Picture 1> establishes <Subject 1> in the first panel (left-to-right, top-to-bottom). {intent} begins here as a single cinematic shot, not as a grid or split screen.\n"
            ));
        } else {
            shots.push_str(&format!(
                "[Shot {i}] At {stamp}, cut to the next panel of <Picture 1>. <Subject 1> stays the same person. Camera height, framing, and action follow that panel's beat.\n"
            ));
        }
    }
    format!(
        "subject_definitions:\n\
- <Subject 1>: The same person shown throughout <Picture 1>, preserving facial identity, body, hair, and outfit in every shot.\n\
- <Picture 1>: The complete {n_shots}-panel photorealistic storyboard, used as the shot-order, composition, action, character, costume, setting, lighting, and cinematic-style reference. Treat each panel as a separate chronological shot beat, not as one composite image.\n\n\
summary:\n\
[reference generation + keyframe completion] Create a {duration_s:.0}-second photoreal cinematic video following the panels of <Picture 1> from left to right and top to bottom. {intent}\n\n\
retention_analysis:\n\
- <Subject 1> — fully_preserved: retain the same face, hair, body, and outfit from <Picture 1> in every shot.\n\
- <Picture 1> — fully_preserved: retain its shot order, camera changes, and action beats. Do not reproduce panel borders, grid lines, shot numbers, captions, or the board as a split-screen composite.\n\n\
detailed_description:\n\
Photoreal live-action cinematography with coherent lighting, stable geometry, and continuous motion. {intent}\n\
{shots}\
Maintain one facial identity and exact wardrobe from <Picture 1>. No storyboard grid, panel borders, numbers, split screen, duplicated extras, subtitles, text, logos, or watermarks.\n\n\
overall_soundscape:\n\
Natural scene ambience locked to visible action: footsteps, cloth, and environment. No narrator unless the brief requires dialogue.\n\n\
non_diegetic_music:\n\
Soft cinematic underscore that follows the shot beats, unobtrusive, dynamics mostly flat with a gentle late settle.\n"
    )
}

pub(crate) use super::scene_ir::compile_scene_ir;

fn format_timestamp(seconds: f64) -> String {
    let ms_total = (seconds.max(0.0) * 1000.0).round() as u64;
    let mm = ms_total / 60_000;
    let ss = (ms_total % 60_000) / 1000;
    let ms = ms_total % 1000;
    format!("{mm:02}:{ss:02}.{ms:03}")
}

pub(crate) fn build_loop_plan_json(
    ir: &str,
    sheet: Option<&Path>,
    board: &Path,
    weights: &Path,
    duration_s: f64,
    scenes: u32,
    seed: u64,
    steps: u32,
) -> Value {
    let scenes_n = scenes.max(1);
    let per = duration_s.max(0.5);
    let mut scene_vals = Vec::new();
    for i in 0..scenes_n {
        let id = format!("scene-{}", i + 1);
        let mut tags = serde_json::Map::new();
        if let Some(sheet) = sheet {
            tags.insert(
                TAG_HERO.into(),
                json!({"kind": "image", "path": sheet}),
            );
        }
        tags.insert(
            TAG_BOARD.into(),
            json!({"kind": "image", "path": board}),
        );
        scene_vals.push(json!({
            "id": id,
            "prompt": ir,
            "mode": "ref2va",
            "duration": per,
            "seed": seed + u64::from(i),
            "continuation_mode": if i == 0 { "none" } else { "masked_av" },
            "tags": tags,
        }));
    }
    json!({
        "schema": "gemmy-h3-loop-v1",
        "mode": "ref2va",
        "weights": weights,
        "cache": "off",
        "steps": steps,
        "shift_video": GANLOSS_SHIFT_VIDEO,
        "shift_audio": GANLOSS_SHIFT_AUDIO,
        "compile_ir": false,
        "canvas": {
            "width": H3CanvasPreset::Mp02.size().0,
            "height": H3CanvasPreset::Mp02.size().1
        },
        "refine_canvas": {
            "width": H3CanvasPreset::Mp10.size().0,
            "height": H3CanvasPreset::Mp10.size().1
        },
        "two_stage": true,
        "audio_mode": "generated_audio",
        "scenes": scene_vals,
    })
}

/// Pack path: one still per scene, hard cuts (not masked_av).
pub(crate) fn build_per_scene_loop_plan(
    scene_irs: &[String],
    boards: &[PathBuf],
    weights: &Path,
    duration_s: f64,
    seed: u64,
    steps: u32,
) -> Value {
    let per = duration_s.max(0.5);
    let n = scene_irs.len();
    let mut scene_vals = Vec::new();
    for i in 0..n {
        let id = format!("scene-{}", i + 1);
        let board = boards.get(i).or_else(|| boards.last());
        let mut tags = serde_json::Map::new();
        if let Some(board) = board {
            tags.insert(
                TAG_BOARD.into(),
                json!({"kind": "image", "path": board}),
            );
        }
        let prompt = scene_irs
            .get(i)
            .cloned()
            .unwrap_or_else(|| scene_irs.first().cloned().unwrap_or_default());
        scene_vals.push(json!({
            "id": id,
            "prompt": prompt,
            "mode": "ref2va",
            "duration": per,
            "seed": seed + i as u64,
            "continuation_mode": "none",
            "tags": tags,
        }));
    }
    json!({
        "schema": "gemmy-h3-loop-v1",
        "mode": "ref2va",
        "weights": weights,
        "cache": "off",
        "steps": steps,
        "shift_video": GANLOSS_SHIFT_VIDEO,
        "shift_audio": GANLOSS_SHIFT_AUDIO,
        "compile_ir": false,
        "audio_mode": "generated_audio",
        "layout": "per-scene",
        "canvas": {
            "width": H3CanvasPreset::Mp02.size().0,
            "height": H3CanvasPreset::Mp02.size().1
        },
        "refine_canvas": {
            "width": H3CanvasPreset::Mp10.size().0,
            "height": H3CanvasPreset::Mp10.size().1
        },
        "two_stage": true,
        "scenes": scene_vals,
    })
}

fn refuse_board_as_first_frame(board: &Path) -> Result<()> {
    // Product rule: the labeled board is a Ref2VA Picture, never I2V --first-frame.
    let name = board
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.contains("first_frame") || name.contains("first-frame") {
        bail!(
            "the labeled scene board must not be used as an H3 first-frame ({})\n\
             This pipeline is Ref2VA: pass the board as @{TAG_BOARD}, not --first-frame.",
            board.display()
        );
    }
    Ok(())
}

fn generate_args_for_shortfilm(
    ir: &str,
    board: &Path,
    weights: &Path,
    output: &Path,
    args: &H3ShortfilmArgs,
    dry_run: bool,
) -> Result<H3GenerateArgs> {
    let mut argv = vec![
        "gemmy-h3-generate".to_string(),
        "--mode".into(),
        "ref2va".into(),
        "--engine".into(),
        "comfy".into(),
        "--cache".into(),
        "off".into(),
        "--no-sol".into(),
        "--no-compile-ir".into(),
        "--prompt".into(),
        ir.to_string(),
        "--ref-image".into(),
        board.display().to_string(),
        "--weights".into(),
        weights.display().to_string(),
        "--width".into(),
        H3CanvasPreset::Mp02.size().0.to_string(),
        "--height".into(),
        H3CanvasPreset::Mp02.size().1.to_string(),
        "--duration".into(),
        args.duration.to_string(),
        "--steps".into(),
        args.steps.to_string(),
        "--shift-video".into(),
        GANLOSS_SHIFT_VIDEO.to_string(),
        "--shift-audio".into(),
        GANLOSS_SHIFT_AUDIO.to_string(),
        "--seed".into(),
        args.seed.to_string(),
        "--output".into(),
        output.display().to_string(),
        "--continuation-mode".into(),
        "ganloss_two_stage".into(),
    ];
    if args.vsa {
        argv.push("--vsa".into());
        argv.push("--vsa-sparsity".into());
        argv.push(args.vsa_sparsity.to_string());
        argv.push("--vsa-gate".into());
        argv.push(args.vsa_gate.clone());
    }
    args.vae_select.push_argv(&mut argv);
    if dry_run {
        argv.push("--dry-run-plan".into());
    }
    H3GenerateArgs::try_parse_from(&argv).map_err(|err| anyhow::anyhow!("{err}"))
}

fn print_dry_run(plan: &ShortfilmPlan, json_out: bool) -> Result<()> {
    if json_out {
        println!("{}", serde_json::to_string_pretty(plan)?);
        return Ok(());
    }
    println!("[h3 shortfilm] dry-run plan");
    println!(
        "[h3 shortfilm] pipeline={} phase={} layout={} scenes={} grid={} mode={} cache={} sampler={} scheduler={} steps={} shift={}/{}",
        plan.pipeline,
        plan.phase,
        plan.layout,
        plan.scenes,
        plan.grid,
        plan.video.mode,
        plan.video.cache,
        plan.video.sampler,
        plan.video.scheduler,
        plan.video.steps,
        plan.video.shift_video,
        plan.video.shift_audio
    );
    println!(
        "[h3 shortfilm] stills panel={}x{} identity_edit_steps={} (photoreal high-res; pack Comfy 768×432 is the fast batch default)",
        plan.stills.panel_size[0],
        plan.stills.panel_size[1],
        plan.stills.identity_edit_steps
    );
    println!(
        "[h3 shortfilm] still quality floors: character-sheet short-edge>={} beat-still>={} 6-grid board>={} (undersized inputs SeedVR2; mushy Krea caps Eros)",
        plan.stills.sheet_min_short_edge,
        plan.stills.still_min_short_edge,
        plan.stills.board_min_short_edge
    );
    if let Some(s) = &plan.stills.character_sheet {
        print_still_size_note(s, plan.stills.sheet_min_short_edge, "character-sheet");
    }
    for (i, p) in plan.stills.panels.iter().enumerate() {
        print_still_size_note(p, plan.stills.still_min_short_edge, &format!("beat still {}", i + 1));
    }
    if let Some(b) = &plan.stills.storyboard {
        print_still_size_note(b, plan.stills.board_min_short_edge, "storyboard");
    }
    println!(
        "[h3 shortfilm] two_stage={} stage1={}x{} stage2={}x{} (ganloss Eros 0.2→1.0, not pack stock FL2VA)",
        plan.video.two_stage,
        plan.video.stage1[0],
        plan.video.stage1[1],
        plan.video.stage2[0],
        plan.video.stage2[1]
    );
    println!(
        "[h3 shortfilm] weights={}",
        plan.video.weights.display()
    );
    println!(
        "[h3 shortfilm] weights_named_bf16={}",
        plan.video.weights_named_bf16.display()
    );
    println!(
        "[h3 shortfilm] eros_root={}",
        plan.download.dest_root.display()
    );
    if plan.layout == "pack-per-scene" {
        println!(
            "[h3 shortfilm] identity_edit={} board_is_first_frame={} dit_refs={} (one still per scene, hard-cut loop)",
            plan.stills.identity_edit,
            plan.stills.board_is_first_frame,
            plan.video.refs.len()
        );
        if let Some(scenes) = plan.loop_plan["scenes"].as_array() {
            for (i, scene) in scenes.iter().enumerate() {
                let cont = scene["continuation_mode"].as_str().unwrap_or("?");
                let pic = scene["tags"]["storyboard"]["path"]
                    .as_str()
                    .unwrap_or("");
                println!(
                    "[h3 shortfilm] scene {} continuation={} Picture 1={}",
                    i + 1,
                    cont,
                    pic
                );
            }
        }
    } else {
        println!(
            "[h3 shortfilm] identity_edit={} board_is_first_frame={} dit_refs={} (ganloss Picture 1 only)",
            plan.stills.identity_edit,
            plan.stills.board_is_first_frame,
            plan.video.refs.len()
        );
        if let Some(b) = &plan.stills.storyboard {
            println!("[h3 shortfilm] @{TAG_BOARD}={}", b.display());
        }
    }
    if let Some(s) = &plan.stills.character_sheet {
        println!("[h3 shortfilm] @{TAG_HERO}={} (stills only, not a DiT ref)", s.display());
    }
    println!("[h3 shortfilm] first_frame={:?}", plan.video.first_frame);
    println!("[h3 shortfilm] output={}", plan.output.display());
    println!("[h3 shortfilm] plan_out={}", plan.plan_out.display());
    println!(
        "[h3 shortfilm] download named {} / {} dest={} present={}",
        plan.download.named.repo,
        plan.download.named.filename,
        plan.download.named.dest.display(),
        plan.download.named.present
    );
    println!(
        "[h3 shortfilm] download int8 {} / {} dest={} present={}",
        plan.download.int8.repo,
        plan.download.int8.filename,
        plan.download.int8.dest.display(),
        plan.download.int8.present
    );
    println!("[h3 shortfilm] Ref2VA IR:");
    println!("{}", plan.video.prompt_ir);
    Ok(())
}

fn run_stills_phase(
    args: &H3ShortfilmArgs,
    plan: &ShortfilmPlan,
    gemmy: &Path,
) -> Result<PathBuf> {
    let sheet = plan
        .stills
        .character_sheet
        .as_ref()
        .context("stills phase needs --character-sheet")?;
    if !sheet.is_file() {
        bail!("character sheet not found: {}", sheet.display());
    }
    std::fs::create_dir_all(&plan.out_dir)?;
    let sheet = ensure_hq_still(
        sheet,
        &plan.out_dir.join("character_sheet_hq.png"),
        SHEET_MIN_SHORT_EDGE,
        args.verbose,
        "character-sheet",
    )?;
    let mut written = Vec::new();
    for (i, prompt) in plan.stills.panel_prompts.iter().enumerate() {
        let dest = plan
            .stills
            .panels
            .get(i)
            .cloned()
            .unwrap_or_else(|| plan.out_dir.join(format!("panel_{}.png", i + 1)));
        if dest.is_file() && args.panel.iter().any(|p| abs(p).ok().as_ref() == Some(&dest)) {
            written.push(dest);
            continue;
        }
        if i == 0 {
            println!(
                "[h3 shortfilm] identity-edit {}x{} steps={} from sheet (sharp photoreal sheet; mushy sheet → mushy Eros)",
                PANEL_W, PANEL_H, STILL_STEPS
            );
        }
        println!(
            "[h3 shortfilm] identity-edit panel {}/{} → {}",
            i + 1,
            plan.stills.panel_prompts.len(),
            dest.display()
        );
        let mut cmd = crate::host::workers::env::gemmy_child_command(gemmy);
        cmd.arg("image")
            .arg("--engine")
            .arg("krea2")
            .arg("--input-image")
            .arg(&sheet)
            .arg("--width")
            .arg(PANEL_W.to_string())
            .arg("--height")
            .arg(PANEL_H.to_string())
            .arg("--steps")
            .arg(STILL_STEPS.to_string())
            .arg("--seed")
            .arg((args.seed + i as u64).to_string())
            .arg("-o")
            .arg(&dest)
            .arg(prompt);
        if args.verbose {
            cmd.arg("-v");
        }
        crate::host::util::run_simple_command(cmd, "h3 shortfilm krea identity-edit", args.verbose)
            .with_context(|| format!("identity-edit panel {}", dest.display()))?;
        if !dest.is_file() {
            bail!("identity-edit panel missing: {}", dest.display());
        }
        written.push(dest);
    }
    let (cols, rows) = grid_shape(written.len() as u32);
    let contact = plan.out_dir.join("contact_sheet.png");
    compose_labeled_board(&written, cols, rows, &contact)?;
    println!(
        "[h3 shortfilm] wrote contact sheet {} (human only, not a DiT ref)",
        contact.display()
    );
    if args.scenes > 1 {
        return Ok(written[0].clone());
    }
    let board = plan
        .stills
        .storyboard
        .clone()
        .unwrap_or_else(|| plan.out_dir.join("scene_board.png"));
    compose_plain_board(&written, cols, rows, &board)?;
    println!("[h3 shortfilm] wrote unlabeled ganloss board {}", board.display());
    Ok(board)
}

pub(crate) fn compose_labeled_board(
    panels: &[PathBuf],
    cols: u32,
    rows: u32,
    dest: &Path,
) -> Result<()> {
    if cols == 0 || rows == 0 {
        bail!("board grid cols/rows must be > 0");
    }
    let gap = 8u32;
    let label_h = 28u32;
    let cell_w = PANEL_W;
    let cell_h = PANEL_H + label_h;
    let width = cols * cell_w + (cols + 1) * gap;
    let height = rows * cell_h + (rows + 1) * gap;
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([18, 18, 20]));
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            let x0 = gap + c * (cell_w + gap);
            let y0 = gap + r * (cell_h + gap);
            fill_rect(
                &mut canvas,
                x0,
                y0,
                cell_w,
                label_h,
                Rgb([32, 32, 36]),
            );
            let label = format!("S{}", idx + 1);
            draw_label(&mut canvas, x0 + 8, y0 + 8, &label, Rgb([240, 240, 242]));
            if let Some(path) = panels.get(idx) {
                if path.is_file() {
                    let img = image::open(path)
                        .with_context(|| format!("open panel {}", path.display()))?
                        .to_rgb8();
                    let resized = image::imageops::resize(
                        &img,
                        cell_w,
                        PANEL_H,
                        image::imageops::FilterType::Lanczos3,
                    );
                    image::imageops::replace(&mut canvas, &resized, i64::from(x0), i64::from(y0 + label_h));
                }
            }
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    canvas
        .save(dest)
        .with_context(|| format!("write board {}", dest.display()))?;
    Ok(())
}

/// Ganloss board: photoreal panels only, no S1 labels (labels make H3 reprint the grid).
pub(crate) fn compose_plain_board(
    panels: &[PathBuf],
    cols: u32,
    rows: u32,
    dest: &Path,
) -> Result<()> {
    if cols == 0 || rows == 0 {
        bail!("board grid cols/rows must be > 0");
    }
    let gap = 4u32;
    let width = cols * PANEL_W + (cols + 1) * gap;
    let height = rows * PANEL_H + (rows + 1) * gap;
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([8, 8, 10]));
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            let x0 = gap + c * (PANEL_W + gap);
            let y0 = gap + r * (PANEL_H + gap);
            if let Some(path) = panels.get(idx) {
                if path.is_file() {
                    let img = image::open(path)
                        .with_context(|| format!("open panel {}", path.display()))?
                        .to_rgb8();
                    let resized = image::imageops::resize(
                        &img,
                        PANEL_W,
                        PANEL_H,
                        image::imageops::FilterType::Lanczos3,
                    );
                    image::imageops::replace(&mut canvas, &resized, i64::from(x0), i64::from(y0));
                }
            }
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    canvas
        .save(dest)
        .with_context(|| format!("write board {}", dest.display()))?;
    Ok(())
}

fn fill_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
    let max_x = img.width();
    let max_y = img.height();
    for yy in y..y.saturating_add(h).min(max_y) {
        for xx in x..x.saturating_add(w).min(max_x) {
            img.put_pixel(xx, yy, color);
        }
    }
}

/// Tiny 5×7 bitmap font for board labels (digits + S).
fn draw_label(img: &mut RgbImage, x: u32, y: u32, text: &str, color: Rgb<u8>) {
    let mut cx = x;
    for ch in text.chars() {
        if let Some(glyph) = glyph(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        let px = cx + col;
                        let py = y + row as u32;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }
        cx += 6;
    }
}

fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        'S' => [0b01110, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b01110],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        _ => return None,
    })
}

fn hf_token() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(value) = std::env::var(key) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    let mut files = Vec::new();
    if let Some(root) = crate::host::paths::models_root() {
        files.push(root.join("huggingface").join("token"));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        let home = home.trim().to_string();
        if !home.is_empty() {
            files.push(PathBuf::from(home).join("token"));
        }
    }
    files.push(PathBuf::from(r"F:\Models\huggingface\token"));
    for path in files {
        if let Ok(value) = std::fs::read_to_string(&path) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn hf_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent("Gemmy/0.1 (h3-shortfilm)")
        .timeout(None)
        .connect_timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(16))
        .build()
        .context("build Hugging Face client")
}

pub(crate) fn download_eros_weights(plan: &ErosDownloadPlan, verbose: bool) -> Result<()> {
    std::fs::create_dir_all(eros_dit_dir())
        .with_context(|| format!("create {}", eros_dit_dir().display()))?;
    // Runnable INT8 first (16 GB generate default), then named BF16 archive.
    download_hf_file(&plan.int8, verbose)?;
    download_hf_file(&plan.named, verbose)?;
    Ok(())
}

fn download_hf_file(file: &ErosFilePlan, verbose: bool) -> Result<()> {
    if file.dest.is_file() {
        let n = std::fs::metadata(&file.dest)?.len();
        if n > 1_000_000 {
            println!(
                "[h3 shortfilm] already present {} ({} bytes)",
                file.dest.display(),
                n
            );
            return Ok(());
        }
        eprintln!(
            "[h3 shortfilm] existing file looks too small ({} bytes); re-downloading {}",
            n,
            file.dest.display()
        );
    }
    println!(
        "[h3 shortfilm] downloading {} / {} → {}",
        file.repo,
        file.filename,
        file.dest.display()
    );
    resume_download(&file.url, &file.dest, verbose, "h3 shortfilm")
        .with_context(|| format!("download {} :: {}", file.repo, file.filename))?;
    let n = std::fs::metadata(&file.dest)?.len();
    if n < 1_000_000 {
        bail!(
            "downloaded {} is only {n} bytes — not a multi-GB weight file",
            file.dest.display()
        );
    }
    println!(
        "[h3 shortfilm] wrote {} ({} bytes)",
        file.dest.display(),
        n
    );
    Ok(())
}

pub(crate) fn resume_download(url: &str, dest: &Path, verbose: bool, label: &str) -> Result<()> {
    let client = hf_client()?;
    let token = hf_token();
    let part = dest.with_extension("safetensors.part");
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=8 {
        match resume_download_once(&client, token.as_deref(), url, dest, &part, verbose, label) {
            Ok(()) => return Ok(()),
            Err(err) => {
                let retryable = err.to_string().contains("429")
                    || err.to_string().to_ascii_lowercase().contains("too many requests");
                if !retryable || attempt == 8 {
                    return Err(err);
                }
                let wait = 15u64.saturating_mul(attempt as u64);
                println!("[{label}] HTTP 429 (attempt {attempt}/8); retrying in {wait}s");
                std::io::stdout().flush().ok();
                std::thread::sleep(Duration::from_secs(wait));
                last_err = Some(err);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("download failed")))
}

fn resume_download_once(
    client: &reqwest::blocking::Client,
    token: Option<&str>,
    url: &str,
    dest: &Path,
    part: &Path,
    verbose: bool,
    label: &str,
) -> Result<()> {
    let mut have = if part.is_file() {
        std::fs::metadata(part)?.len()
    } else {
        0
    };

    let mut req = client.get(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        println!("[{label}] resuming at byte {have}");
    }
    let resp = req.send().with_context(|| format!("request {url}"))?;
    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        bail!("429 Too Many Requests for {url}");
    }
    let mut resp = resp
        .error_for_status()
        .with_context(|| format!("download {url}"))?;
    let status = resp.status();
    if have > 0 && status == reqwest::StatusCode::OK {
        // Server ignored Range — restart.
        have = 0;
        let _ = std::fs::remove_file(&part);
        println!("[{label}] server ignored Range; restarting from 0");
    }
    let total = resp
        .content_length()
        .map(|n| if status == reqwest::StatusCode::PARTIAL_CONTENT { have + n } else { n });

    let mut out = if have > 0 {
        std::fs::OpenOptions::new().append(true).open(&part)
    } else {
        if let Some(parent) = part.parent() {
            std::fs::create_dir_all(parent)?;
        }
        File::create(&part)
    }
    .with_context(|| format!("open {}", part.display()))?;

    let mut buf = vec![0u8; 1024 * 1024];
    let mut got = have;
    let t0 = Instant::now();
    let mut last_report = Instant::now();
    loop {
        let n = resp.read(&mut buf).context("read download body")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        got += n as u64;
        if last_report.elapsed() >= Duration::from_secs(5) || verbose {
            let secs = t0.elapsed().as_secs_f64().max(0.001);
            let mb = got as f64 / 1_048_576.0;
            let speed = (got.saturating_sub(have) as f64 / 1_048_576.0) / secs;
            match total {
                Some(t) => println!(
                    "[{label}] {:.1} / {:.1} MB ({:.1} MB/s)",
                    mb,
                    t as f64 / 1_048_576.0,
                    speed
                ),
                None => println!("[{label}] {mb:.1} MB ({speed:.1} MB/s)"),
            }
            last_report = Instant::now();
            std::io::stdout().flush().ok();
        }
    }
    out.flush()?;
    drop(out);
    if dest.exists() {
        std::fs::remove_file(dest).ok();
    }
    std::fs::rename(&part, dest)
        .with_context(|| format!("rename {} → {}", part.display(), dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scene_ir::{contains_scene_n_of_m, per_scene_ir_board_leak};
    use super::super::args::H3GenerateArgs;
    use clap::Parser;

    fn write_tiny_png(path: &Path) {
        let img = RgbImage::from_pixel(32, 32, Rgb([200, 40, 40]));
        img.save(path).unwrap();
    }

    #[test]
    fn shortfilm_plan_is_ref2va_eros_int8_with_six_ir_sections() {
        let dir = tempfile::tempdir().unwrap();
        let board = dir.path().join("board.png");
        write_tiny_png(&board);
        let args = H3ShortfilmArgs::try_parse_from([
            "gemmy-shortfilm",
            "--storyboard",
            board.to_str().unwrap(),
            "--prompt",
            "a rooftop dance at night in a blue outfit",
            "--grid",
            "6",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_shortfilm_plan(&args).unwrap();
        assert_eq!(plan.video.mode, "ref2va");
        assert_eq!(plan.video.refs.len(), 1);
        assert_eq!(plan.video.refs[0].role, "ganloss_picture_1");
        let dit = plan
            .video
            .weights
            .file_name()
            .unwrap()
            .to_string_lossy();
        assert!(
            dit.contains("10Eros_Max_h3_TURBO_ref2va_beta2"),
            "dit={dit}"
        );
        assert!(dit.contains("int8_convrot"), "dit={dit}");
        assert_eq!(plan.video.cache, "off");
        assert!(plan.video.two_stage);
        assert_eq!(plan.video.stage1, [608, 352]);
        assert_eq!(plan.video.stage2, [1344, 768]);
        assert!(!plan.video.spectrum);
        assert!(!plan.video.fbc);
        assert_eq!(plan.video.sampler, "euler");
        assert_eq!(plan.video.scheduler, "simple");
        assert_eq!(plan.video.steps, GANLOSS_EROS_STEPS);
        assert_eq!(plan.video.shift_video, GANLOSS_SHIFT_VIDEO);
        assert_eq!(plan.video.shift_audio, GANLOSS_SHIFT_AUDIO);
        assert_eq!(plan.loop_plan["steps"], json!(GANLOSS_EROS_STEPS));
        assert!(plan.video.first_frame.is_none());
        assert!(!plan.stills.board_is_first_frame);
        assert!(plan.stills.identity_edit);
        assert_eq!(plan.stills.panel_size, [1536, 864]);
        assert_eq!(plan.stills.identity_edit_steps, 12);
        assert_eq!(plan.stills.sheet_min_short_edge, 1536);
        assert_eq!(plan.stills.still_min_short_edge, 864);
        assert_eq!(plan.stills.board_min_short_edge, 1536);
        let ir = &plan.video.prompt_ir;
        for heading in [
            "subject_definitions:",
            "summary:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        ] {
            assert!(ir.contains(heading), "missing {heading}");
        }
        assert!(ir.contains("[Shot"));
        assert!(ir.contains("<Picture 1>"));
        assert!(!ir.contains("<Picture 2>"));
        assert!(!ir.contains("@hero_face"));
        assert!(ir.contains("chronological shot beat"));
        assert!(
            ir.to_ascii_lowercase().contains("panel borders")
                || ir.to_ascii_lowercase().contains("storyboard grid")
        );
        let loop_txt = serde_json::to_string_pretty(&plan.loop_plan).unwrap();
        assert!(loop_txt.contains("storyboard"));
        assert!(!loop_txt.contains("hero_face"));
        assert_eq!(plan.loop_plan["schema"], "gemmy-h3-loop-v1");
        assert_eq!(plan.loop_plan["mode"], "ref2va");
        assert_eq!(plan.loop_plan["cache"], "off");
        assert_eq!(plan.download.named.repo, "TenStrip/10Eros-Max");
        assert_eq!(
            plan.download.named.filename,
            "10Eros_Max_h3_TURBO_ref2va_beta2.safetensors"
        );
        assert_eq!(
            plan.download.int8.repo,
            "cicalooo/10Eros-Max-h3-int8-convrot"
        );
        assert_eq!(
            plan.download.int8.filename,
            "10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors"
        );
        assert!(plan
            .download
            .dest_root
            .to_string_lossy()
            .contains("minimax-h3-eros"));
    }

    #[test]
    fn default_generate_i2v_still_uses_stock_fl2va() {
        let args = H3GenerateArgs::try_parse_from([
            "gemmy-h3-generate",
            "--prompt",
            "a red fox trots through fresh snow",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_h3_generate_plan(&args).unwrap();
        assert_eq!(plan.mode, "i2v");
        let name = plan.weights.file_name().unwrap().to_string_lossy();
        assert!(
            name.contains("minimax_h3_fl2va_pruned_int8_convrot"),
            "default I2V dit={name}"
        );
        assert!(!name.contains("10Eros"));
        assert_eq!(plan.steps, 20);
        assert_eq!(plan.shift_video, 12.0);
        assert_eq!(plan.shift_audio, 3.0);
    }

    #[test]
    fn shortfilm_generate_args_use_ganloss_eros_recipe_not_stock_20() {
        let dir = tempfile::tempdir().unwrap();
        let sheet = dir.path().join("sheet.png");
        let board = dir.path().join("board.png");
        write_tiny_png(&sheet);
        write_tiny_png(&board);
        let args = H3ShortfilmArgs::try_parse_from([
            "gemmy-shortfilm",
            "--character-sheet",
            sheet.to_str().unwrap(),
            "--storyboard",
            board.to_str().unwrap(),
            "--prompt",
            "a rooftop dance at night",
            "--dry-run-plan",
        ])
        .unwrap();
        assert_eq!(args.steps, GANLOSS_EROS_STEPS);
        let gen_args = generate_args_for_shortfilm(
            "subject_definitions:\n",
            &board,
            &eros_ref2va_int8_path(),
            &dir.path().join("out.mp4"),
            &args,
            true,
        )
        .unwrap();
        assert_eq!(gen_args.steps, Some(GANLOSS_EROS_STEPS));
        assert_eq!(gen_args.shift_video, GANLOSS_SHIFT_VIDEO);
        assert_eq!(gen_args.shift_audio, GANLOSS_SHIFT_AUDIO);
        assert_eq!(gen_args.cache.as_str(), "off");
        assert_eq!(gen_args.continuation_mode, "ganloss_two_stage");
        assert_eq!(gen_args.ref_image.len(), 1);
        assert_eq!(gen_args.width, Some(608));
        assert_eq!(gen_args.height, Some(352));
        let gp = build_h3_generate_plan(&gen_args).unwrap();
        assert_eq!(gp.mode, "ref2va");
        assert_eq!(gp.steps, GANLOSS_EROS_STEPS);
        assert_eq!(gp.shift_video, GANLOSS_SHIFT_VIDEO);
        assert_eq!(gp.shift_audio, GANLOSS_SHIFT_AUDIO);
        assert_eq!(gp.cache, "off");
        assert!(!gp.compile_ir);
    }

    #[test]
    fn compile_ganloss_ir_is_one_picture_six_beats() {
        let ir = compile_ganloss_ir("walks across a moonlit roof", 12.0, 6);
        assert!(ir.contains("[Shot 1]"));
        assert!(ir.contains("[Shot 6]"));
        assert!(ir.contains("<Picture 1>"));
        assert!(!ir.contains("<Picture 2>"));
        assert!(!ir.contains("@hero_face"));
        assert!(ir.to_ascii_lowercase().contains("split screen"));
    }

    #[test]
    fn labeled_board_compose_writes_png() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.png");
        let b = dir.path().join("b.png");
        write_tiny_png(&a);
        write_tiny_png(&b);
        let dest = dir.path().join("board.png");
        compose_labeled_board(&[a, b], 3, 2, &dest).unwrap();
        assert!(dest.is_file());
        let img = image::open(&dest).unwrap();
        assert!(img.width() > 100);
        assert!(img.height() > 100);
    }

    #[test]
    fn compile_scene_ir_matches_parlor_quality_lock() {
        // outputs/locked_h3_shortfilm_parlor_20260827/ — one still, one 10s take.
        let ir = compile_scene_ir(
            "the woman in the navy pea coat stands in the firelit parlor, slow breath, firelight on her face, she looks toward camera then slightly aside",
            10.0,
            1,
            1,
        );
        assert_eq!(per_scene_ir_board_leak(&ir), None);
        assert!(ir.contains("Stay inside the same room and framing as <Picture 1>"));
        assert!(ir.contains("Photoreal live-action photograph"));
        assert!(ir.contains("Create a 10-second photoreal cinematic clip"));
        assert!(!ir.contains("[Shot"));
        assert!(!ir.contains("scene 1 of"));
    }

    #[test]
    fn compile_scene_ir_is_one_picture_not_a_grid() {
        let ir = compile_scene_ir("walks into a diner", 8.0, 2, 4);
        assert!(ir.contains("<Picture 1>"));
        assert!(!ir.contains("<Picture 2>"));
        assert!(!ir.contains("@hero_face"));
        assert!(!ir.contains("scene 2 of 4"));
        assert_eq!(per_scene_ir_board_leak(&ir), None);
        for heading in [
            "subject_definitions:",
            "summary:",
            "retention_analysis:",
            "detailed_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        ] {
            assert!(ir.contains(heading), "missing {heading}");
        }
    }

    #[test]
    fn per_scene_ir_board_leak_catches_old_poison() {
        let old = "scene 1 of 2. This is one composition, not a storyboard grid or split screen. Do not show a contact sheet. [Shot 1]";
        assert_eq!(per_scene_ir_board_leak(old), Some("storyboard"));
        assert!(contains_scene_n_of_m("scene 1 of 2"));
        assert!(!contains_scene_n_of_m("natural scene ambience"));
    }

    #[test]
    fn expand_panel_prompts_is_one_photograph() {
        let ps = expand_panel_prompts(2, &[], "stands in a diner");
        assert_eq!(ps.len(), 2);
        for p in &ps {
            let l = p.to_ascii_lowercase();
            assert!(!l.contains("storyboard"), "{p}");
            assert!(!l.contains("shot 1 of"), "{p}");
            assert!(!l.contains("shot 2 of"), "{p}");
            assert!(!l.contains("comic"), "{p}");
            assert!(!l.contains("grid"), "{p}");
        }
        let reused = expand_panel_prompts(2, &["leather armchair, firelight".into()], "");
        assert_eq!(reused[0], reused[1]);
        assert!(!reused[1].to_ascii_lowercase().contains("shot"));
    }

    #[test]
    fn shortfilm_scenes_gt_1_is_one_picture_per_scene_hard_cut() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.png");
        let b = dir.path().join("b.png");
        write_tiny_png(&a);
        write_tiny_png(&b);
        let args = H3ShortfilmArgs::try_parse_from([
            "gemmy-shortfilm",
            "--scenes",
            "2",
            "--panel",
            a.to_str().unwrap(),
            "--panel",
            b.to_str().unwrap(),
            "--prompt",
            "walks then sits",
            "--phase",
            "video",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_shortfilm_plan(&args).unwrap();
        assert_eq!(plan.layout, "pack-per-scene");
        assert_eq!(plan.scenes, 2);
        assert_eq!(plan.grid, 2);
        assert_eq!(plan.video.refs.len(), 2);
        assert_eq!(plan.video.refs[0].role, "scene_picture_1");
        assert_eq!(plan.video.refs[1].path, b);
        assert_eq!(plan.loop_plan["layout"], json!("per-scene"));
        assert_eq!(plan.loop_plan["compile_ir"], json!(false));
        assert_eq!(plan.loop_plan["two_stage"], json!(true));
        assert_eq!(plan.loop_plan["canvas"]["width"], json!(608));
        assert_eq!(plan.loop_plan["canvas"]["height"], json!(352));
        assert_eq!(plan.loop_plan["refine_canvas"]["width"], json!(1344));
        assert_eq!(plan.loop_plan["refine_canvas"]["height"], json!(768));
        assert_eq!(plan.loop_plan["steps"], json!(GANLOSS_EROS_STEPS));
        assert!(plan.video.two_stage);
        assert_eq!(plan.video.stage1, [608, 352]);
        assert_eq!(plan.video.stage2, [1344, 768]);
        let scenes = plan.loop_plan["scenes"].as_array().unwrap();
        assert_eq!(scenes.len(), 2);
        for scene in scenes.iter() {
            assert_eq!(scene["continuation_mode"], json!("none"));
            assert_eq!(scene["mode"], json!("ref2va"));
            let tags = scene["tags"].as_object().unwrap();
            assert_eq!(tags.len(), 1);
            assert!(tags.contains_key("storyboard"));
            assert!(!tags.contains_key("hero_face"));
            let prompt = scene["prompt"].as_str().unwrap();
            assert!(prompt.contains("<Picture 1>"));
            assert!(!prompt.contains("<Picture 2>"));
            assert!(prompt.contains("walks then sits"));
            assert!(!prompt.contains("Keep the same person"));
            assert!(!prompt.contains("Photoreal cinematic still"));
            assert!(!prompt.contains("scene 1 of 2"));
            assert!(!prompt.contains("scene 2 of 2"));
            assert_eq!(per_scene_ir_board_leak(prompt), None);
            assert!(!prompt.contains("[Shot 6]"));
        }
        let pic0 = scenes[0]["tags"]["storyboard"]["path"].as_str().unwrap();
        let pic1 = scenes[1]["tags"]["storyboard"]["path"].as_str().unwrap();
        assert!(pic0.ends_with("a.png") || pic0.replace('\\', "/").ends_with("a.png"));
        assert!(pic1.ends_with("b.png") || pic1.replace('\\', "/").ends_with("b.png"));
        assert_ne!(pic0, pic1);
        let loop_txt = serde_json::to_string(&plan.loop_plan).unwrap();
        assert!(!loop_txt.contains("hero_face"));
        assert!(!loop_txt.contains("masked_av"));
    }

    #[test]
    fn shortfilm_scenes_gt_1_rejects_composite_storyboard_alone() {
        let dir = tempfile::tempdir().unwrap();
        let board = dir.path().join("board.png");
        write_tiny_png(&board);
        let args = H3ShortfilmArgs::try_parse_from([
            "gemmy-shortfilm",
            "--scenes",
            "2",
            "--storyboard",
            board.to_str().unwrap(),
            "--prompt",
            "walks then sits",
            "--dry-run-plan",
        ])
        .unwrap();
        let err = build_shortfilm_plan(&args).unwrap_err().to_string();
        assert!(
            err.contains("one still per scene"),
            "err={err}"
        );
    }

    #[test]
    fn shortfilm_scenes_1_stays_ganloss_one_window() {
        let dir = tempfile::tempdir().unwrap();
        let board = dir.path().join("board.png");
        write_tiny_png(&board);
        let args = H3ShortfilmArgs::try_parse_from([
            "gemmy-shortfilm",
            "--scenes",
            "1",
            "--storyboard",
            board.to_str().unwrap(),
            "--prompt",
            "a rooftop dance",
            "--dry-run-plan",
        ])
        .unwrap();
        let plan = build_shortfilm_plan(&args).unwrap();
        assert_eq!(plan.layout, "ganloss-one-window");
        assert_eq!(plan.scenes, 1);
        assert_eq!(plan.video.refs.len(), 1);
        assert_eq!(plan.video.refs[0].role, "ganloss_picture_1");
        let scenes = plan.loop_plan["scenes"].as_array().unwrap();
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0]["continuation_mode"], json!("none"));
    }

    #[test]
    fn still_needs_upscale_below_quality_floor() {
        assert!(still_needs_upscale(32, 32, SHEET_MIN_SHORT_EDGE));
        assert!(still_needs_upscale(768, 432, STILL_MIN_SHORT_EDGE));
        assert!(still_needs_upscale(1024, 1024, SHEET_MIN_SHORT_EDGE));
        assert!(!still_needs_upscale(1536, 864, STILL_MIN_SHORT_EDGE));
        assert!(!still_needs_upscale(1536, 1536, SHEET_MIN_SHORT_EDGE));
        assert!(!still_needs_upscale(2688, 1536, BOARD_MIN_SHORT_EDGE));
        assert!(still_needs_upscale(1280, 720, BOARD_MIN_SHORT_EDGE));
    }

    #[test]
    fn patch_loop_plan_rewrites_storyboard_paths() {
        let a = PathBuf::from("a.png");
        let b = PathBuf::from("b_hq.png");
        let mut plan = json!({
            "scenes": [
                {"tags": {"storyboard": {"kind": "image", "path": "a.png"}}},
                {"tags": {"storyboard": {"kind": "image", "path": "old.png"}}},
            ]
        });
        patch_loop_plan_still_paths(&mut plan, &[a, b.clone()]);
        assert_eq!(
            plan["scenes"][1]["tags"]["storyboard"]["path"],
            json!(b)
        );
    }
}
