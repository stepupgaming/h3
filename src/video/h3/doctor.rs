//! `h3 doctor` — paths, weights, CUDA, Sage, FA2, ffmpeg.

use super::args::H3DoctorArgs;
use super::models::{all_required_ok, collect_asset_reports};
use super::paths::{
    checkpoints_root, comfy_root, comfy_worker_script, h3_python, h3_root, runtime_root,
    realism_people_lora, turbo_lora_v4, worker_script,
};
use crate::host::config::H3Config;
use crate::host::paths::find_ffmpeg;
use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct DoctorReport {
    ok: bool,
    h3_root: String,
    runtime_root: String,
    checkpoints_root: String,
    comfy_root: Option<String>,
    python: Option<String>,
    worker: Option<String>,
    comfy_worker: Option<String>,
    ffmpeg: Option<String>,
    assets_ok: bool,
    assets: Vec<super::models::AssetReport>,
    cuda: Option<Value>,
    sage: Option<Value>,
    fa2: Option<Value>,
    comfy: Option<Value>,
    notes: Vec<String>,
    blockers: Vec<String>,
}

pub(crate) fn run_doctor(args: H3DoctorArgs, _config: &H3Config) -> Result<()> {
    let mut notes = Vec::new();
    let mut blockers = Vec::new();

    let root = h3_root();
    let py_path = h3_python();
    let worker = worker_script();

    if !root.is_dir() {
        blockers.push(format!("h3_root missing: {}", root.display()));
    }
    let python = if py_path.is_file() {
        Some(py_path.display().to_string())
    } else {
        blockers.push(format!(
            "H3 python missing: {} (run: h3 install)",
            py_path.display()
        ));
        None
    };
    let worker_s = if worker.is_file() {
        Some(worker.display().to_string())
    } else {
        blockers.push(format!("python worker missing: {}", worker.display()));
        None
    };

    let comfy_w = comfy_worker_script();
    let comfy_worker_s = if comfy_w.is_file() {
        Some(comfy_w.display().to_string())
    } else {
        notes.push(format!(
            "comfy worker missing: {} (default --engine comfy needs it)",
            comfy_w.display()
        ));
        None
    };

    let croot = comfy_root();
    let comfy_root_s = if croot.join("run_h3_workflow.py").is_file() {
        Some(croot.display().to_string())
    } else {
        notes.push(format!(
            "ComfyUI root incomplete: {} (expected runtimes\\minimax-h3\\ComfyUI; dev override GEMMY_H3_COMFY)",
            croot.display()
        ));
        None
    };

    let ffmpeg = match find_ffmpeg() {
        Ok(p) => Some(p.display().to_string()),
        Err(err) => {
            notes.push(format!(
                "ffmpeg not found via Gemmy paths ({err:#}); decode may use imageio-ffmpeg"
            ));
            None
        }
    };

    let assets = collect_asset_reports();
    let assets_ok = all_required_ok(&assets);
    if !assets_ok {
        for r in assets.iter().filter(|r| r.required && !r.ok) {
            blockers.push(format!("required {}: {}", r.id, r.path));
        }
    }

    let want_cuda = args.cuda && !args.no_cuda;
    let want_sage = args.sage && !args.no_sage;
    let want_fa2 = args.fa2 && !args.no_fa2;

    let mut cuda = None;
    let mut sage = None;
    let mut fa2 = None;
    let mut comfy_probe = None;

    if let Some(ref py) = python {
        if want_cuda {
            cuda = Some(probe_python(
                py,
                r#"
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "cuda_available": bool(torch.cuda.is_available()),
  "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
  "device0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
"#,
                args.verbose,
            ));
            if let Some(Value::Object(map)) = &cuda {
                if map.get("cuda_available") != Some(&Value::Bool(true)) {
                    blockers.push("torch.cuda.is_available() is false".into());
                }
            } else if cuda.as_ref().and_then(|v| v.get("error")).is_some() {
                blockers.push("CUDA probe failed".into());
            }
        }
        if want_sage {
            sage = Some(probe_python(
                py,
                r#"
import json
try:
    import sageattention
    print(json.dumps({"ok": True, "module": getattr(sageattention, "__file__", "sageattention")}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
"#,
                args.verbose,
            ));
            if sage
                .as_ref()
                .and_then(|v| v.get("ok"))
                .and_then(|v| v.as_bool())
                != Some(true)
            {
                blockers.push("sageattention import failed (default --quality fast)".into());
            }
        }
        if want_fa2 {
            fa2 = Some(probe_python(
                py,
                r#"
import json
try:
    from flash_attn import flash_attn_func
    print(json.dumps({"ok": True, "symbol": "flash_attn_func"}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
"#,
                args.verbose,
            ));
            if fa2
                .as_ref()
                .and_then(|v| v.get("ok"))
                .and_then(|v| v.as_bool())
                != Some(true)
            {
                notes.push(
                    "flash_attn not importable — --quality hq will fail; fast/Sage still OK".into(),
                );
            }
        }
    }

    // Comfy engine readiness (default product path).
    if let Some(ref py) = python {
        if comfy_root_s.is_some() {
            let lora = turbo_lora_v4();
            let realism_lora = realism_people_lora();
            let cr_esc = croot.display().to_string().replace('\\', "\\\\");
            let lora_esc = lora.display().to_string().replace('\\', "\\\\");
            let realism_esc = realism_lora.display().to_string().replace('\\', "\\\\");
            comfy_probe = Some(probe_python(
                py,
                &format!(
                    r#"
import json, os, sys
from pathlib import Path
comfy = Path(r'''{cr}''')
sys.path.insert(0, str(comfy))
os.chdir(comfy)
pins = ["ComfyUI-MiniMax-H3-Turbo","ComfyUI-sol-attn","ComfyUI-Spectrum-MiniMax-H3","ComfyUI-KJNodes","ComfyUI-MiniMaxH3-FirstBlockCache","ComfyUI-NVIDIA-RTX-VSR-Pro","gemmy-h3-context","ComfyUI-Ref2VA-VSA","ComfyUI-PixelForge-H3","ComfyUI-NVIDIA-DLSS-Frame-Interpolation","ComfyUI-MiniMaxH3Mod"]
report = {{
  "comfy_root": str(comfy),
  "runner": (comfy/"run_h3_workflow.py").is_file(),
  "builder_py": (comfy/"h3_workflow_build.py").is_file(),
  "turbo_v4_lora": Path(r'''{lora}''').is_file(),
  "realism_people_lora": Path(r'''{realism}''').is_file(),
  "custom_nodes": {{p: (comfy/"custom_nodes"/p).is_dir() for p in pins}},
}}
try:
    import h3_workflow_build
    report["builder"] = True
except Exception as e:
    report["builder"] = False
    report["builder_error"] = f"{{type(e).__name__}}: {{e}}"
try:
    import comfyui_version
    report["comfyui_version"] = getattr(comfyui_version, "__version__", None)
except Exception:
    report["comfyui_version"] = None
try:
    import importlib.metadata as md
    report["comfy_kitchen"] = md.version("comfy-kitchen")
except Exception:
    report["comfy_kitchen"] = None
print(json.dumps(report))
"#,
                    cr = cr_esc,
                    lora = lora_esc,
                    realism = realism_esc,
                ),
                args.verbose,
            ));
            if let Some(Value::Object(map)) = &comfy_probe {
                if map.get("runner") != Some(&Value::Bool(true)) {
                    notes.push(
                        "Comfy run_h3_workflow.py missing — --engine comfy will fail".into(),
                    );
                }
                if map.get("turbo_v4_lora") != Some(&Value::Bool(true)) {
                    notes.push(
                        "Larryvrh turbo v4 LoRA missing under checkpoints/loras/ (needed for --turbo v4)"
                            .into(),
                    );
                }
                if map.get("realism_people_lora") != Some(&Value::Bool(true)) {
                    notes.push(
                        "fal Realism People LoRA missing under checkpoints/loras/ (needed for --realism)"
                            .into(),
                    );
                }
                if map.get("builder") != Some(&Value::Bool(true)) {
                    notes.push(
                        "h3_workflow_build import failed — Comfy graph builder unavailable".into(),
                    );
                }
                if let Some(Value::Object(nodes)) = map.get("custom_nodes") {
                    for (name, okv) in nodes {
                        if okv != &Value::Bool(true) {
                            notes.push(format!("Comfy custom_node missing: {name}"));
                        }
                    }
                }
                match map.get("comfyui_version").and_then(|v| v.as_str()) {
                    Some("0.36.0") => {}
                    other => notes.push(format!(
                        "ComfyUI pin expected 0.36.0, got {other:?} (fused VAE kernels need this tag)"
                    )),
                }
                match map.get("comfy_kitchen").and_then(|v| v.as_str()) {
                    Some("0.2.34") => {}
                    other => notes.push(format!(
                        "comfy-kitchen pin expected 0.2.34, got {other:?} (fused fp16/int8 VAE decode)"
                    )),
                }
            }
            let optional_nodes = [
                ("ComfyUI-H3-FaceRefine", "optional face-refine pack"),
                (
                    "Comfyui_Minimax_h3_latent_Upscaler",
                    "optional 3D latent upscaler",
                ),
                (
                    "ComfyUI-H3-Motion-Context",
                    "optional Motion Context pack (continuous FL2VA speech; GPL local clone, v0.3.1)",
                ),
                (
                    "ComfyUI-MiniMax-H3-009jev",
                    "optional 009jev pack (H3JevNativeSLAPatch; GPL; --jev / --sla-fixed, not default generate)",
                ),
            ];
            for (name, role) in optional_nodes {
                let dir = croot.join("custom_nodes").join(name);
                if !dir.is_dir() {
                    notes.push(format!("{role} not installed ({name}) — not required for generate"));
                }
            }
            let jev_sdk = super::paths::jev_sdk_python();
            if !jev_sdk.is_file() {
                notes.push(
                    "009jev SDK Python missing (runtimes/minimax-h3/jev-sdk `.venv`; uv sync) — required only for --jev"
                        .into(),
                );
            }
        }
    }

    notes.push(
        "Default generate: --engine comfy, I2V via Krea 2 still → H3, 864x480, 5s snapped, 20 steps, Sage fast, open IR compile on."
            .into(),
    );
    notes.push(
        "Masked region replace (`h3 edit` / video-h3-edit): SAM3.1 track → overlay → gemmy analyze gate → ganloss two-stage Ref2VA (crop as ref_video, 0.2 MP SplitSigmas@4 then 3D SR + 3-step) → uncrop. --stop-after-mask exits before Eros. Source: ganloss 2026-08-31 mask-edit JSON + Veteran AI PJWfUAO1Oco. Crop is Gemmy-owned (not GPL MaskVidExperiments)."
            .into(),
    );
    notes.push(format!(
        "Short Film Director pipeline (`h3 shortfilm` / video-h3-shortfilm): `--scenes 1` ganloss one 6-panel Picture; `--scenes N` Krea identity-edit stills + one Picture per scene, hard-cut loop. Eros TURBO (`{}` runnable INT8 on F:; named BF16 `{}` is download-only). Cache/spectrum/fbc stay off. Stock FL2VA generate is unchanged. See docs\\H3_SHORTFILM_PIPELINE.md.",
        super::paths::EROS_REF2VA_INT8,
        super::paths::EROS_REF2VA_BF16,
    ));
    let eros_int8 = super::paths::eros_ref2va_int8_path();
    let eros_bf16 = super::paths::eros_ref2va_bf16_path();
    if eros_int8.is_file() {
        notes.push(format!(
            "Eros INT8 ConvRot present: {}",
            eros_int8.display()
        ));
    } else {
        notes.push(format!(
            "Eros INT8 ConvRot missing (required for --mode ref2va and shortfilm): {} — h3 shortfilm --phase download",
            eros_int8.display()
        ));
    }
    let singularity = super::paths::singularity_ref2va_int8_path();
    if singularity.is_file() {
        notes.push(format!(
            "Singularity INT8 present (opt-in --dit singularity, not the Ref2VA default): {}",
            singularity.display()
        ));
    } else {
        notes.push(format!(
            "Singularity INT8 missing (optional --dit singularity): {}. Hugging Face WarmBloodAban/Minimax-h3_Singularity",
            singularity.display()
        ));
    }
    if eros_bf16.is_file() {
        notes.push(format!(
            "Eros named BF16 present: {}",
            eros_bf16.display()
        ));
    } else {
        notes.push(format!(
            "Eros named BF16 missing (not loaded on 16 GB): {}",
            eros_bf16.display()
        ));
    }
    notes.push(
        "Video VAE: --vae fp16|int8 (default int8, official Comfy-Org). Audio VAE is always fp32 (not a --vae choice). --video-vae NAME under checkpoints/vae/ wins. --engine python is fp16 only. Continue/loop reuse the sidecar vae when --vae/--video-vae are omitted."
            .into(),
    );
    notes.push(
        "Speed/style flags (Comfy): --turbo v4|v1|ema_wan, --realism (fal People LoRA + r34l1sm), --sol, --vsa (Ref2VA gate, mutex --sol/--cache), --jev (009jev native SLA; omit --steps = 4-step res_multistep; --steps 20 = HQ euler; mutex --vsa/--sol/--cache, needs TYPESAFE_API_KEY; not default), --sla-fixed 1|3|5|10 (true fixed native SLA, no Jev), --sla-table PATH (N×50), --jev-log-dataset DIR, --cache spectrum|easy|fbc, --dit-quant int8|w4a8. Fallback: --engine python."
            .into(),
    );
    notes.push(
        "Pass --first-frame to skip auto Krea; --mode t2va pure text; --mode fl2va first+last; --mode ref2va with --ref-* (Eros two-stage 608→1344 default, including --ref-audio; stock Comfy-Org Ref2VA is --weights on G:\\Models\\minimax-h3-backup); --allow-keyframe-refs for experimental keyframe+ref mix."
            .into(),
    );
    notes.push(
        "Masked region replace: h3 edit --video plate.mp4 --ref-image still.png --mask-prompt head (SAM3.1 track, overlay verify via gemmy analyze, Eros crop sample; --stop-after-mask to inspect). Not whole-frame Ref2VA."
            .into(),
    );
    let sam3 = super::paths::sam3_checkpoint_path("");
    if sam3.is_file() {
        notes.push(format!("SAM3.1 checkpoint present: {}", sam3.display()));
    } else {
        notes.push(format!(
            "SAM3.1 checkpoint missing (required for video h3 edit unless --mask): {} — Comfy-Org/sam3.1 sam3.1_multiplex_fp16.safetensors",
            sam3.display()
        ));
    }
    notes.push(
        "Sprites (`h3 sprites`): pinned ComfyUI-PixelForge-H3 + package comfy-workflow-h3-pixelforge. Key at full res → crop to character → loop trim → sheet/GIF. Do not quantize the full chroma plate. Needs the pin under custom_nodes; not required for generate. Old scripts/sprites cut/atlas is not the product path."
            .into(),
    );
    notes.push(
        "Interpolate (`h3 interpolate`): pinned ComfyUI-NVIDIA-DLSS-Frame-Interpolation + package comfy-workflow-h3-dlss-interpolate. Frame Generation post on a finished MP4 (default 24→48). Not an upscale path (VSR-Pro stays `upscale --backend rtx`). NVIDIA SDK DLLs are local Git LFS, not committed; `interpolate --check` probes them. Do not feed the result into continue/loop encode. Hardware-accelerated GPU scheduling recommended. Not required for generate."
            .into(),
    );
    notes.push(
        "RefMods (`h3 refmod`): pinned MIT ComfyUI-MiniMaxH3Mod v0.2.6 + packages comfy-workflow-h3-refmod-create / create-audio / create-master / t2va-refmod / ganloss-stage1-still-refmod. Saved VAE latents under checkpoints/refmods (GEMMY_H3_REFMODS). Not LoRAs; no extra weights. Generate with --ref-mod NAME. Audio extract is experimental (no speaker clone). Picker UI is not pinned."
            .into(),
    );
    notes.push(
        "Longer takes: h3 continue (native_guide keyframe tail; --ref-mod / --ref-image on Ref2VA; --legacy-masked-av freeze-prefix; --legacy-fl2va = WanGP last-RGB). Multi-scene: h3 loop. Deliverable SR: h3 upscale (RTX / LBH latent refine / Video2X / ffmpeg). Continue after latent refine at the new canvas; do not copy pixel SR or latent-preview into same-size masked_av. Continuous FL2VA speech uses ComfyUI-H3-Motion-Context v0.3.1 (GPL local clone) — not gemmy-h3-context, not a video-loop replacement. Keep Spectrum off on those speech graphs. Product Comfy is 0.36.0 so v0.6.2 is possible later; this pass does not bump that clone."
            .into(),
    );
    notes.push(
        "No disk weight offload; Comfy uses dynamic VRAM; python engine streams int8 RAM→VRAM.".into(),
    );
    notes.push(format!(
        "Code root is runtimes\\minimax-h3; weights root is {} (GEMMY_H3_CHECKPOINTS / model_paths.minimax_h3).",
        checkpoints_root().display()
    ));

    if comfy_worker_s.is_none() {
        blockers.push("default --engine comfy needs gemmy_h3_comfy_generate.py in runtime".into());
    }
    if comfy_root_s.is_none() {
        blockers.push(
            "default --engine comfy needs internalized runtimes\\minimax-h3\\ComfyUI\\run_h3_workflow.py"
                .into(),
        );
    }

    let ok = blockers.is_empty();
    let report = DoctorReport {
        ok,
        h3_root: root.display().to_string(),
        runtime_root: runtime_root().display().to_string(),
        checkpoints_root: checkpoints_root().display().to_string(),
        comfy_root: comfy_root_s,
        python,
        worker: worker_s,
        comfy_worker: comfy_worker_s,
        ffmpeg,
        assets_ok,
        assets,
        cuda,
        sage,
        fa2,
        comfy: comfy_probe,
        notes,
        blockers,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("[h3 doctor] h3_root={}", report.h3_root);
        println!(
            "[h3 doctor] checkpoints_root={}",
            report.checkpoints_root
        );
        println!(
            "[h3 doctor] python={}",
            report.python.as_deref().unwrap_or("<missing>")
        );
        println!(
            "[h3 doctor] worker(python)={}",
            report.worker.as_deref().unwrap_or("<missing>")
        );
        println!(
            "[h3 doctor] worker(comfy)={}",
            report.comfy_worker.as_deref().unwrap_or("<missing>")
        );
        println!(
            "[h3 doctor] comfy_root={}",
            report.comfy_root.as_deref().unwrap_or("<missing>")
        );
        println!(
            "[h3 doctor] ffmpeg={}",
            report.ffmpeg.as_deref().unwrap_or("<optional>")
        );
        if let Some(v) = &report.cuda {
            println!("[h3 doctor] cuda={v}");
        }
        if let Some(v) = &report.sage {
            println!("[h3 doctor] sage={v}");
        }
        if let Some(v) = &report.fa2 {
            println!("[h3 doctor] fa2={v}");
        }
        if let Some(v) = &report.comfy {
            println!("[h3 doctor] comfy={v}");
        }
        for n in &report.notes {
            println!("[h3 doctor] note: {n}");
        }
        for b in &report.blockers {
            println!("[h3 doctor] BLOCKER: {b}");
        }
        if report.ok {
            println!("[h3 doctor] OK");
        } else {
            println!("[h3 doctor] NOT READY");
        }
    }

    if !ok {
        anyhow::bail!("h3 doctor found blockers");
    }
    Ok(())
}

fn probe_python(python: &str, code: &str, verbose: bool) -> Value {
    // native_tool_command (no -I) so venv site-packages resolve.
    let mut cmd = crate::host::workers::env::native_tool_command(python);
    cmd.arg("-c").arg(code);
    match cmd.output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if verbose && !stderr.is_empty() {
                eprintln!("[h3 doctor] probe stderr: {stderr}");
            }
            if !out.status.success() {
                return serde_json::json!({
                    "ok": false,
                    "error": format!("exit {}", out.status),
                    "stderr": stderr,
                    "stdout": stdout,
                });
            }
            serde_json::from_str(&stdout).unwrap_or_else(|_| {
                serde_json::json!({"ok": false, "error": "non-json probe output", "stdout": stdout})
            })
        }
        Err(err) => serde_json::json!({"ok": false, "error": err.to_string()}),
    }
}
