//! `h3 download` — choose a weight set and fetch it.
//!
//! Filenames and byte sizes are the Hugging Face tree listings for the repos
//! this CLI already loads. A set is skipped when the file on disk matches
//! that size.

use super::args::H3DownloadArgs;
use super::paths::{
    checkpoints_root, comfy_root, eros_checkpoints_root, singularity_checkpoints_root,
    stock_ref2va_backup_root, EROS_BF16_REPO, EROS_INT8_REPO, EROS_REF2VA_BF16, EROS_REF2VA_INT8,
    SINGULARITY_REF2VA_INT8,
};
use super::shortfilm::resume_download;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

const COMFY_ORG: &str = "Comfy-Org/MiniMax-H3";
const LABEL: &str = "h3 download";

#[derive(Clone, Copy)]
enum WeightRoot {
    Checkpoints,
    Eros,
    Stock,
    Singularity,
}

struct WeightFile {
    repo: &'static str,
    filename: &'static str,
    rel: &'static str,
    bytes: u64,
    root: WeightRoot,
}

struct WeightSet {
    id: &'static str,
    title: &'static str,
    about: &'static str,
    files: &'static [WeightFile],
    copy_tokenizer: bool,
}

fn catalog() -> &'static [WeightSet] {
    &[
        WeightSet {
            id: "base",
            title: "I2V / T2VA / FL2VA",
            about: "pruned int8 FL2VA DiT, NVFP4 text encoder, int8 video VAE, audio VAE, tokenizer copied from the bundled Comfy pack",
            copy_tokenizer: true,
            files: &[
                WeightFile {
                    repo: COMFY_ORG,
                    filename: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    rel: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    bytes: 20_970_379_616,
                    root: WeightRoot::Checkpoints,
                },
                WeightFile {
                    repo: COMFY_ORG,
                    filename: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    rel: "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    bytes: 15_687_142_551,
                    root: WeightRoot::Checkpoints,
                },
                WeightFile {
                    repo: COMFY_ORG,
                    filename: "minimax_h3_video_vae_int8_convrot.safetensors",
                    rel: "vae/minimax_h3_video_vae_int8_convrot.safetensors",
                    bytes: 2_811_065_184,
                    root: WeightRoot::Checkpoints,
                },
                WeightFile {
                    repo: COMFY_ORG,
                    filename: "minimax_h3_audio_vae_fp32.safetensors",
                    rel: "vae/minimax_h3_audio_vae_fp32.safetensors",
                    bytes: 605_254_808,
                    root: WeightRoot::Checkpoints,
                },
            ],
        },
        WeightSet {
            id: "vae-fp16",
            title: "fp16 video VAE",
            about: "optional official fp16 video VAE (`--vae fp16`)",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: COMFY_ORG,
                filename: "minimax_h3_video_vae_fp16.safetensors",
                rel: "vae/minimax_h3_video_vae_fp16.safetensors",
                bytes: 5_207_808_496,
                root: WeightRoot::Checkpoints,
            }],
        },
        WeightSet {
            id: "eros",
            title: "Eros INT8",
            about: "default Ref2VA / shortfilm DiT (INT8). Does not fetch the 40 GB BF16 file",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: EROS_INT8_REPO,
                filename: EROS_REF2VA_INT8,
                rel: "diffusion_models/10Eros_Max_h3_TURBO_ref2va_beta2_int8_convrot.safetensors",
                bytes: 20_973_099_200,
                root: WeightRoot::Eros,
            }],
        },
        WeightSet {
            id: "eros-bf16",
            title: "Eros BF16",
            about: "named 40 GB BF16 twin. Kept on disk. Not the 16 GB generate default",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: EROS_BF16_REPO,
                filename: EROS_REF2VA_BF16,
                rel: "diffusion_models/10Eros_Max_h3_TURBO_ref2va_beta2.safetensors",
                bytes: 40_228_444_088,
                root: WeightRoot::Eros,
            }],
        },
        WeightSet {
            id: "ref2va-stock",
            title: "stock Ref2VA",
            about: "Comfy-Org pruned int8 Ref2VA backup (`--weights`)",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: COMFY_ORG,
                filename: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                rel: "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                bytes: 20_970_379_616,
                root: WeightRoot::Stock,
            }],
        },
        WeightSet {
            id: "singularity",
            title: "Singularity",
            about: "full WarmBloodAban INT8 for `--dit singularity` (not the pruned sibling)",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "WarmBloodAban/Minimax-h3_Singularity",
                filename: SINGULARITY_REF2VA_INT8,
                rel: "diffusion_models/Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors",
                bytes: 34_004_507_622,
                root: WeightRoot::Singularity,
            }],
        },
        WeightSet {
            id: "turbo",
            title: "Turbo LoRA",
            about: "Larryvrh v4 EMA (`--turbo v4`). Not used on product Eros Ref2VA",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "larryvrh/MiniMax-H3-Turbo-Lora",
                filename: "minimax_h3_turbo_v4_step600_ema.safetensors",
                rel: "loras/minimax_h3_turbo_v4_step600_ema.safetensors",
                bytes: 779_849_816,
                root: WeightRoot::Checkpoints,
            }],
        },
        WeightSet {
            id: "realism",
            title: "Realism LoRA",
            about: "fal people LoRA (`--realism`, trigger r34l1sm)",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "fal/MiniMax-H3-Realism-People-LoRA",
                filename: "h3-realism-people-t2v-i2v-r2v.safetensors",
                rel: "loras/h3-realism-people-t2v-i2v-r2v.safetensors",
                bytes: 131_229_656,
                root: WeightRoot::Checkpoints,
            }],
        },
        WeightSet {
            id: "vsa",
            title: "VSA gate",
            about: "FastH3 gate for `--vsa`",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "barelymining/ComfyUI-MiniMax-H3-FastVideo",
                filename: "fasth3_vsa_gate.safetensors",
                rel: "loras/fasth3_vsa_gate.safetensors",
                bytes: 3_853_522_616,
                root: WeightRoot::Checkpoints,
            }],
        },
        WeightSet {
            id: "latent",
            title: "latent upscaler",
            about: "3D latent-upscaler weights for `h3 upscale --backend latent` (Apache-2.0)",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "LBH-123-AI/Minimax_h3_latent_Upscaler",
                filename: "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                rel: "latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                bytes: 690_592_992,
                root: WeightRoot::Checkpoints,
            }],
        },
        WeightSet {
            id: "face",
            title: "face detector",
            about: "YOLOv8 face weights for `h3 face-refine`",
            copy_tokenizer: false,
            files: &[WeightFile {
                repo: "Bingsu/adetailer",
                filename: "face_yolov8m.pt",
                rel: "ultralytics/bbox/face_yolov8m.pt",
                bytes: 52_026_019,
                root: WeightRoot::Checkpoints,
            }],
        },
    ]
}

fn root_dir(root: WeightRoot) -> PathBuf {
    match root {
        WeightRoot::Checkpoints => checkpoints_root(),
        WeightRoot::Eros => eros_checkpoints_root(),
        WeightRoot::Stock => stock_ref2va_backup_root(),
        WeightRoot::Singularity => singularity_checkpoints_root(),
    }
}

fn hf_url(repo: &str, filename: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{filename}?download=1")
}

fn bundled_tokenizer() -> PathBuf {
    crate::host::paths::runtime_path("minimax-h3")
        .join("ComfyUI")
        .join("comfy")
        .join("text_encoders")
        .join("qwen25_tokenizer")
}

pub(crate) fn tokenizer_dest() -> PathBuf {
    checkpoints_root()
        .join("text_encoders")
        .join("qwen25_tokenizer")
}

fn file_ready(path: &Path, bytes: u64) -> bool {
    std::fs::metadata(path).map(|m| m.len() == bytes).unwrap_or(false)
}

/// Canonical path, or the older on-disk name for the same latent-upscaler bytes.
fn existing_weight(file: &WeightFile) -> Option<PathBuf> {
    let dest = root_dir(file.root).join(file.rel);
    if file_ready(&dest, file.bytes) {
        return Some(dest);
    }
    if file
        .rel
        .ends_with("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")
    {
        let legacy = dest
            .parent()?
            .join("minimax_h3_latent_upscaler_3d_bf16.safetensors");
        if file_ready(&legacy, file.bytes) {
            return Some(legacy);
        }
    }
    None
}

#[derive(Serialize)]
pub(crate) struct FileStatus {
    repo: String,
    dest: String,
    bytes: u64,
    present: bool,
    url: String,
}

#[derive(Serialize)]
pub(crate) struct SetStatus {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) about: String,
    pub(crate) ready: bool,
    pub(crate) files: Vec<FileStatus>,
}

pub(crate) fn set_statuses() -> Vec<SetStatus> {
    catalog()
        .iter()
        .map(|set| {
            let mut files: Vec<FileStatus> = set
                .files
                .iter()
                .map(|file| {
                    let dest = root_dir(file.root).join(file.rel);
                    let present = existing_weight(file);
                    FileStatus {
                        repo: file.repo.to_string(),
                        dest: present
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| dest.display().to_string()),
                        bytes: file.bytes,
                        present: present.is_some(),
                        url: hf_url(file.repo, file.filename),
                    }
                })
                .collect();
            if set.copy_tokenizer {
                let dest = tokenizer_dest().join("vocab.json");
                files.push(FileStatus {
                    repo: "bundled Comfy pack".into(),
                    dest: tokenizer_dest().display().to_string(),
                    bytes: 0,
                    present: dest.is_file(),
                    url: String::new(),
                });
            }
            let ready = files.iter().all(|file| file.present);
            SetStatus {
                id: set.id.to_string(),
                title: set.title.to_string(),
                about: set.about.to_string(),
                ready,
                files,
            }
        })
        .collect()
}

fn print_catalog(json: bool) -> Result<()> {
    let sets = set_statuses();
    if json {
        println!("{}", serde_json::to_string_pretty(&sets)?);
        return Ok(());
    }
    println!("[h3 download] checkpoints={}", checkpoints_root().display());
    println!("[h3 download] eros={}", eros_checkpoints_root().display());
    println!(
        "[h3 download] ref2va-stock={}",
        stock_ref2va_backup_root().display()
    );
    println!(
        "[h3 download] singularity={}",
        singularity_checkpoints_root().display()
    );
    println!("[h3 download] sets: h3 download <name> [<name>…]   or   h3 download --dry-run <name>");
    for set in &sets {
        let mark = if set.ready { "present" } else { "missing" };
        let gb = set.files.iter().map(|f| f.bytes).sum::<u64>() as f64 / 1_000_000_000.0;
        println!(
            "[h3 download] {mark:7} {:<14} {gb:5.1} GB  {}",
            set.id, set.about
        );
    }
    Ok(())
}

pub(crate) fn copy_bundled_tokenizer() -> Result<bool> {
    let src = bundled_tokenizer();
    let dest = tokenizer_dest();
    if dest.join("vocab.json").is_file() && dest.join("merges.txt").is_file() {
        return Ok(false);
    }
    if !src.join("vocab.json").is_file() {
        bail!(
            "bundled tokenizer missing at {} (expected inside runtimes\\minimax-h3\\ComfyUI)",
            src.display()
        );
    }
    std::fs::create_dir_all(&dest).with_context(|| format!("create {}", dest.display()))?;
    for name in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
        let from = src.join(name);
        let to = dest.join(name);
        if from.is_file() {
            std::fs::copy(&from, &to)
                .with_context(|| format!("copy {} → {}", from.display(), to.display()))?;
        }
    }
    println!("[{LABEL}] copied tokenizer → {}", dest.display());
    Ok(true)
}

fn fetch_file(file: &WeightFile, verbose: bool, dry_run: bool) -> Result<()> {
    let dest = root_dir(file.root).join(file.rel);
    let url = hf_url(file.repo, file.filename);
    if let Some(found) = existing_weight(file) {
        println!("[{LABEL}] present {}", found.display());
        return Ok(());
    }
    if dry_run {
        println!(
            "[{LABEL}] dry-run {} / {} → {} ({} bytes)",
            file.repo,
            file.filename,
            dest.display(),
            file.bytes
        );
        return Ok(());
    }
    if dest.is_file() {
        let n = std::fs::metadata(&dest)?.len();
        println!(
            "[{LABEL}] {} is {n} bytes, expected {}; replacing",
            dest.display(),
            file.bytes
        );
        std::fs::remove_file(&dest)?;
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    println!(
        "[{LABEL}] {} / {} → {}",
        file.repo,
        file.filename,
        dest.display()
    );
    resume_download(&url, &dest, verbose, LABEL)
        .with_context(|| format!("download {} :: {}", file.repo, file.filename))?;
    let n = std::fs::metadata(&dest)?.len();
    if n != file.bytes {
        bail!(
            "downloaded {} is {n} bytes; Hugging Face lists {}",
            dest.display(),
            file.bytes
        );
    }
    println!("[{LABEL}] wrote {} ({n} bytes)", dest.display());
    Ok(())
}

pub(crate) fn run_download(args: H3DownloadArgs) -> Result<()> {
    if args.sets.is_empty() || args.list {
        return print_catalog(args.json);
    }
    let known: Vec<&str> = catalog().iter().map(|set| set.id).collect();
    let mut chosen = Vec::new();
    for name in &args.sets {
        let Some(set) = catalog().iter().find(|set| set.id == name) else {
            bail!(
                "unknown weight set `{name}`. Choose one of: {}",
                known.join(", ")
            );
        };
        chosen.push(set);
    }
    fetch_sets(
        &chosen.iter().map(|set| set.id).collect::<Vec<_>>(),
        args.verbose,
        args.dry_run,
    )?;
    if args.dry_run {
        return Ok(());
    }
    match super::paths::write_comfy_extra_model_paths() {
        Ok(path) => println!("[{LABEL}] wrote {}", path.display()),
        Err(err) => println!(
            "[{LABEL}] note: extra_model_paths.yaml not written ({err:#}). Comfy root: {}",
            comfy_root().display()
        ),
    }
    Ok(())
}

pub(crate) fn fetch_sets(ids: &[&str], verbose: bool, dry_run: bool) -> Result<()> {
    let known: Vec<&str> = catalog().iter().map(|set| set.id).collect();
    for name in ids {
        let Some(set) = catalog().iter().find(|set| set.id == *name) else {
            bail!(
                "unknown weight set `{name}`. Choose one of: {}",
                known.join(", ")
            );
        };
        println!("[{LABEL}] set {} — {}", set.id, set.about);
        if set.copy_tokenizer {
            if dry_run {
                println!(
                    "[{LABEL}] dry-run copy tokenizer → {}",
                    tokenizer_dest().display()
                );
            } else {
                copy_bundled_tokenizer()?;
            }
        }
        for file in set.files {
            fetch_file(file, verbose, dry_run)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_urls_are_huggingface() {
        let mut ids = Vec::new();
        for set in catalog() {
            assert!(set.files.iter().all(|f| f.bytes > 1_000_000), "{}", set.id);
            ids.push(set.id);
            for file in set.files {
                let url = hf_url(file.repo, file.filename);
                assert!(url.starts_with("https://huggingface.co/"), "{url}");
                assert!(url.contains(file.filename), "{url}");
            }
        }
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n);
        assert!(ids.contains(&"base"));
        assert!(ids.contains(&"eros"));
        assert!(ids.contains(&"eros-bf16"));
        assert!(ids.contains(&"latent"));
        assert!(ids.contains(&"face"));
    }
}
