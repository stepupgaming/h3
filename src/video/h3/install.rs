//! `h3 install|verify` — uv env under runtimes/minimax-h3 + presence checks.

use super::args::{H3InstallArgs, H3VerifyArgs};
use super::models::{all_required_ok, collect_asset_reports};
use super::paths::{
    checkpoints_root, comfy_root, h3_python, h3_root, runtime_root, worker_script,
};
use crate::host::config::H3Config;
use crate::host::util::run_simple_command;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::Path;
use std::process::Command;

#[derive(Serialize)]
struct InstallSummary {
    h3_root: String,
    runtime_root: String,
    checkpoints_root: String,
    python: String,
    worker: String,
    runtime_ready: bool,
    actions: Vec<String>,
    notes: Vec<String>,
    assets: Vec<super::models::AssetReport>,
}

pub(crate) fn run_install(args: H3InstallArgs, _config: &H3Config) -> Result<()> {
    let root = h3_root();
    let mut actions = Vec::new();
    let mut notes = Vec::new();

    if !root.is_dir() {
        bail!(
            "MiniMax-H3 runtime root not found: {}\n\
             Expected internalized code at runtimes\\minimax-h3 (or set GEMMY_H3_ROOT).",
            root.display()
        );
    }

    let want_runtime = args.runtime && !args.no_runtime;
    if want_runtime {
        if args.dry_run {
            actions.push(format!("dry-run: would install uv if needed and uv sync in {}", root.display()));
            actions.push("dry-run: would install ffmpeg if it is missing".into());
        } else {
            ensure_h3_uv_env(&root, args.verbose)?;
            actions.push("h3 runtime uv sync completed or already present".into());
            ensure_ffmpeg()?;
            actions.push("ffmpeg present".into());
            ensure_jev_sdk(args.verbose)?;
        }
    } else {
        notes.push("skipped runtime uv sync (--no-runtime)".into());
    }

    let comfy = comfy_root();
    ensure_product_nodes(&comfy, args.verbose, args.dry_run)?;
    actions.push(format!("product nodes under {}", comfy.join("custom_nodes").display()));

    super::download::fetch_sets(&["base", "eros", "latent", "face"], args.verbose, args.dry_run)?;
    actions.push(format!(
        "weights checked under {}",
        checkpoints_root().display()
    ));

    let worker = worker_script();
    if !worker.is_file() {
        bail!("H3 worker missing: {}", worker.display());
    }
    actions.push(format!("worker present: {}", worker.display()));

    if args.dry_run {
        actions.push(format!(
            "dry-run: would write {}",
            comfy.join("extra_model_paths.yaml").display()
        ));
    } else {
        match super::paths::write_comfy_extra_model_paths() {
            Ok(extra) => actions.push(format!(
                "wrote Comfy extra_model_paths {}",
                extra.display()
            )),
            Err(err) => notes.push(format!("extra_model_paths.yaml not written: {err:#}")),
        }
    }

    let assets = collect_asset_reports();
    let runtime_ready = all_required_ok(&assets);
    let summary = InstallSummary {
        h3_root: root.display().to_string(),
        runtime_root: runtime_root().display().to_string(),
        checkpoints_root: checkpoints_root().display().to_string(),
        python: h3_python().display().to_string(),
        worker: worker.display().to_string(),
        runtime_ready,
        actions,
        notes,
        assets,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("[h3 install] h3_root={}", summary.h3_root);
        println!(
            "[h3 install] checkpoints_root={}",
            summary.checkpoints_root
        );
        println!("[h3 install] python={}", summary.python);
        for a in &summary.actions {
            println!("[h3 install] {a}");
        }
        for n in &summary.notes {
            println!("[h3 install] note: {n}");
        }
        if summary.runtime_ready {
            println!("[h3 install] ready — all required assets present");
        } else if args.dry_run {
            println!("[h3 install] dry-run finished");
        } else {
            for r in summary.assets.iter().filter(|r| r.required && !r.ok) {
                println!("[h3 install] missing: {} @ {}", r.id, r.path);
            }
        }
    }

    if !summary.runtime_ready && !args.dry_run {
        bail!("h3 install did not finish the Python env or the required weight files");
    }
    Ok(())
}

pub(crate) fn run_verify(args: H3VerifyArgs, _config: &H3Config) -> Result<()> {
    let assets = collect_asset_reports();
    let ok = all_required_ok(&assets);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": ok,
                "h3_root": h3_root().display().to_string(),
                "checkpoints_root": checkpoints_root().display().to_string(),
                "assets": assets,
            }))?
        );
    } else {
        println!("[h3 verify] h3_root={}", h3_root().display());
        println!(
            "[h3 verify] checkpoints_root={}",
            checkpoints_root().display()
        );
        for r in &assets {
            let mark = if r.ok { "ok" } else { "FAIL" };
            println!("[h3 verify] [{mark}] {} — {}", r.id, r.path);
            if args.verbose && r.bytes.is_some() {
                println!("[h3 verify]   bytes={}", r.bytes.unwrap());
            }
        }
        if ok {
            println!("[h3 verify] all required assets present");
        } else {
            bail!("h3 verify failed — see missing assets above");
        }
    }
    if !ok {
        bail!("h3 verify failed");
    }
    Ok(())
}

pub(crate) fn ensure_h3_uv_env(root: &Path, verbose: bool) -> Result<()> {
    let py = root.join(r".venv\Scripts\python.exe");
    let pyproject = root.join("pyproject.toml");
    if !pyproject.is_file() {
        bail!(
            "H3 pyproject.toml missing at {} — expected runtimes\\minimax-h3 (or GEMMY_H3_ROOT).",
            pyproject.display()
        );
    }

    let uv = ensure_uv()?;
    let mut cmd = Command::new(&uv);
    cmd.arg("sync")
        .current_dir(root)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8");
    if verbose {
        eprintln!(
            "[h3 install] running: {} sync (cwd={})",
            uv.display(),
            root.display()
        );
    }
    run_simple_command(cmd, "h3 uv sync", verbose)
        .with_context(|| format!("uv sync in {}", root.display()))?;

    if !py.is_file() {
        bail!(
            "after uv sync, python still missing: {}",
            py.display()
        );
    }
    Ok(())
}

fn uv_candidates() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(raw) = std::env::var("UV") {
        let path = std::path::PathBuf::from(raw.trim());
        if !path.as_os_str().is_empty() {
            out.push(path);
        }
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let home = std::path::PathBuf::from(home);
        out.push(home.join(r".local\bin\uv.exe"));
        out.push(home.join(r"AppData\Roaming\uv\uv.exe"));
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        out.push(std::path::PathBuf::from(local).join(r"uv\uv.exe"));
    }
    out
}

fn which_uv() -> Option<std::path::PathBuf> {
    for path in uv_candidates() {
        if path.is_file() {
            return Some(path);
        }
    }
    let mut cmd = crate::host::workers::env::native_tool_command("where");
    cmd.arg("uv.exe");
    if let Ok(output) = cmd.output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = stdout.lines().next() {
            let t = line.trim();
            if !t.is_empty() && Path::new(t).is_file() {
                return Some(std::path::PathBuf::from(t));
            }
        }
    }
    None
}

fn ensure_uv() -> Result<std::path::PathBuf> {
    if let Some(path) = which_uv() {
        return Ok(path);
    }
    println!("[h3 install] uv is not installed. Installing it.");
    let mut cmd = crate::host::workers::env::native_tool_command("powershell.exe");
    cmd.args([
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "irm https://astral.sh/uv/install.ps1 | iex",
    ]);
    run_simple_command(cmd, "h3 uv install", true).context("install uv")?;
    which_uv().context("uv install finished but uv.exe was not found")
}

fn ensure_ffmpeg() -> Result<()> {
    if let Ok(path) = crate::host::paths::find_ffmpeg() {
        println!("[h3 install] ffmpeg={}", path.display());
        return Ok(());
    }
    println!("[h3 install] ffmpeg is not installed. Installing it.");
    let mut cmd = crate::host::workers::env::native_tool_command("winget");
    cmd.args([
        "install",
        "--id",
        "Gyan.FFmpeg",
        "-e",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]);
    let installed = run_simple_command(cmd, "h3 ffmpeg", true);
    if let Ok(path) = crate::host::paths::find_ffmpeg() {
        if installed.is_err() {
            println!("[h3 install] ffmpeg already on disk; winget had nothing to install");
        }
        println!("[h3 install] ffmpeg={}", path.display());
        return Ok(());
    }
    installed.context("install ffmpeg")?;
    bail!("ffmpeg is still missing after winget install");
}

fn ensure_jev_sdk(verbose: bool) -> Result<()> {
    let root = runtime_root().join("jev-sdk");
    if !root.join("pyproject.toml").is_file() {
        return Ok(());
    }
    let py = root.join(r".venv\Scripts\python.exe");
    if py.is_file() {
        println!("[h3 install] jev sdk python={}", py.display());
        return Ok(());
    }
    println!("[h3 install] uv sync {}", root.display());
    let uv = ensure_uv()?;
    let mut cmd = Command::new(&uv);
    cmd.arg("sync")
        .current_dir(&root)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8");
    run_simple_command(cmd, "h3 jev sdk", verbose)
        .with_context(|| format!("uv sync in {}", root.display()))?;
    if !py.is_file() {
        bail!("jev sdk python missing after uv sync: {}", py.display());
    }
    Ok(())
}

struct ProductNode {
    name: &'static str,
    repo: &'static str,
    sha: &'static str,
    marker: &'static str,
}

const PRODUCT_NODES: &[ProductNode] = &[
    ProductNode {
        name: "Comfyui_Minimax_h3_latent_Upscaler",
        repo: "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler",
        sha: "40316cf008b2fd8663263270669eb4da23f89d2c",
        marker: "nodes/minimax_h3_latent_upscaler_3d.py",
    },
    ProductNode {
        name: "ComfyUI-H3-FaceRefine",
        repo: "Carasibana/ComfyUI-H3-FaceRefine",
        sha: "d8521d14fe0d721d80cd9417fff5a559cbc21aba",
        marker: "__init__.py",
    },
];

const RUNNER_FILES: &[&str] = &["run_h3_workflow.py", "h3_workflow_build.py"];

pub(crate) fn bundled_comfy() -> std::path::PathBuf {
    crate::host::paths::runtime_path("minimax-h3").join("ComfyUI")
}

fn looks_like_comfy(path: &Path) -> bool {
    path.join("main.py").is_file()
}

/// Folder the user pointed at, or `ComfyUI` one level down (portable layout).
pub(crate) fn resolve_comfy_dir(path: &Path) -> Result<std::path::PathBuf> {
    if looks_like_comfy(path) {
        return Ok(path.to_path_buf());
    }
    let nested = path.join("ComfyUI");
    if looks_like_comfy(&nested) {
        return Ok(nested);
    }
    bail!(
        "{} is not a ComfyUI folder (no main.py).\n\
         Pass the folder that contains ComfyUI's main.py.\n\
         With no --comfy, h3 uses the bundled pack and h3 install fills that in.",
        path.display()
    )
}

pub(crate) fn prepare_comfy_folder(dest: &Path, verbose: bool, fetch_nodes: bool) -> Result<std::path::PathBuf> {
    let dest = resolve_comfy_dir(dest)?;
    let bundled = bundled_comfy();
    if !same_dir(&dest, &bundled) {
        copy_runner(&bundled, &dest)?;
        copy_shipped_nodes(&bundled, &dest)?;
    }
    if fetch_nodes {
        ensure_product_nodes(&dest, verbose, false)?;
    }
    Ok(dest)
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn copy_runner(bundled: &Path, dest: &Path) -> Result<()> {
    for name in RUNNER_FILES {
        let src = bundled.join(name);
        if !src.is_file() {
            bail!("bundled Comfy is missing {}", src.display());
        }
        let to = dest.join(name);
        std::fs::copy(&src, &to).with_context(|| format!("copy {} → {}", src.display(), to.display()))?;
        println!("[h3 setup] {} → {}", name, to.display());
    }
    Ok(())
}

fn copy_shipped_nodes(bundled: &Path, dest: &Path) -> Result<()> {
    let src_nodes = bundled.join("custom_nodes");
    let dest_nodes = dest.join("custom_nodes");
    std::fs::create_dir_all(&dest_nodes)
        .with_context(|| format!("create {}", dest_nodes.display()))?;
    let entries = std::fs::read_dir(&src_nodes)
        .with_context(|| format!("read {}", src_nodes.display()))?;
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str == "__pycache__" || name_str.starts_with('.') {
            continue;
        }
        let target = dest_nodes.join(&name);
        if target.exists() {
            continue;
        }
        copy_tree(&entry.path(), &target)?;
        println!("[h3 setup] node {} → {}", name_str, target.display());
    }
    Ok(())
}

fn copy_tree(src: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str == "__pycache__" || name_str == ".git" || name_str.ends_with(".pyc") {
            continue;
        }
        let to = dest.join(&name);
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)
                .with_context(|| format!("copy {}", entry.path().display()))?;
        }
    }
    Ok(())
}

pub(crate) fn ensure_product_nodes(comfy: &Path, verbose: bool, dry_run: bool) -> Result<()> {
    let nodes = comfy.join("custom_nodes");
    if !dry_run {
        std::fs::create_dir_all(&nodes).with_context(|| format!("create {}", nodes.display()))?;
    }
    for node in PRODUCT_NODES {
        let dest = nodes.join(node.name);
        if dest.join(node.marker).is_file() {
            println!("[h3 install] node present {}", dest.display());
            continue;
        }
        if dry_run {
            println!(
                "[h3 install] dry-run: would download {} @ {}",
                node.repo, node.sha
            );
            continue;
        }
        fetch_product_node(&dest, node, verbose)?;
    }
    Ok(())
}

fn fetch_product_node(dest: &Path, node: &ProductNode, _verbose: bool) -> Result<()> {
    let url = format!("https://github.com/{}/archive/{}.zip", node.repo, node.sha);
    let tmp_zip = std::env::temp_dir().join(format!("h3-{}-{}.zip", node.name, &node.sha[..12]));
    let extract = std::env::temp_dir().join(format!("h3-{}-{}", node.name, &node.sha[..12]));
    println!("[h3 install] downloading {} ({})", node.name, node.sha);
    download_url(&url, &tmp_zip)?;
    if extract.exists() {
        std::fs::remove_dir_all(&extract)?;
    }
    std::fs::create_dir_all(&extract)?;
    let status = Command::new("tar")
        .arg("-xf")
        .arg(&tmp_zip)
        .arg("-C")
        .arg(&extract)
        .status()
        .context("run tar to unpack the node pack (Windows tar.exe)")?;
    if !status.success() {
        bail!("tar exited {status} unpacking {}", tmp_zip.display());
    }
    let unpacked = single_child_dir(&extract)?;
    if dest.exists() {
        std::fs::remove_dir_all(dest)?;
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if std::fs::rename(&unpacked, dest).is_err() {
        copy_tree(&unpacked, dest)?;
        std::fs::remove_dir_all(&unpacked).ok();
    }
    let _ = std::fs::remove_file(&tmp_zip);
    let _ = std::fs::remove_dir_all(&extract);
    if !dest.join(node.marker).is_file() {
        bail!(
            "downloaded {} but {} is missing",
            node.name,
            dest.join(node.marker).display()
        );
    }
    println!("[h3 install] installed {}", dest.display());
    Ok(())
}

fn single_child_dir(root: &Path) -> Result<std::path::PathBuf> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    if dirs.len() == 1 {
        return Ok(dirs.remove(0));
    }
    bail!(
        "expected one folder inside {}, found {}",
        root.display(),
        dirs.len()
    )
}

fn download_url(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("build http client")?;
    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url}"))?;
    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("create {}", dest.display()))?;
    std::io::copy(&mut resp, &mut file).with_context(|| format!("write {}", dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_nodes_cover_latent_upscale_and_face_refine() {
        let names: Vec<&str> = PRODUCT_NODES.iter().map(|node| node.name).collect();
        assert!(names.contains(&"Comfyui_Minimax_h3_latent_Upscaler"));
        assert!(names.contains(&"ComfyUI-H3-FaceRefine"));
        assert!(PRODUCT_NODES.iter().all(|node| !node.sha.is_empty()));
        assert!(PRODUCT_NODES.iter().all(|node| node.marker.contains('.')));
    }

    #[test]
    fn empty_folder_is_not_a_comfy() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_comfy_dir(dir.path()).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("main.py"), "{text}");
    }

    #[test]
    fn portable_root_resolves_to_the_inner_comfy() {
        let dir = tempfile::tempdir().unwrap();
        let inner = dir.path().join("ComfyUI");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join("main.py"), b"print('comfy')\n").unwrap();
        let resolved = resolve_comfy_dir(dir.path()).unwrap();
        assert_eq!(resolved, inner);
    }

    #[test]
    fn dry_run_does_not_download_node_packs() {
        let dir = tempfile::tempdir().unwrap();
        ensure_product_nodes(dir.path(), false, true).unwrap();
        assert!(!dir
            .path()
            .join("custom_nodes")
            .join("Comfyui_Minimax_h3_latent_Upscaler")
            .exists());
    }

    #[test]
    fn copy_tree_skips_pycache() {
        let src = tempfile::tempdir().unwrap();
        let dest_root = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("a.py"), b"x").unwrap();
        std::fs::create_dir(src.path().join("__pycache__")).unwrap();
        std::fs::write(src.path().join("__pycache__").join("a.pyc"), b"y").unwrap();
        let dest = dest_root.path().join("pack");
        copy_tree(src.path(), &dest).unwrap();
        assert!(dest.join("a.py").is_file());
        assert!(!dest.join("__pycache__").exists());
    }

    #[test]
    fn copies_runner_scripts_into_an_external_comfy() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.py"), b"print('comfy')\n").unwrap();
        copy_runner(&bundled_comfy(), dir.path()).unwrap();
        assert!(dir.path().join("run_h3_workflow.py").is_file());
        assert!(dir.path().join("h3_workflow_build.py").is_file());
    }
}
