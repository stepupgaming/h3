"""Shared Comfy job helpers for generate / continue / loop / latent upscale."""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

VIDEO_VAE_INT8 = "minimax_h3_video_vae_int8_convrot.safetensors"
VIDEO_VAE_FP16 = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def video_vae_filename(req: dict[str, Any] | None = None, default: str = VIDEO_VAE_INT8) -> str:
    raw = ""
    if isinstance(req, dict):
        raw = str(req.get("video_vae") or "").strip()
    name = Path(raw).name if raw else default
    if not name.endswith(".safetensors"):
        name = f"{name}.safetensors"
    return name


def video_vae_fingerprint_id(filename: str) -> str:
    name = Path(filename).name
    if name.endswith(".safetensors"):
        return name[: -len(".safetensors")]
    return name


def bind_video_vae(params: dict[str, Any], req: dict[str, Any] | None = None) -> str:
    name = video_vae_filename(req)
    params["video_vae"] = name
    return name


def materialize_workflow(package: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind a compiled prompt.template.json (no Node, no IR compilation)."""
    root = Path(os.environ.get("GEMMY_COMFY_WORKFLOWS") or Path(__file__).resolve().parents[2] / "comfy-workflows")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from materialize import materialize_package  # type: ignore

    return materialize_package(package, params)


def force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_comfy_root(req: dict[str, Any], h3_root: Path) -> Path:
    candidates: list[Path] = []
    raw = req.get("comfy_root") or os.environ.get("GEMMY_H3_COMFY")
    if raw:
        candidates.append(Path(str(raw)))
    candidates.append(h3_root / "ComfyUI")
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "run_h3_workflow.py").is_file() and (c / "comfy").is_dir():
            return c
    raise SystemExit(
        "ComfyUI root not found under the internalized H3 runtime.\n"
        f"Expected: {h3_root / 'ComfyUI' / 'run_h3_workflow.py'}\n"
        "Dev override only: set GEMMY_H3_COMFY or request.comfy_root."
    )


def stage_file(src: Path, comfy_input: Path, dest_name: str) -> str:
    comfy_input.mkdir(parents=True, exist_ok=True)
    dest = comfy_input / dest_name
    src = src.resolve()
    if dest.exists():
        try:
            if dest.samefile(src):
                return dest_name
        except Exception:
            pass
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest_name


def find_latest(output_dir: Path, prefix: str, t_after: float, suffix: str) -> Path | None:
    prefix = prefix.replace("\\", "/").strip("/")
    candidates: list[Path] = []
    if not output_dir.is_dir():
        return None
    for p in output_dir.rglob(f"*{suffix}"):
        try:
            if p.stat().st_mtime + 0.5 < t_after:
                continue
        except OSError:
            continue
        rel = p.relative_to(output_dir).as_posix()
        if prefix in rel or p.stem.startswith(prefix.split("/")[-1]):
            candidates.append(p)
    if not candidates:
        for p in output_dir.rglob(f"*{suffix}"):
            try:
                if p.stat().st_mtime + 0.5 >= t_after:
                    candidates.append(p)
            except OSError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def sidecar_paths(mp4: Path) -> tuple[Path, Path]:
    return mp4.with_suffix(".h3av.safetensors"), mp4.with_suffix(".h3av.json")


def copy_sidecar_next_to(mp4: Path, st_src: Path | None) -> Path | None:
    if st_src is None or not st_src.is_file():
        return None
    dest_st, dest_json = sidecar_paths(mp4)
    shutil.copy2(st_src, dest_st)
    meta = st_src.with_suffix(".json")
    if meta.is_file():
        shutil.copy2(meta, dest_json)
    return dest_st


def existing_sidecar(mp4: Path) -> Path | None:
    st, _meta = sidecar_paths(mp4)
    return st if st.is_file() else None


def fingerprint_payload(
    *,
    width: int,
    height: int,
    frames: int,
    vae: str = "minimax_h3_video_vae_int8_convrot",
    dit: str = "",
    lora: list[str] | None = None,
    mode: str = "",
    context_frames: int = 39,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comfy = Path(__file__).resolve().parent / "ComfyUI"
    sys.path.insert(0, str(comfy / "custom_nodes" / "gemmy-h3-context"))
    from av_math import fingerprint  # type: ignore

    return fingerprint(
        width=width,
        height=height,
        frames=frames,
        vae=vae,
        dit=dit,
        lora=lora or [],
        mode=mode,
        context_frames=context_frames,
        extra=extra,
    )


def fingerprints_match(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    comfy = Path(__file__).resolve().parent / "ComfyUI"
    sys.path.insert(0, str(comfy / "custom_nodes" / "gemmy-h3-context"))
    from av_math import fingerprints_match as _match  # type: ignore

    return _match(a, b)


def load_fingerprint(st: Path) -> dict[str, Any] | None:
    meta = st.with_suffix(".json")
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def compile_ir_text(
    *,
    python: Path,
    h3_root: Path,
    prompt: str,
    ir_mode: str,
    frames: int,
    work: Path,
    realism: bool,
) -> tuple[str, float]:
    text = prompt
    if realism:
        trig = "r34l1sm"
        if not text.lstrip().lower().startswith(trig):
            text = f"{trig} {text.lstrip()}"
    t = time.perf_counter()
    ir_out = work / "prompt_ir.txt"
    ir_script = h3_root / "scripts" / "h3_prompt_ir.py"
    if ir_script.is_file():
        cmd = [
            str(python),
            str(ir_script),
            text,
            "--mode",
            ir_mode,
            "--duration",
            f"{frames / 24.0:.4f}",
            "-o",
            str(ir_out),
        ]
        print(f"[h3-comfy] prompt_ir: {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        proc = subprocess.run(cmd, cwd=str(h3_root), env=env)
        if proc.returncode != 0:
            raise SystemExit(f"prompt_ir failed with exit {proc.returncode}")
        text = ir_out.read_text(encoding="utf-8")
        if realism:
            trig = "r34l1sm"
            if not text.lstrip().lower().startswith(trig):
                text = f"{trig}\n{text.lstrip()}"
    else:
        print("[h3-comfy] warn: h3_prompt_ir.py missing; using raw prompt", flush=True)
    return text, time.perf_counter() - t


def comfy_python(python: Path, comfy_root: Path) -> Path:
    for cand in (
        comfy_root.parent / ".venv" / "Scripts" / "python.exe",
        comfy_root.parent / ".venv" / "bin" / "python",
        python,
    ):
        if cand.is_file():
            return cand
    return python


COMFY_SERVE_ENV = "GEMMY_H3_COMFY_SERVE"


def start_comfy_serve(
    *,
    comfy_root: Path,
    python: Path,
    attn: str,
    require_accel: bool = False,
    ready_timeout_s: float = 180.0,
) -> tuple[subprocess.Popen, str]:
    """Boot one headless Comfy process that executes many graphs."""
    runner = comfy_root / "run_h3_workflow.py"
    cmd = [
        str(comfy_python(python, comfy_root)),
        "-u",
        str(runner),
        "--serve",
        "--serve-bind",
        "127.0.0.1:0",
        "--attn",
        attn if attn != "none" else "sage",
    ]
    if require_accel:
        cmd.append("--require-accel")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    print(f"[h3-comfy] serve: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ready: queue.Queue[str] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("h3-comfy-serve tcp="):
                ready.put(line.strip().split("=", 1)[1].strip())

    threading.Thread(target=_pump, daemon=True).start()
    try:
        addr = ready.get(timeout=ready_timeout_s)
    except queue.Empty:
        proc.kill()
        raise SystemExit("comfy serve did not print h3-comfy-serve tcp= in time")
    if proc.poll() is not None:
        raise SystemExit(f"comfy serve exited {proc.returncode} before ready")
    print(f"[h3-comfy] serve ready at {addr}", flush=True)
    return proc, addr


def stop_comfy_serve(proc: subprocess.Popen, addr: str) -> None:
    try:
        _comfy_serve_rpc(addr, {"cmd": "quit"}, timeout_s=15.0)
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def _comfy_serve_rpc(addr: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    host, _, port_s = addr.rpartition(":")
    with socket.create_connection((host, int(port_s)), timeout=timeout_s) as sock:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise SystemExit("comfy serve closed the connection")
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))


def run_comfy_graph(
    *,
    comfy_root: Path,
    python: Path,
    graph: dict[str, Any],
    graph_path: Path,
    timing_path: Path,
    attn: str,
    require_accel: bool,
) -> tuple[Path, dict[str, Any], float]:
    write_json(graph_path, graph)
    addr = (os.environ.get(COMFY_SERVE_ENV) or "").strip()
    if addr:
        print(f"[h3-comfy] execute via serve {addr}: {graph_path}", flush=True)
        t0 = time.perf_counter()
        reply = _comfy_serve_rpc(
            addr,
            {"workflow": str(graph_path), "out_note": str(timing_path)},
            timeout_s=36000.0,
        )
        elapsed = time.perf_counter() - t0
        if not reply.get("ok"):
            raise SystemExit(f"comfy serve: {reply.get('error') or reply}")
        note = reply.get("note") if isinstance(reply.get("note"), dict) else {}
        outputs_dir = Path(str(note.get("outputs_dir") or (comfy_root / "output")))
        if timing_path.is_file():
            try:
                disk = json.loads(timing_path.read_text(encoding="utf-8"))
                if disk.get("outputs_dir"):
                    outputs_dir = Path(disk["outputs_dir"])
                note = disk
            except Exception:
                pass
        return outputs_dir, note, elapsed
    runner = comfy_root / "run_h3_workflow.py"
    cmd = [
        str(comfy_python(python, comfy_root)),
        "-u",
        str(runner),
        str(graph_path),
        "--out-note",
        str(timing_path),
        "--attn",
        attn if attn != "none" else "sage",
    ]
    if require_accel:
        cmd.append("--require-accel")
    print(f"[h3-comfy] execute: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(comfy_root), env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"comfy runner failed with exit {proc.returncode}")
    outputs_dir = comfy_root / "output"
    note: dict[str, Any] = {}
    if timing_path.is_file():
        try:
            note = json.loads(timing_path.read_text(encoding="utf-8"))
            if note.get("outputs_dir"):
                outputs_dir = Path(note["outputs_dir"])
        except Exception:
            pass
    return outputs_dir, note, elapsed


def _ffmpeg() -> str:
    raw = os.environ.get("GEMMY_FFMPEG")
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p)
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise SystemExit("ffmpeg not found (set GEMMY_FFMPEG or put ffmpeg on PATH)")


def assemble_mp4s(segs: list[Path], output: Path) -> None:
    """Concat already-trimmed H3 segments. Stream-copy first; re-encode if needed."""
    if not segs:
        raise SystemExit("no segments to assemble")
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(segs) == 1:
        if segs[0].resolve() != output.resolve():
            shutil.copy2(segs[0], output)
        return
    lst = output.with_suffix(".concat.txt")
    lines = []
    for seg in segs:
        p = str(seg.resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{p}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(output)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        cmd = [
            _ffmpeg(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "aac",
            str(output),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"ffmpeg concat failed for {output}")
    try:
        lst.unlink()
    except OSError:
        pass
