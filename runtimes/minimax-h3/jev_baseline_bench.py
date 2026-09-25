"""Matched NO-SLA vs fixed native-SLA vs Jev bench. One case, six policies.

Does not invent timings. GPU generate is launched by the caller / `run-one` /
`run-campaign` subcommands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent
ROOT = RUNTIME.parents[1]
CASE_PATH = ROOT / "docs" / "jev-baseline" / "case.json"


def load_case() -> dict:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


def argv_for(policy: str, *, output: Path, dataset: Path | None, keep_work: bool = True) -> list[str]:
    sys.path.insert(0, str(RUNTIME))
    from jev_teacher import benchmark_commands

    case = load_case()
    row = next(c for c in benchmark_commands(case) if c["policy"] == policy)
    argv = list(row["argv"]) + ["-o", str(output), "--keep-work"]
    if not keep_work:
        argv.remove("--keep-work")
    if dataset is not None:
        argv.extend(["--jev-log-dataset", str(dataset)])
    return argv


def extract_frames(mp4: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    n = int((probe.stdout or "124").strip() or "124")
    picks = {
        "first": 0,
        "p25": max(0, n // 4),
        "p50": max(0, n // 2),
        "p75": max(0, (3 * n) // 4),
        "final": max(0, n - 1),
    }
    written = []
    for name, idx in picks.items():
        out = dest / f"{name}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4),
                "-vf",
                f"select=eq(n\\,{idx})",
                "-vframes",
                "1",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        written.append(out)
    return written


def parse_log_metrics(text: str) -> dict:
    out: dict = {}
    m = re.findall(r"(\d+)/4 \[(\d+):(\d+)<[^,]*,\s*([\d.]+)s/it\]", text)
    if m:
        last = m[-1]
        out["sampler_seconds"] = int(last[1]) * 60 + int(last[2])
        out["seconds_per_it"] = float(last[3])
    m = re.search(r"\[h3-comfy\] wrote .* in ([\d.]+)s", text)
    if m:
        out["comfy_wall_seconds"] = float(m.group(1))
    keeps = []
    controller_s = 0.0
    sla_n = 0
    fallback_n = 0
    for line in text.splitlines():
        idx = line.find("[009jev] ")
        if idx < 0:
            continue
        try:
            ev = json.loads(line[idx + 9 :].strip())
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "step" and "applied_keep_percent" in ev:
            keeps.append(float(ev["applied_keep_percent"]))
            att = ev.get("actual_attention") or {}
            sla_n += sum(1 for v in att.values() if v == "sla")
            fallback_n += sum(1 for v in att.values() if v != "sla")
        if ev.get("decision_seconds") is not None:
            controller_s += float(ev["decision_seconds"])
        if ev.get("event") == "end":
            out["jev_requests"] = ev.get("requests")
        if ev.get("event") == "begin":
            out["initial_policy"] = ev.get("initial_policy")
    if keeps:
        out["avg_keep_percent"] = sum(keeps) / len(keeps)
        out["per_step_keep_percent"] = keeps
    out["controller_seconds"] = controller_s
    out["actual_attention_sla"] = sla_n
    out["actual_attention_fallback"] = fallback_n
    out["log_has_009jev"] = "[009jev]" in text
    out["log_has_H3JevNativeSLAPatch"] = any(
        "H3JevNativeSLAPatch" in line and "no H3JevNativeSLAPatch" not in line
        for line in text.splitlines()
    )
    out["log_has_h3_sparse_attention"] = (
        "h3_sparse_attention" in text or "SparseAttnPatch" in text
    )
    return out


def _smi_poller(stop: threading.Event, samples: list[float], interval: float = 0.5) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                samples.append(float(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        stop.wait(interval)


def run_one(*, policy: str, rep: str, out_dir: Path, dataset: Path | None) -> dict:
    sys.path.insert(0, str(RUNTIME))
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / policy / f"{rep}.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    log_path = mp4.with_suffix(".log")
    argv = argv_for(policy, output=mp4, dataset=dataset if policy == "jev" else None)
    print(f"[bench] {policy} {rep}: {' '.join(argv)}", flush=True)
    stop = threading.Event()
    vram: list[float] = []
    t = threading.Thread(target=_smi_poller, args=(stop, vram), daemon=True)
    t.start()
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            logf.write(line)
            logf.flush()
            print(line, end="", flush=True)
        proc.wait()
    wall = time.perf_counter() - t0
    stop.set()
    t.join(timeout=2)
    log_text = "".join(chunks)
    metrics = parse_log_metrics(log_text)
    record = {
        "policy": policy,
        "rep": rep,
        "argv": argv,
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "output": str(mp4) if mp4.is_file() else None,
        "log": str(log_path),
        "peak_vram_mb": max(vram) if vram else None,
        "ok": proc.returncode == 0 and mp4.is_file() and mp4.stat().st_size > 0,
        **metrics,
    }
    if "CUDA" in log_text and ("out of memory" in log_text.lower() or "OOM" in log_text):
        record["failure"] = "cuda_oom"
    if "Access violation" in log_text or "access violation" in log_text.lower():
        record["failure"] = "access_violation"
    work_dirs = sorted(
        mp4.parent.glob(".h3_work_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if work_dirs:
        wf = work_dirs[0] / "comfy_workflow.json"
        record["work_dir"] = str(work_dirs[0])
        if wf.is_file():
            try:
                graph = json.loads(wf.read_text(encoding="utf-8"))
                classes = sorted(
                    {
                        n.get("class_type")
                        for n in graph.values()
                        if isinstance(n, dict) and n.get("class_type")
                    }
                )
                record["workflow_class_types"] = classes
                record["workflow_has_H3JevNativeSLAPatch"] = (
                    "H3JevNativeSLAPatch" in classes
                )
                record["workflow_has_KSampler"] = "KSampler" in classes
                dest = mp4.with_name(f"{rep}.workflow.json")
                dest.write_text(wf.read_text(encoding="utf-8"), encoding="utf-8")
                record["workflow_copy"] = str(dest)
            except Exception as exc:
                record["workflow_error"] = str(exc)
    if record["ok"]:
        try:
            frames = extract_frames(mp4, mp4.parent / f"{rep}_frames")
            record["frames"] = [str(p) for p in frames]
        except Exception as exc:
            record["frames_error"] = str(exc)
    (mp4.parent / f"{rep}.metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in ("policy", "rep", "ok", "wall_seconds", "sampler_seconds", "avg_keep_percent", "peak_vram_mb", "failure")}, indent=2), flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("print-commands")
    pr.add_argument("--out-dir", type=Path, default=Path("outputs/jev_baseline_bench"))
    one = sub.add_parser("run-one")
    one.add_argument("--policy", required=True)
    one.add_argument("--rep", required=True)
    one.add_argument("--out-dir", type=Path, default=Path("outputs/jev_baseline_bench"))
    one.add_argument("--dataset", type=Path, default=Path("datasets/h3_jev_teacher"))
    camp = sub.add_parser("run-campaign")
    camp.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/jev_nosla_bench"),
    )
    camp.add_argument(
        "--policies",
        nargs="+",
        default=["no-sla", "fixed-10", "fixed-5", "fixed-3", "fixed-1", "jev"],
    )
    camp.add_argument("--reps", nargs="+", default=["cold", "warm1", "warm2"])
    camp.add_argument("--dataset", type=Path, default=Path("datasets/h3_jev_teacher"))
    args = p.parse_args(argv)
    if args.cmd == "print-commands":
        sys.path.insert(0, str(RUNTIME))
        from jev_teacher import benchmark_commands

        for row in benchmark_commands(load_case()):
            print(json.dumps(row, indent=2))
        return 0
    if args.cmd == "run-one":
        rec = run_one(policy=args.policy, rep=args.rep, out_dir=args.out_dir, dataset=args.dataset)
        return 0 if rec.get("ok") else 1
    if args.cmd == "run-campaign":
        records = []
        summary_path = args.out_dir / "campaign.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for policy in args.policies:
            for rep in args.reps:
                rec = run_one(
                    policy=policy,
                    rep=rep,
                    out_dir=args.out_dir,
                    dataset=args.dataset if policy == "jev" else None,
                )
                records.append(rec)
                summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
                if not rec.get("ok"):
                    fail = args.out_dir / "gpu-unavailable.txt"
                    fail.write_text(
                        json.dumps(
                            {
                                "failed_policy": policy,
                                "failed_rep": rep,
                                "returncode": rec.get("returncode"),
                                "failure": rec.get("failure"),
                                "log": rec.get("log"),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    return 1
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
