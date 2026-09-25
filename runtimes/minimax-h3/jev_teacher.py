"""Parse 009jev native-SLA telemetry into a teacher JSONL dataset.

Preserves Jev raw belief separately from the keep H3 actually applied.
Stdlib only. Does not call Jev or load GPU models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

VALID_KEEP = {1, 3, 5, 10}
PROB_KEYS = ("1", "3", "5", "10")
N_LAYERS = 50
NUM_STEPS = 4  # author recipe default; HQ harvest infers N from events
SECRET_RE = re.compile(
    r"(TYPESAFE_API_KEY\s*=\s*\S+)|(\bsk-[A-Za-z0-9]{16,}\b)|(\bsk-proj-[A-Za-z0-9_-]{16,}\b)",
    re.IGNORECASE,
)


def prompt_id(prompt: str) -> str:
    return hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:12]


def parse_009jev_log(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        idx = line.find("[009jev] ")
        if idx < 0:
            continue
        payload = line[idx + 9 :].strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("event"):
            events.append(obj)
    return events


def load_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        events = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
        return events
    raw = json.loads(text)
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    raise ValueError(f"unsupported events file: {path}")


def _keep_int(value: Any) -> int:
    k = int(round(float(value)))
    if k not in VALID_KEEP:
        raise ValueError(f"keep {value!r} not in {sorted(VALID_KEEP)}")
    return k


def _prob_map(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    out = {k: float(raw.get(k, 0.0) or 0.0) for k in PROB_KEYS}
    return out


def _modality(block: dict[str, Any] | None, kind: str) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    row = block.get(kind)
    if not isinstance(row, dict):
        return None
    return {
        "residual_relative_l2": row.get("residual_relative_l2"),
        "rank": row.get("rank"),
        "cross_step_change": row.get("cross_step_change"),
    }


def _decision(answer: dict[str, Any] | None, layer: int) -> dict[str, Any] | None:
    if not isinstance(answer, dict):
        return None
    decisions = answer.get("decisions") or {}
    row = decisions.get(str(layer))
    if not isinstance(row, dict):
        return None
    return row


def records_from_events(
    events: list[dict[str, Any]],
    *,
    generation_id: str,
    prompt: str = "",
    seed: int | None = None,
    quality_tag: str = "clean",
    jev_model_default: str = "jev-1.13.0",
    source: str = "jev_teacher",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build generation metadata extras, layer rows, and per-call Jev request rows."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_name.setdefault(str(event.get("event")), []).append(event)

    initials = by_name.get("initial_decision") or []
    steps = sorted(by_name.get("step") or [], key=lambda e: int(e.get("step") or 0))
    begins = by_name.get("begin") or []
    ends = by_name.get("end") or []
    if not steps:
        raise ValueError(f"{generation_id}: no step events")
    begin_n = None
    if begins and begins[0].get("n_steps") is not None:
        begin_n = int(begins[0]["n_steps"])
    inferred = max(int(e.get("step") or 0) for e in steps)
    num_steps = begin_n or inferred
    if num_steps < 1:
        raise ValueError(f"{generation_id}: bad n_steps {num_steps}")
    if len(steps) != num_steps:
        raise ValueError(
            f"{generation_id}: expected {num_steps} step events, got {len(steps)}"
        )

    initial = initials[0] if initials else {}
    begin = begins[0] if begins else {}
    initial_policy = str(begin.get("initial_policy") or initial.get("initial_policy") or "jev_first")
    controller = "jev" if initial_policy == "jev_first" else initial_policy

    applied_by_step: dict[int, list[float]] = {}
    attention_by_step: dict[int, dict[str, str]] = {}
    answer_for_next: dict[int, dict[str, Any]] = {}
    telemetry_for_next: dict[int, dict[str, Any]] = {}
    seconds_for_next: dict[int, float] = {}
    model_for_next: dict[int, str] = {}
    calls: list[dict[str, Any]] = []

    if initial:
        applied = initial.get("applied_layer_keep_percent") or begin.get("first_step_keep")
        if applied:
            applied_by_step[1] = [float(x) for x in applied]
        if initial.get("answer"):
            answer_for_next[1] = initial["answer"]
            seconds_for_next[1] = float(initial.get("decision_seconds") or 0.0)
            model_for_next[1] = str((initial.get("answer") or {}).get("model") or jev_model_default)
            req = initial.get("jev_request") or _request_from_legacy(initial, n_questions=50, step=1)
            if req:
                calls.append(req)

    for event in steps:
        step = int(event["step"])
        applied_by_step[step] = [float(x) for x in event["applied_layer_keep_percent"]]
        attention_by_step[step] = {
            str(k): str(v) for k, v in (event.get("actual_attention") or {}).items()
        }
        if event.get("answer"):
            # This event's answer is the decision for the *next* step.
            answer_for_next[step + 1] = event["answer"]
            seconds_for_next[step + 1] = float(event.get("decision_seconds") or 0.0)
            model_for_next[step + 1] = str((event.get("answer") or {}).get("model") or jev_model_default)
            req = event.get("jev_request") or _request_from_legacy(
                event, n_questions=49, step=step + 1
            )
            if req:
                calls.append(req)
        if event.get("state"):
            telemetry_for_next[step + 1] = event["state"]

    pid = prompt_id(prompt)
    rows: list[dict[str, Any]] = []
    controller_seconds = sum(float(c.get("request_duration") or 0.0) for c in calls)

    for step in range(1, num_steps + 1):
        applied = applied_by_step.get(step)
        if not applied or len(applied) != N_LAYERS:
            raise ValueError(f"{generation_id}: step {step} missing 50 applied keeps")
        prev = applied_by_step.get(step - 1)
        answer = answer_for_next.get(step)
        blocks = (telemetry_for_next.get(step) or {}).get("blocks") or {}
        first_step = step == 1
        for layer in range(N_LAYERS):
            applied_keep = _keep_int(applied[layer])
            raw_row = _decision(answer, layer)
            raw_choice = None
            confidence = None
            probs = None
            if raw_row is not None:
                raw_choice = _keep_int(raw_row.get("choice"))
                confidence = float(raw_row["confidence"])
                probs = _prob_map(raw_row.get("probabilities"))
            fallback = False
            reason = None
            if first_step and layer == 0 and raw_choice is not None and raw_choice not in (5, 10):
                raise ValueError(f"{generation_id}: block0 first-step raw {raw_choice}")
            if not first_step and layer == 0 and raw_choice is None:
                reason = "block0_later_keep5"
            elif raw_choice is None and controller.startswith("const"):
                reason = "const"
            elif raw_choice is not None and raw_choice != applied_keep:
                fallback = True
                if confidence is not None and confidence < 0.3:
                    reason = "confidence_floor"
                else:
                    reason = "applied_differs"
            rows.append(
                {
                    "generation_id": generation_id,
                    "prompt_id": pid,
                    "seed": seed,
                    "step": step,
                    "num_steps": num_steps,
                    "layer": layer,
                    "layer_depth": round(layer / (N_LAYERS - 1), 4),
                    "previous_keep": None if prev is None else _keep_int(prev[layer]),
                    "audio": _modality(blocks.get(str(layer)), "audio") if blocks else None,
                    "video": _modality(blocks.get(str(layer)), "video") if blocks else None,
                    "jev_raw_choice": raw_choice,
                    "jev_confidence": confidence,
                    "jev_probabilities": probs,
                    "applied_keep": applied_keep,
                    "fallback_triggered": fallback,
                    "fallback_reason": reason,
                    "decision_seconds": seconds_for_next.get(step),
                    "jev_model": model_for_next.get(step) if raw_choice is not None else None,
                    "actual_attention": attention_by_step.get(step, {}).get(str(layer)),
                    "source": source,
                    "quality_tag": quality_tag,
                    "controller": controller,
                    "initial_policy": initial_policy,
                }
            )

    gen = {
        "generation_id": generation_id,
        "prompt": prompt,
        "prompt_id": pid,
        "seed": seed,
        "controller": controller,
        "controller_model": model_for_next.get(1) or (jev_model_default if controller == "jev" else None),
        "initial_policy": initial_policy,
        "steps": num_steps,
        "sampler": str(begin.get("sampler") or ("res_multistep" if num_steps == 4 else "euler")),
        "quality_tag": quality_tag,
        "source": source,
        "jev_requests": int((ends[0].get("requests") if ends else 0) or len(calls)),
        "controller_seconds": controller_seconds,
        "jev_calls": calls,
    }
    return gen, rows, calls


def _request_from_legacy(event: dict[str, Any], *, n_questions: int, step: int) -> dict[str, Any] | None:
    answer = event.get("answer")
    state = event.get("state")
    if not answer:
        return None
    usage = answer.get("usage") or {}
    raw_state_chars = None
    if state is not None:
        raw_state_chars = len(json.dumps(state, ensure_ascii=False))
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "raw_state_chars": raw_state_chars,
        "n_questions": n_questions,
        "step": step,
        "request_duration": event.get("decision_seconds"),
    }


def scan_secrets(obj: Any) -> list[str]:
    blob = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return [m.group(0) for m in SECRET_RE.finditer(blob)]


def validate_layer_rows(rows: list[dict[str, Any]], *, generation_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["no layer rows"]
    by_gen: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gid = str(row.get("generation_id") or "")
        if not gid:
            errors.append("missing generation_id")
            continue
        by_gen.setdefault(gid, []).append(row)
        leaks = scan_secrets(row)
        if leaks:
            errors.append(f"{gid}: secret-like substring {leaks[0]!r}")
        if row.get("applied_keep") not in VALID_KEEP:
            errors.append(f"{gid}: bad applied_keep {row.get('applied_keep')}")
        raw = row.get("jev_raw_choice")
        if raw is not None and raw not in VALID_KEEP:
            errors.append(f"{gid}: bad jev_raw_choice {raw}")
        step = row.get("step")
        row_n = int(row.get("num_steps") or NUM_STEPS)
        if not isinstance(step, int) or step not in range(1, row_n + 1):
            errors.append(f"{gid}: bad step {step} for num_steps={row_n}")
        layer = row.get("layer")
        if layer not in range(N_LAYERS):
            errors.append(f"{gid}: bad layer {layer}")
        probs = row.get("jev_probabilities")
        if probs is not None:
            missing = [k for k in PROB_KEYS if k not in probs]
            if missing:
                errors.append(f"{gid}: missing probability keys {missing}")
            else:
                total = sum(float(probs[k]) for k in PROB_KEYS)
                if not math.isclose(total, 1.0, abs_tol=0.05):
                    errors.append(f"{gid} L{layer} s{step}: probabilities sum {total}")
        if row.get("step") == 1 and row.get("layer") == 0 and raw is not None and raw not in (5, 10):
            errors.append(f"{gid}: block0 first-step raw {raw}")
        if (
            row.get("fallback_triggered")
            and raw is not None
            and row.get("applied_keep") == raw
        ):
            errors.append(f"{gid}: fallback_triggered but raw==applied")
    for gid, group in by_gen.items():
        if generation_id and gid != generation_id:
            continue
        n_here = int(group[0].get("num_steps") or NUM_STEPS)
        if len(group) != N_LAYERS * n_here:
            errors.append(f"{gid}: expected {N_LAYERS * n_here} rows, got {len(group)}")
        layers = {r["layer"] for r in group if r.get("step") == 1}
        if layers != set(range(N_LAYERS)):
            errors.append(f"{gid}: step 1 layers {sorted(layers)[:8]}...")
    return errors


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in records:
            leaks = scan_secrets(row)
            if leaks:
                raise ValueError(f"refusing to write secret-like data: {leaks[0]!r}")
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def benchmark_commands(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic matched-case argv for each policy. No GPU."""
    base = [
        "gemmy",
        "video",
        "h3",
        "generate",
        "--no-compile-ir",
        "--first-frame",
        str(case["first_frame"]),
        "--prompt-file",
        str(case["prompt_file"]),
        "--width",
        str(case["width"]),
        "--height",
        str(case["height"]),
        "--duration",
        str(case["duration"]),
        "--seed",
        str(case["seed"]),
        "--steps",
        str(case["steps"]),
    ]
    out = []
    for policy in case["policies"]:
        argv = list(base)
        if policy == "jev":
            argv.append("--jev")
        elif policy == "no-sla":
            argv.append("--no-sla")
        elif policy.startswith("fixed-"):
            argv.extend(["--sla-fixed", policy.split("-", 1)[1]])
        else:
            raise ValueError(policy)
        out.append({"policy": policy, "argv": argv})
    return out


def append_from_run(
    *,
    dataset_dir: Path,
    events: list[dict[str, Any]],
    generation: dict[str, Any],
) -> dict[str, Any]:
    gid = str(generation["generation_id"])
    gen, rows, _calls = records_from_events(
        events,
        generation_id=gid,
        prompt=str(generation.get("prompt") or ""),
        seed=generation.get("seed"),
        quality_tag=str(generation.get("quality_tag") or "clean"),
    )
    merged = {**gen, **generation, "generation_id": gid, "prompt_id": gen["prompt_id"]}
    errors = validate_layer_rows(rows, generation_id=gid)
    leaks = scan_secrets(merged)
    if leaks:
        errors.append(f"generation secret-like substring {leaks[0]!r}")
    if errors:
        raise ValueError("; ".join(errors[:8]))
    append_jsonl(dataset_dir / "generations.jsonl", [merged])
    append_jsonl(dataset_dir / "layer_decisions.jsonl", rows)
    return {"generation_id": gid, "layer_rows": len(rows), "dataset_dir": str(dataset_dir)}


def harvest_paths(
    *,
    dataset_dir: Path,
    generation_id: str,
    events_path: Path | None,
    log_path: Path | None,
    prompt: str,
    seed: int,
    quality_tag: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if events_path and events_path.is_file():
        if events_path.suffix.lower() == ".json" and events_path.name.endswith("_events.json"):
            events = load_events(events_path)
        elif events_path.suffix.lower() == ".jsonl":
            events = load_events(events_path)
        else:
            events = load_events(events_path)
    if not events and log_path and log_path.is_file():
        events = parse_009jev_log(log_path.read_text(encoding="utf-8", errors="replace"))
    if not events:
        raise FileNotFoundError(f"no 009jev events for {generation_id}")
    generation = {
        "generation_id": generation_id,
        "prompt": prompt,
        "seed": seed,
        "quality_tag": quality_tag,
        **extra,
    }
    return append_from_run(dataset_dir=dataset_dir, events=events, generation=generation)


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jev_teacher")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--dataset", type=Path, required=True)
    h.add_argument("--generation-id", required=True)
    h.add_argument("--events", type=Path)
    h.add_argument("--log", type=Path)
    h.add_argument("--prompt-file", type=Path)
    h.add_argument("--prompt", default="")
    h.add_argument("--seed", type=int, default=42)
    h.add_argument("--quality-tag", default="clean")
    h.add_argument("--output-video", default="")
    h.add_argument("--first-frame", default="")
    h.add_argument("--width", type=int)
    h.add_argument("--height", type=int)
    h.add_argument("--duration", type=float)
    h.add_argument("--model", default="")
    h.add_argument("--compile-ir", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "harvest":
        prompt = args.prompt
        if args.prompt_file:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        extra = {
            "output_video": args.output_video,
            "first_frame": args.first_frame,
            "width": args.width,
            "height": args.height,
            "duration": args.duration,
            "model": args.model,
            "compile_ir": bool(args.compile_ir),
        }
        summary = harvest_paths(
            dataset_dir=args.dataset,
            generation_id=args.generation_id,
            events_path=args.events,
            log_path=args.log,
            prompt=prompt,
            seed=args.seed,
            quality_tag=args.quality_tag,
            extra=extra,
        )
        print(json.dumps(summary, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
