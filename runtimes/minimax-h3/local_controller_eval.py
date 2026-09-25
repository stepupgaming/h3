"""Zero-shot local-controller eval: JSONL load, representations, baselines, metrics.

CPU only. No GLiNER, Needle, torch, or H3 generate.
Primary target is ``jev_raw_choice``, never ``applied_keep``.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

KEEP = (1, 3, 5, 10)
KEEP_SET = set(KEEP)
ORDINAL = {1: 0, 3: 1, 5: 2, 10: 3}

# Documented in docs/local-controller/JEV_TASK.md. Telemetry-only; no Jev labels.
RESIDUAL_CUTS = (0.10, 0.32, 0.67, 0.85)
RESIDUAL_NAMES = ("very_low", "low", "medium", "high", "very_high")
CROSS_CUTS = (0.55, 0.79, 0.97, 1.13)
CROSS_NAMES = ("very_low", "low", "medium", "high", "very_high")
RANK_CUTS = (0.20, 0.40, 0.60, 0.80)
RANK_NAMES = ("very_low", "low", "mid", "high", "very_high")

LATER_CRITERIA = {
    1: "Aggressive: low impact in both modalities",
    3: "Moderate reduction: relatively low effect or stable",
    5: "Baseline: mixed/ordinary evidence",
    10: "Protect: unusually high importance or risk",
}
FIRST_CRITERIA = {
    1: "Strong reduction with compelling low-impact evidence",
    3: "Moderate reduction with supporting evidence",
    5: "Intermediate coverage",
    10: "Highest allowed coverage for formation, text or uncertainty",
}
FIRST_BLOCK0_CRITERIA = {
    5: "Moderate coverage",
    10: "Highest allowed coverage",
}

LEAK_PATTERNS = (
    "jev_raw_choice",
    "jev_choice",
    "applied_keep",
    "applied keep",
    "fallback_triggered",
    "confidence_floor",
)


def default_paths(root: Path | None = None) -> tuple[Path, Path]:
    base = root or Path(__file__).resolve().parents[2]
    d = base / "datasets" / "h3_jev_teacher"
    return d / "generations.jsonl", d / "layer_decisions.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _as_keep(value: Any) -> int | None:
    if value is None:
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n in KEEP_SET else None


def _mod(row: dict[str, Any], key: str) -> dict[str, Any]:
    d = row.get(key)
    return d if isinstance(d, dict) else {}


def canonical_record(row: dict[str, Any], *, prompt: str = "") -> dict[str, Any]:
    audio = _mod(row, "audio")
    video = _mod(row, "video")
    rec = {
        "sample_id": f"{row.get('generation_id')}:s{row.get('step')}:l{row.get('layer')}",
        "generation_id": row.get("generation_id"),
        "step": int(row["step"]),
        "layer": int(row["layer"]),
        "layer_depth": row.get("layer_depth"),
        "previous_keep": _as_keep(row.get("previous_keep")),
        "audio_residual_relative_l2": audio.get("residual_relative_l2"),
        "audio_rank": audio.get("rank"),
        "audio_cross_step_change": audio.get("cross_step_change"),
        "video_residual_relative_l2": video.get("residual_relative_l2"),
        "video_rank": video.get("rank"),
        "video_cross_step_change": video.get("cross_step_change"),
        "jev_choice": _as_keep(row.get("jev_raw_choice")),
        "jev_confidence": row.get("jev_confidence"),
        "jev_probabilities": row.get("jev_probabilities"),
        "applied_keep": _as_keep(row.get("applied_keep")),
        "fallback_triggered": bool(row.get("fallback_triggered")),
        "fallback_reason": row.get("fallback_reason"),
        "quality_tag": row.get("quality_tag"),
        "prompt": prompt,
        "num_steps": int(row.get("num_steps") or 4),
    }
    return rec


def load_dataset(
    *,
    generations_path: Path | None = None,
    decisions_path: Path | None = None,
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpath, dpath = default_paths(root)
    if generations_path is not None:
        gpath = generations_path
    if decisions_path is not None:
        dpath = decisions_path
    gens = load_jsonl(gpath)
    prompts = {g["generation_id"]: g.get("prompt") or "" for g in gens}
    rows = []
    for raw in load_jsonl(dpath):
        rows.append(canonical_record(raw, prompt=prompts.get(raw.get("generation_id"), "")))
    return gens, rows


def is_eligible(rec: dict[str, Any]) -> bool:
    return rec.get("jev_choice") in KEEP_SET


def is_block0_later(rec: dict[str, Any]) -> bool:
    return rec.get("fallback_reason") == "block0_later_keep5" or (
        rec.get("layer") == 0
        and int(rec.get("step") or 0) > 1
        and rec.get("jev_choice") is None
    )


def eligible_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if is_eligible(r)]


def excluded_null_raw(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not is_eligible(r)]


def _bin(value: Any, cuts: tuple[float, ...], names: tuple[str, ...], *, missing: str = "missing") -> str:
    if value is None:
        return missing
    x = float(value)
    for cut, name in zip(cuts, names):
        if x < cut:
            return name
    return names[-1]


def residual_bin(value: Any) -> str:
    return _bin(value, RESIDUAL_CUTS, RESIDUAL_NAMES)


def cross_bin(value: Any) -> str:
    return _bin(value, CROSS_CUTS, CROSS_NAMES)


def rank_bin(value: Any) -> str:
    return _bin(value, RANK_CUTS, RANK_NAMES)


def disagreement_bin(audio_rank: Any, video_rank: Any) -> str:
    if audio_rank is None or video_rank is None:
        return "missing"
    d = abs(float(audio_rank) - float(video_rank))
    if d < 0.20:
        return "low"
    if d < 0.40:
        return "mid"
    return "high"


def depth_bin(depth: Any) -> str:
    if depth is None:
        return "missing"
    x = float(depth)
    if x < 0.33:
        return "early"
    if x < 0.67:
        return "mid"
    return "late"


def class_criteria(rec: dict[str, Any]) -> dict[int, str]:
    if int(rec["step"]) == 1:
        if int(rec["layer"]) == 0:
            return FIRST_BLOCK0_CRITERIA
        return FIRST_CRITERIA
    return LATER_CRITERIA


def fmt_num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    return f"{float(value):.{digits}g}"


def representation_raw(rec: dict[str, Any], *, include_prompt: bool = False) -> str:
    n_layers = 50
    lines = [
        f"Layer {rec['layer']} of {n_layers}",
        f"Diffusion step {rec['step']} of {rec['num_steps']}",
        f"Previous attention keep: {rec['previous_keep'] if rec['previous_keep'] is not None else 'none'}%",
        "",
        "Audio:",
        f"residual_relative_l2 = {fmt_num(rec['audio_residual_relative_l2'])}",
        f"rank = {fmt_num(rec['audio_rank'], 4)}",
        f"cross_step_change = {fmt_num(rec['audio_cross_step_change'])}",
        "",
        "Video:",
        f"residual_relative_l2 = {fmt_num(rec['video_residual_relative_l2'])}",
        f"rank = {fmt_num(rec['video_rank'], 4)}",
        f"cross_step_change = {fmt_num(rec['video_cross_step_change'])}",
    ]
    if include_prompt and rec.get("prompt"):
        lines.extend(["", "Generation prompt:", str(rec["prompt"])[:1200]])
    return "\n".join(lines)


def representation_derived(rec: dict[str, Any], *, include_prompt: bool = False) -> str:
    ar, vr = rec["audio_rank"], rec["video_rank"]
    lines = [
        f"Layer {rec['layer']} of 50 (depth {fmt_num(rec['layer_depth'], 4)} = {depth_bin(rec['layer_depth'])})",
        f"Diffusion step {rec['step']} of {rec['num_steps']}",
        f"Previous keep = {rec['previous_keep'] if rec['previous_keep'] is not None else 'none'}%",
        f"audio residual bin = {residual_bin(rec['audio_residual_relative_l2'])} (value {fmt_num(rec['audio_residual_relative_l2'])})",
        f"audio rank bin = {rank_bin(ar)} (value {fmt_num(ar, 4)})",
        f"audio change bin = {cross_bin(rec['audio_cross_step_change'])} (value {fmt_num(rec['audio_cross_step_change'])})",
        f"video residual bin = {residual_bin(rec['video_residual_relative_l2'])} (value {fmt_num(rec['video_residual_relative_l2'])})",
        f"video rank bin = {rank_bin(vr)} (value {fmt_num(vr, 4)})",
        f"video change bin = {cross_bin(rec['video_cross_step_change'])} (value {fmt_num(rec['video_cross_step_change'])})",
        f"audio/video rank disagreement = {disagreement_bin(ar, vr)}",
    ]
    if include_prompt and rec.get("prompt"):
        lines.extend(["Generation prompt:", str(rec["prompt"])[:1200]])
    return "\n".join(lines)


def representation_semantic(rec: dict[str, Any], *, include_prompt: bool = False) -> str:
    step = int(rec["step"])
    layer = int(rec["layer"])
    if step == 1:
        bits = [
            f"Layer {layer}/50, diffusion step 1 of {rec['num_steps']} (cold start).",
            "No current-run residual telemetry exists yet.",
            "This pass establishes scene and identity from the prompt.",
            "Historical low residual is not proof that a layer is safe to strip.",
        ]
        if layer == 0:
            bits.append("Block 0 may only use 5 or 10.")
        bits.append("Prefer broad coverage; reduce only with a specific defensible low-impact signal.")
    else:
        ar, vr = rec["audio_rank"], rec["video_rank"]
        a_res, v_res = residual_bin(rec["audio_residual_relative_l2"]), residual_bin(
            rec["video_residual_relative_l2"]
        )
        a_rk, v_rk = rank_bin(ar), rank_bin(vr)
        both_low = a_rk in ("very_low", "low") and v_rk in ("very_low", "low")
        either_high = a_rk in ("high", "very_high") or v_rk in ("high", "very_high")
        drift = cross_bin(rec["audio_cross_step_change"]) in ("high", "very_high") or cross_bin(
            rec["video_cross_step_change"]
        ) in ("high", "very_high")
        missing_hist = rec["audio_cross_step_change"] is None and rec["video_cross_step_change"] is None
        bits = [
            f"Layer {layer}/50, diffusion step {step} of {rec['num_steps']}.",
            f"The layer is in the {depth_bin(rec['layer_depth'])} third of the transformer.",
            f"Video residual importance is {v_res}; peer rank is {v_rk}.",
            f"Audio residual importance is {a_res}; peer rank is {a_rk}.",
        ]
        if missing_hist:
            bits.append("Temporal history is missing; missing history is not proof of danger.")
        else:
            bits.append(
                f"Audio change vs last step is {cross_bin(rec['audio_cross_step_change'])}; "
                f"video change is {cross_bin(rec['video_cross_step_change'])}."
            )
        bits.append(f"Audio and video rank disagreement is {disagreement_bin(ar, vr)}.")
        if both_low and not either_high:
            bits.append("Both modalities show relatively low peer rank.")
        if either_high:
            bits.append("At least one modality shows high peer rank.")
        if drift:
            bits.append("Cross-step drift is large on at least one modality (includes normal denoising).")
        bits.append("Preserve the worse modality.")
        if rec["previous_keep"] is not None:
            bits.append(f"Previous keep was {rec['previous_keep']}%.")
    if include_prompt and rec.get("prompt"):
        bits.append("Prompt: " + str(rec["prompt"])[:1200])
    return "\n".join(bits)


REPRESENTATIONS = {
    "raw": representation_raw,
    "derived": representation_derived,
    "semantic": representation_semantic,
}


def build_input(
    rec: dict[str, Any],
    style: str,
    *,
    include_prompt: bool = False,
    ablation: str | None = None,
) -> str:
    rec2 = dict(rec)
    if ablation == "no_audio":
        rec2["audio_residual_relative_l2"] = None
        rec2["audio_rank"] = None
        rec2["audio_cross_step_change"] = None
    elif ablation == "no_video":
        rec2["video_residual_relative_l2"] = None
        rec2["video_rank"] = None
        rec2["video_cross_step_change"] = None
    elif ablation == "no_cross_step":
        rec2["audio_cross_step_change"] = None
        rec2["video_cross_step_change"] = None
    elif ablation == "no_rank":
        rec2["audio_rank"] = None
        rec2["video_rank"] = None
    elif ablation == "no_previous_keep":
        rec2["previous_keep"] = None
    elif ablation == "layer_step_only":
        rec2["audio_residual_relative_l2"] = None
        rec2["audio_rank"] = None
        rec2["audio_cross_step_change"] = None
        rec2["video_residual_relative_l2"] = None
        rec2["video_rank"] = None
        rec2["video_cross_step_change"] = None
        rec2["previous_keep"] = None
        rec2["prompt"] = ""
        include_prompt = False
    fn = REPRESENTATIONS[style]
    return fn(rec2, include_prompt=include_prompt)


def input_leaks_label(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in LEAK_PATTERNS):
        return True
    if re.search(r"jev[_\s-]?raw", low):
        return True
    return False


def always_predict(keep: int, records: Iterable[dict[str, Any]]) -> list[int]:
    return [keep for _ in records]


def majority_class(records: list[dict[str, Any]]) -> int:
    c = Counter(int(r["jev_choice"]) for r in records)
    # Deterministic: highest count, then more conservative keep.
    return sorted(c.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]


def layer_step_lookup_logo(
    records: list[dict[str, Any]],
) -> list[int]:
    """Leave-one-generation-out majority (layer, step). Tie → larger keep."""
    by_gen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_gen[str(r["generation_id"])].append(r)
    pred_map: dict[str, int] = {}
    for hold in by_gen:
        train = [r for gid, rows in by_gen.items() if gid != hold for r in rows]
        table: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r in train:
            table[(int(r["layer"]), int(r["step"]))][int(r["jev_choice"])] += 1
        for r in by_gen[hold]:
            counts = table.get((int(r["layer"]), int(r["step"])))
            if not counts:
                choice = 10
            else:
                choice = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
            pred_map[r["sample_id"]] = choice
    return [pred_map[r["sample_id"]] for r in records]


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, dict[str, int]]:
    mat = {str(a): {str(b): 0 for b in KEEP} for a in KEEP}
    for t, p in zip(y_true, y_pred):
        mat[str(t)][str(p)] += 1
    return mat


def per_class_recall(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    out = {}
    for k in KEEP:
        idx = [i for i, t in enumerate(y_true) if t == k]
        if not idx:
            out[str(k)] = float("nan")
            continue
        hit = sum(1 for i in idx if y_pred[i] == k)
        out[str(k)] = hit / len(idx)
    return out


def per_class_precision(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    out = {}
    for k in KEEP:
        idx = [i for i, p in enumerate(y_pred) if p == k]
        if not idx:
            out[str(k)] = 0.0
            continue
        hit = sum(1 for i in idx if y_true[i] == k)
        out[str(k)] = hit / len(idx)
    return out


def f1_from_pr(p: float, r: float) -> float:
    if math.isnan(p):
        p = 0.0
    if math.isnan(r):
        r = 0.0
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def class_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    rec = per_class_recall(y_true, y_pred)
    prec = per_class_precision(y_true, y_pred)
    return {str(k): f1_from_pr(prec[str(k)], rec[str(k)]) for k in KEEP}


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Unweighted mean F1 over {1,3,5,10}. Missing/never-predicted class F1 is 0."""
    f1s = class_f1(y_true, y_pred)
    return sum(f1s[str(k)] for k in KEEP) / len(KEEP)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    rec = per_class_recall(y_true, y_pred)
    finite = [v for v in rec.values() if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def ordinal_errors(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    deltas = [ORDINAL[p] - ORDINAL[t] for t, p in zip(y_true, y_pred)]
    n = len(deltas) or 1
    hist = Counter(deltas)
    return {
        "mean_delta_pred_minus_true": sum(deltas) / n,
        "mean_abs_delta": sum(abs(d) for d in deltas) / n,
        "n_more_aggressive": sum(1 for d in deltas if d < 0),
        "n_more_conservative": sum(1 for d in deltas if d > 0),
        "n_exact": sum(1 for d in deltas if d == 0),
        "n_true10_pred1": sum(1 for t, p in zip(y_true, y_pred) if t == 10 and p == 1),
        "n_true1_pred10": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 10),
        "n_true10_pred5": sum(1 for t, p in zip(y_true, y_pred) if t == 10 and p == 5),
        "delta_histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def exact_agreement(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("nan")
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


def metrics_bundle(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    y = [int(r["jev_choice"]) for r in records]
    assert len(y) == len(preds)
    rec = per_class_recall(y, preds)
    return {
        "n": len(y),
        "exact_agreement": exact_agreement(y, preds),
        "balanced_accuracy": balanced_accuracy(y, preds),
        "macro_f1": macro_f1(y, preds),
        "per_class_f1": class_f1(y, preds),
        "per_class_recall": rec,
        "per_class_precision": per_class_precision(y, preds),
        "confusion": confusion(y, preds),
        "ordinal": ordinal_errors(y, preds),
        "true_histogram": {str(k): sum(1 for t in y if t == k) for k in KEEP},
        "pred_histogram": {str(k): sum(1 for p in preds if p == k) for k in KEEP},
    }


def layer_analysis(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    """Per-layer exact/error on aligned records. Layers 0–16 / 17–33 / 34–49."""
    assert len(records) == len(preds)
    per: dict[str, Any] = {}
    for layer in range(50):
        idx = [i for i, r in enumerate(records) if int(r["layer"]) == layer]
        if not idx:
            continue
        recs = [records[i] for i in idx]
        yhat = [preds[i] for i in idx]
        bundle = metrics_bundle(recs, yhat)
        y = [int(r["jev_choice"]) for r in recs]
        per[str(layer)] = {
            "n": len(idx),
            "exact_agreement": bundle["exact_agreement"],
            "n_true10_pred1": sum(1 for t, p in zip(y, yhat) if t == 10 and p == 1),
            "n_true1_pred10": sum(1 for t, p in zip(y, yhat) if t == 1 and p == 10),
            "mean_true_keep": sum(y) / len(y),
            "mean_pred_keep": sum(yhat) / len(yhat),
            "frac_true_10": sum(1 for t in y if t == 10) / len(y),
            "frac_pred_10": sum(1 for p in yhat if p == 10) / len(yhat),
        }

    def _band(lo: int, hi: int) -> dict[str, Any]:
        idx = [i for i, r in enumerate(records) if lo <= int(r["layer"]) <= hi]
        if not idx:
            return {"n": 0}
        recs = [records[i] for i in idx]
        yhat = [preds[i] for i in idx]
        b = metrics_bundle(recs, yhat)
        y = [int(r["jev_choice"]) for r in recs]
        return {
            "n": len(idx),
            "layers": f"{lo}-{hi}",
            "exact_agreement": b["exact_agreement"],
            "macro_f1": b["macro_f1"],
            "frac_true_10": sum(1 for t in y if t == 10) / len(y),
            "frac_pred_10": sum(1 for p in yhat if p == 10) / len(yhat),
            "n_true10_pred1": sum(1 for t, p in zip(y, yhat) if t == 10 and p == 1),
        }

    always_correct = sorted(
        (int(k) for k, v in per.items() if v["exact_agreement"] == 1.0)
    )
    frequent_miss = sorted(
        (int(k) for k, v in per.items() if v["exact_agreement"] < 0.5),
        key=lambda L: per[str(L)]["exact_agreement"],
    )
    return {
        "per_layer": per,
        "always_correct_layers": always_correct,
        "frequent_miss_layers": frequent_miss,
        "bands": {
            "early_0_16": _band(0, 16),
            "mid_17_33": _band(17, 33),
            "late_34_49": _band(34, 49),
        },
    }


def failure_examples(
    records: list[dict[str, Any]], preds: list[int], n: int = 10
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    danger: list[dict[str, Any]] = []
    for rec, p in zip(records, preds):
        t = int(rec["jev_choice"])
        item = {
            "sample_id": rec["sample_id"],
            "generation_id": rec["generation_id"],
            "step": rec["step"],
            "layer": rec["layer"],
            "jev_choice": t,
            "jev_confidence": rec.get("jev_confidence"),
            "jev_probabilities": rec.get("jev_probabilities"),
            "prediction": int(p),
            "telemetry": {
                "audio_residual_relative_l2": rec.get("audio_residual_relative_l2"),
                "audio_rank": rec.get("audio_rank"),
                "audio_cross_step_change": rec.get("audio_cross_step_change"),
                "video_residual_relative_l2": rec.get("video_residual_relative_l2"),
                "video_rank": rec.get("video_rank"),
                "video_cross_step_change": rec.get("video_cross_step_change"),
                "previous_keep": rec.get("previous_keep"),
            },
        }
        if p == t:
            matches.append(item)
            continue
        item["ordinal_delta"] = ORDINAL[int(p)] - ORDINAL[t]
        bad.append(item)
        if (t == 10 and p == 1) or (t == 1 and p == 10):
            danger.append(item)
    bad.sort(key=lambda x: -abs(x.get("ordinal_delta", 0)))
    return {
        "strong_matches": matches[:n],
        "bad_disagreements": bad[:n],
        "dangerous": danger[:n],
        "n_matches": len(matches),
        "n_disagreements": len(bad),
        "n_true10_pred1": sum(1 for x in bad if x["jev_choice"] == 10 and x["prediction"] == 1),
        "n_true1_pred10": sum(1 for x in bad if x["jev_choice"] == 1 and x["prediction"] == 10),
    }


def align_predictions(
    records: list[dict[str, Any]], pred_rows: list[dict[str, Any]], style: str
) -> tuple[list[dict[str, Any]], list[int]]:
    by_id = {
        r["sample_id"]: r
        for r in pred_rows
        if r.get("input_representation") == style
    }
    recs_ok: list[dict[str, Any]] = []
    preds: list[int] = []
    for rec in records:
        pr = by_id.get(rec["sample_id"])
        if pr is None:
            continue
        p = parse_keep_label(pr.get("prediction"))
        if p not in KEEP_SET:
            continue
        recs_ok.append(rec)
        preds.append(int(p))
    return recs_ok, preds


def confidence_buckets() -> tuple[tuple[str, float | None], ...]:
    return (
        ("all", None),
        ("ge_0.50", 0.50),
        ("ge_0.70", 0.70),
        ("ge_0.80", 0.80),
        ("ge_0.90", 0.90),
    )


def stratified_confidence(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    out = {}
    for name, thresh in confidence_buckets():
        idx = []
        for i, r in enumerate(records):
            c = r.get("jev_confidence")
            if c is None:
                continue
            if thresh is None or float(c) >= thresh:
                idx.append(i)
        sub_r = [records[i] for i in idx]
        sub_p = [preds[i] for i in idx]
        out[name] = {
            "n": len(idx),
            **({} if not idx else metrics_bundle(sub_r, sub_p)),
        }
        if idx:
            # metrics_bundle already has n
            pass
        else:
            out[name]["exact_agreement"] = float("nan")
    return out


def stratified_step(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step in (1, 2, 3, 4):
        idx = [i for i, r in enumerate(records) if int(r["step"]) == step]
        out[f"step_{step}"] = metrics_bundle(
            [records[i] for i in idx], [preds[i] for i in idx]
        ) if idx else {"n": 0}
    later = [i for i, r in enumerate(records) if int(r["step"]) >= 2]
    out["steps_2_4"] = metrics_bundle(
        [records[i] for i in later], [preds[i] for i in later]
    ) if later else {"n": 0}
    return out


def stratified_generation(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    gens = []
    seen = []
    for r in records:
        gid = str(r["generation_id"])
        if gid not in seen:
            seen.append(gid)
    per = {}
    exacts = []
    bals = []
    for gid in seen:
        idx = [i for i, r in enumerate(records) if str(r["generation_id"]) == gid]
        bundle = metrics_bundle([records[i] for i in idx], [preds[i] for i in idx])
        per[gid] = bundle
        exacts.append(bundle["exact_agreement"])
        bals.append(bundle["balanced_accuracy"])
        gens.append(gid)
    return {
        "per_generation": per,
        "mean_exact_across_generations": sum(exacts) / len(exacts) if exacts else float("nan"),
        "mean_balanced_across_generations": sum(bals) / len(bals) if bals else float("nan"),
        "generation_ids": gens,
    }


def baseline_predictions(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    maj = majority_class(records)
    return {
        "always_10": always_predict(10, records),
        "always_5": always_predict(5, records),
        "majority": always_predict(maj, records),
        "layer_step_logo": layer_step_lookup_logo(records),
        "majority_class_value": [maj],
    }


def parse_keep_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("label") or value.get("keep") or value.get("choice")
    s = str(value).strip().replace("%", "")
    return _as_keep(s)


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    z = sum(e) or 1.0
    return [v / z for v in e]


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for k in KEEP:
        pk = max(float(p.get(str(k), 0.0)), 1e-12)
        qk = max(float(q.get(str(k), 0.0)), 1e-12)
        total += pk * math.log(pk / qk)
    return total


def brier(p: dict[str, float], y: int) -> float:
    s = 0.0
    for k in KEEP:
        t = 1.0 if k == y else 0.0
        s += (float(p.get(str(k), 0.0)) - t) ** 2
    return s


def dataset_counts(gens: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = Counter(str(r["jev_choice"]) for r in rows)
    applied = Counter(str(r["applied_keep"]) for r in rows)
    eligible = eligible_records(rows)
    nulls = excluded_null_raw(rows)
    return {
        "n_generations": len(gens),
        "generation_ids": [g.get("generation_id") for g in gens],
        "n_layer_rows": len(rows),
        "raw_choice_histogram": dict(raw),
        "applied_keep_histogram": dict(applied),
        "eligible_n": len(eligible),
        "null_raw_n": len(nulls),
        "block0_later_n": sum(1 for r in rows if is_block0_later(r)),
        "fallback_triggered_n": sum(1 for r in rows if r.get("fallback_triggered")),
        "confidence_floor_n": sum(1 for r in rows if r.get("fallback_reason") == "confidence_floor"),
    }
