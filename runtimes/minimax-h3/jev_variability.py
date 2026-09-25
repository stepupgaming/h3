"""Jev keep variability: pair stats, STATIC labels, lookups, hybrid budget.

CPU only. Target is jev_raw_choice. Later-step layer 0 null raw stays excluded.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Hashable, Iterable

from local_controller_eval import (
    KEEP,
    KEEP_SET,
    cross_bin,
    eligible_records,
    exact_agreement,
    layer_step_lookup_logo,
    metrics_bundle,
    rank_bin,
    residual_bin,
    stratified_confidence,
)

# Documented in docs/local-controller/JEV_CONTENT_ADAPTIVE.md
# STATIC/MOSTLY_STATIC labels only. Table *fill* does not use min_n.
VARIABILITY_THRESHOLDS = {
    "min_n": 5,
    "static": 0.95,
    "mostly_static": 0.80,
    "content_sensitive": 0.55,
}

# HQ table fill (majority_keep_table conservative=True). Independent of STATIC min_n.
# 1/3/5/10 are native-SLA attention keep ratios, not percent of model compute.
FILL_THRESHOLDS = {
    "high_conf": 0.7,
    "min_n_high_conf": 2,
    "high_conf_frac": 0.75,
    "min_n_all": 2,
    "all_frac": 0.75,
}

STATIC_LABELS = ("STATIC", "MOSTLY_STATIC")
VARIABLE_LABELS = ("CONTENT_SENSITIVE", "HIGHLY_VARIABLE")
LOOKUP_DEFAULT = 10
N_LAYERS = 50
N_STEPS = 4  # author 4-step recipe; HQ analysis infers N from records
LATER_LAYER0_KEEP = 5
DECISIONS_PER_GENERATION = 197  # 50 step-1 + 49×3 later eligible (4-step)


def infer_n_steps(records: Iterable[dict[str, Any]]) -> int:
    nums = [int(r["num_steps"]) for r in records if r.get("num_steps") is not None]
    if nums:
        return max(nums)
    steps = [int(r["step"]) for r in records if r.get("step") is not None]
    return max(steps) if steps else N_STEPS


def decisions_per_generation(n_steps: int) -> int:
    """Step-1 all 50 layers + later layers 1..49."""
    n = int(n_steps)
    return 50 + 49 * max(0, n - 1)

# Documented in docs/local-controller/JEV_VARIABLE_NOISE.md
NOISE_LABELS = ("REAL_DYNAMIC_SIGNAL", "AMBIGUOUS", "LIKELY_JEV_NOISE")
NOISE_THRESHOLDS = {
    "high_conf": 0.8,
    "mid_conf": 0.7,
    "low_conf": 0.5,
    "high_conf_min_n": 5,
    "mid_conf_min_n_strong": 8,
    "still_mixed_frac": 0.80,
    "collapsed_frac": 0.80,
    "collapsed_min_n": 3,
    "lift_real": 0.10,
    "lift_strong": 0.20,
}


def shannon_entropy(counts: Iterable[int]) -> float:
    vals = [int(c) for c in counts if int(c) > 0]
    n = sum(vals)
    if n <= 0:
        return 0.0
    ent = 0.0
    for c in vals:
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _majority(counter: Counter) -> tuple[int, float]:
    if not counter:
        return LOOKUP_DEFAULT, 0.0
    choice, n = sorted(counter.items(), key=lambda kv: (-kv[1], -kv[0]))[0]
    total = sum(counter.values())
    return int(choice), (n / total if total else 0.0)


def pair_cell_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Stats per (step, layer) on eligible rows only."""
    eligible = eligible_records(records)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in eligible:
        buckets[(int(rec["step"]), int(rec["layer"]))].append(rec)
    out: dict[str, dict[str, Any]] = {}
    n_steps = infer_n_steps(records)
    for step in range(1, n_steps + 1):
        for layer in range(50):
            rows = buckets.get((step, layer), [])
            counts = Counter(int(r["jev_choice"]) for r in rows)
            n = len(rows)
            maj, frac = _majority(counts)
            confs = [
                float(r["jev_confidence"])
                for r in rows
                if r.get("jev_confidence") is not None
            ]
            hist = {str(k): int(counts.get(k, 0)) for k in KEEP}
            high_thr = float(FILL_THRESHOLDS["high_conf"])
            high_rows = [
                r
                for r in rows
                if r.get("jev_confidence") is not None
                and float(r["jev_confidence"]) >= high_thr
            ]
            high_counts = Counter(int(r["jev_choice"]) for r in high_rows)
            high_n = len(high_rows)
            high_maj, high_frac = _majority(high_counts)
            high_hist = {str(k): int(high_counts.get(k, 0)) for k in KEEP}
            cell = {
                "step": step,
                "layer": layer,
                "n": n,
                "count_1": hist["1"],
                "count_3": hist["3"],
                "count_5": hist["5"],
                "count_10": hist["10"],
                "majority": maj if n else None,
                "majority_frac": frac if n else 0.0,
                "entropy": shannon_entropy(counts.values()) if n else 0.0,
                "mean_confidence": (sum(confs) / len(confs)) if confs else None,
                "high_conf_n": high_n,
                "high_conf_majority": high_maj if high_n else None,
                "high_conf_frac": high_frac if high_n else 0.0,
                "high_conf_count_1": high_hist["1"],
                "high_conf_count_3": high_hist["3"],
                "high_conf_count_5": high_hist["5"],
                "high_conf_count_10": high_hist["10"],
            }
            cell["label"] = label_variability(cell)
            out[f"{step}:{layer}"] = cell
    return out


def label_variability(cell: dict[str, Any]) -> str:
    """STATIC / MOSTLY_STATIC / CONTENT_SENSITIVE / HIGHLY_VARIABLE / INSUFFICIENT."""
    n = int(cell.get("n") or 0)
    if n < int(VARIABILITY_THRESHOLDS["min_n"]):
        return "INSUFFICIENT"
    frac = float(cell.get("majority_frac") or 0.0)
    if frac >= float(VARIABILITY_THRESHOLDS["static"]):
        return "STATIC"
    if frac >= float(VARIABILITY_THRESHOLDS["mostly_static"]):
        return "MOSTLY_STATIC"
    if frac >= float(VARIABILITY_THRESHOLDS["content_sensitive"]):
        return "CONTENT_SENSITIVE"
    return "HIGHLY_VARIABLE"


def is_lookup_static(label: str) -> bool:
    return label in STATIC_LABELS


def hybrid_budget(
    records: list[dict[str, Any]],
    pair_table: dict[str, dict[str, Any]] | None = None,
    *,
    fix_step1: bool = True,
) -> dict[str, Any]:
    """How many eligible decisions a hybrid policy would freeze.

    Step 1 → fixed 10% when ``fix_step1``. Later STATIC/MOSTLY_STATIC cells →
    lookup. Remainder → needs a controller. Counts are over eligible rows.
    """
    eligible = eligible_records(records)
    table = pair_table or pair_cell_stats(records)
    n_gen = len({str(r["generation_id"]) for r in eligible})
    step1 = 0
    static_later = 0
    variable = 0
    insufficient = 0
    for rec in eligible:
        step = int(rec["step"])
        layer = int(rec["layer"])
        if fix_step1 and step == 1:
            step1 += 1
            continue
        label = table.get(f"{step}:{layer}", {}).get("label") or "INSUFFICIENT"
        if is_lookup_static(label):
            static_later += 1
        elif label == "INSUFFICIENT":
            insufficient += 1
            variable += 1
        else:
            variable += 1
    total = len(eligible)
    frozen = step1 + static_later
    return {
        "n_eligible": total,
        "n_generations": n_gen,
        "decisions_per_generation": (total / n_gen) if n_gen else 0.0,
        "fix_step1": fix_step1,
        "step1_fixed": step1,
        "static_lookup_later": static_later,
        "needs_controller": variable,
        "insufficient_as_controller": insufficient,
        "frozen": frozen,
        "frac_frozen": (frozen / total) if total else 0.0,
        "frac_needs_controller": (variable / total) if total else 0.0,
        "thresholds": dict(VARIABILITY_THRESHOLDS),
    }


def summarize_labels(pair_table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[str]] = defaultdict(list)
    later: dict[str, list[str]] = defaultdict(list)
    for key, cell in pair_table.items():
        lab = cell["label"]
        by_label[lab].append(key)
        if int(cell["step"]) >= 2:
            later[lab].append(key)
    return {
        "all": {k: len(v) for k, v in sorted(by_label.items())},
        "steps_2_4": {k: len(v) for k, v in sorted(later.items())},
        "static_or_mostly_all": len(by_label["STATIC"]) + len(by_label["MOSTLY_STATIC"]),
        "static_or_mostly_later": len(later["STATIC"]) + len(later["MOSTLY_STATIC"]),
        "variable_later": len(later["CONTENT_SENSITIVE"]) + len(later["HIGHLY_VARIABLE"]),
        "examples": {
            "static_later": sorted(
                later["STATIC"],
                key=lambda k: -pair_table[k]["majority_frac"],
            )[:12],
            "highly_variable_later": sorted(
                later["HIGHLY_VARIABLE"],
                key=lambda k: pair_table[k]["majority_frac"],
            )[:12],
            "content_sensitive_later": later["CONTENT_SENSITIVE"][:12],
        },
    }


KeyFn = Callable[[dict[str, Any]], Hashable]


def key_layer_step(rec: dict[str, Any]) -> tuple[int, int]:
    return (int(rec["layer"]), int(rec["step"]))


def key_layer_step_prev(rec: dict[str, Any]) -> tuple[int, int, int | None]:
    return (int(rec["layer"]), int(rec["step"]), rec.get("previous_keep"))


def key_layer_step_prev_ranks(rec: dict[str, Any]) -> tuple:
    return (
        int(rec["layer"]),
        int(rec["step"]),
        rec.get("previous_keep"),
        rank_bin(rec.get("video_rank")),
        rank_bin(rec.get("audio_rank")),
        residual_bin(rec.get("video_residual_relative_l2")),
    )


def leave_one_gen_lookup(
    records: list[dict[str, Any]],
    key_fn: KeyFn,
    *,
    backoff: list[KeyFn] | None = None,
) -> list[int]:
    """Leave-one-generation-out majority lookup. Tie → larger keep. Unseen → 10.

    ``backoff`` is tried in order when the primary key is missing in train.
    """
    by_gen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_gen[str(rec["generation_id"])].append(rec)
    pred_map: dict[str, int] = {}
    chain = [key_fn] + list(backoff or [])
    for hold in by_gen:
        train = [r for gid, rows in by_gen.items() if gid != hold for r in rows]
        tables: list[dict[Hashable, Counter]] = []
        for fn in chain:
            table: dict[Hashable, Counter] = defaultdict(Counter)
            for r in train:
                table[fn(r)][int(r["jev_choice"])] += 1
            tables.append(table)
        for r in by_gen[hold]:
            choice = LOOKUP_DEFAULT
            for fn, table in zip(chain, tables):
                counts = table.get(fn(r))
                if counts:
                    choice = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
                    break
            pred_map[r["sample_id"]] = int(choice)
    return [pred_map[r["sample_id"]] for r in records]


def predictor_bundle(records: list[dict[str, Any]], preds: list[int]) -> dict[str, Any]:
    bundle = metrics_bundle(records, preds)
    bundle["confidence"] = stratified_confidence(records, preds)
    return bundle


def compare_predictors(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = eligible_records(records)
    a = leave_one_gen_lookup(eligible, key_layer_step)
    b = leave_one_gen_lookup(
        eligible, key_layer_step_prev, backoff=[key_layer_step]
    )
    c = leave_one_gen_lookup(
        eligible,
        key_layer_step_prev_ranks,
        backoff=[key_layer_step_prev, key_layer_step],
    )
    # Identity check vs existing LOG helper
    legacy = layer_step_lookup_logo(eligible)
    return {
        "n": len(eligible),
        "A_layer_step": predictor_bundle(eligible, a),
        "B_layer_step_prev": predictor_bundle(eligible, b),
        "C_layer_step_prev_telemetry": predictor_bundle(eligible, c),
        "A_matches_legacy_logo": a == legacy,
        "C_minus_A_exact": (
            predictor_bundle(eligible, c)["exact_agreement"]
            - predictor_bundle(eligible, a)["exact_agreement"]
        ),
    }


def step1_histogram(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in eligible_records(records) if int(r["step"]) == 1]
    counts = Counter(int(r["jev_choice"]) for r in rows)
    n = len(rows)
    return {
        "n": n,
        "histogram": {str(k): int(counts.get(k, 0)) for k in KEEP},
        "frac_10": (counts.get(10, 0) / n) if n else 0.0,
        "all_10": n > 0 and counts.get(10, 0) == n,
    }


def key_constant(_rec: dict[str, Any]) -> tuple:
    return ()


def key_previous_keep(rec: dict[str, Any]) -> tuple:
    return (rec.get("previous_keep"),)


def key_ranks(rec: dict[str, Any]) -> tuple:
    return (rank_bin(rec.get("video_rank")), rank_bin(rec.get("audio_rank")))


def key_residuals(rec: dict[str, Any]) -> tuple:
    return (
        residual_bin(rec.get("video_residual_relative_l2")),
        residual_bin(rec.get("audio_residual_relative_l2")),
    )


def key_cross_step(rec: dict[str, Any]) -> tuple:
    return (
        cross_bin(rec.get("video_cross_step_change")),
        cross_bin(rec.get("audio_cross_step_change")),
    )


def key_all_telemetry(rec: dict[str, Any]) -> tuple:
    return key_previous_keep(rec) + key_ranks(rec) + key_residuals(rec) + key_cross_step(rec)


def _hist_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(r["jev_choice"]) for r in rows)
    n = len(rows)
    maj, frac = _majority(counts)
    return {
        "n": n,
        "histogram": {str(k): int(counts.get(k, 0)) for k in KEEP},
        "majority": maj if n else None,
        "majority_frac": frac if n else 0.0,
        "entropy": shannon_entropy(counts.values()) if n else 0.0,
    }


def _mean_or_none(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _cell_rows(
    records: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in eligible_records(records):
        buckets[(int(rec["step"]), int(rec["layer"]))].append(rec)
    return buckets


def _logo_exact(rows: list[dict[str, Any]], key_fn: KeyFn, backoff: list[KeyFn] | None = None) -> float:
    if not rows:
        return 0.0
    preds = leave_one_gen_lookup(rows, key_fn, backoff=backoff)
    y = [int(r["jev_choice"]) for r in rows]
    return exact_agreement(y, preds)


def cell_predictor_lifts(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Leave-one-generation-out exact vs cell majority, on this cell's rows only."""
    maj = _logo_exact(rows, key_constant)
    prev = _logo_exact(rows, key_previous_keep, backoff=[key_constant])
    ranks = _logo_exact(rows, key_ranks, backoff=[key_constant])
    residual = _logo_exact(rows, key_residuals, backoff=[key_constant])
    cross = _logo_exact(rows, key_cross_step, backoff=[key_constant])
    telem = _logo_exact(
        rows,
        key_all_telemetry,
        backoff=[key_previous_keep, key_constant],
    )
    return {
        "majority_logo_exact": maj,
        "previous_keep_exact": prev,
        "previous_keep_lift": prev - maj,
        "rank_exact": ranks,
        "rank_lift": ranks - maj,
        "residual_exact": residual,
        "residual_lift": residual - maj,
        "cross_step_exact": cross,
        "cross_step_lift": cross - maj,
        "telem_all_exact": telem,
        "telem_all_lift": telem - maj,
    }


def classify_variable_cell(stats: dict[str, Any], pred: dict[str, float]) -> tuple[str, list[str]]:
    """REAL_DYNAMIC_SIGNAL / AMBIGUOUS / LIKELY_JEV_NOISE.

    Rules (later-step CONTENT_SENSITIVE / HIGHLY_VARIABLE cells only):

    REAL DYNAMIC SIGNAL if previous_keep or rank/residual/cross-step bins
    beat the cell majority by at least ``lift_strong`` (0.20)
    leave-one-generation-out — the choices change with telemetry even when
    Jev is unconfident — **or** the class mix **remains** at high Jev
    confidence (n≥5 at ≥0.8, majority_frac<0.80) **and** lift ≥ ``lift_real``
    (0.10). Mix remaining at ≥0.7 (n≥8) with lift ≥0.20 also counts.

    LIKELY JEV NOISE if it is not REAL SIGNAL, telemetry/previous_keep do
    not beat majority by ``lift_real``, and the mix collapses once low-
    confidence rows are dropped (majority_frac ≥ 0.80 at ≥ 0.5 or ≥ 0.7,
    or almost no rows at ≥ 0.7).

    AMBIGUOUS otherwise (high-conf mix without a telemetry split, or a
    weak 0.10–0.20 lift without a remaining high-conf mix).
    """
    t = NOISE_THRESHOLDS
    n8 = int(stats["at_ge_0.8"]["n"])
    maj8 = float(stats["at_ge_0.8"]["majority_frac"])
    n7 = int(stats["at_ge_0.7"]["n"])
    maj7 = float(stats["at_ge_0.7"]["majority_frac"])
    n5 = int(stats["at_ge_0.5"]["n"])
    maj5 = float(stats["at_ge_0.5"]["majority_frac"])
    lift_prev = float(pred.get("previous_keep_lift") or 0.0)
    lift_telem = float(pred.get("telem_all_lift") or 0.0)
    predicts = lift_prev >= t["lift_real"] or lift_telem >= t["lift_real"]
    strong = lift_prev >= t["lift_strong"] or lift_telem >= t["lift_strong"]
    mixed_high = n8 >= t["high_conf_min_n"] and maj8 < t["still_mixed_frac"]
    mixed_mid_strong = n7 >= t["mid_conf_min_n_strong"] and maj7 < t["still_mixed_frac"]
    collapsed = (
        (n5 >= t["collapsed_min_n"] and maj5 >= t["collapsed_frac"])
        or (n8 == 0 and n7 <= 2)
        or (n7 >= t["collapsed_min_n"] and maj7 >= t["collapsed_frac"])
    )
    reasons: list[str] = []
    if strong:
        reasons.append(
            "previous_keep/telemetry leave-one-gen exact beats cell majority by >=lift_strong"
        )
        return "REAL_DYNAMIC_SIGNAL", reasons
    if mixed_high and predicts:
        reasons.append(
            "class mix remains at jev_confidence>=0.8 and previous_keep/telemetry "
            "beat cell majority by >=lift_real"
        )
        return "REAL_DYNAMIC_SIGNAL", reasons
    if mixed_mid_strong and predicts:
        reasons.append(
            "class mix remains at jev_confidence>=0.7 and previous_keep/telemetry "
            "beat cell majority by >=lift_real"
        )
        return "REAL_DYNAMIC_SIGNAL", reasons
    if (not predicts) and collapsed:
        reasons.append(
            "variability concentrated below mid/high jev_confidence; "
            "telemetry does not predict the remaining choices"
        )
        return "LIKELY_JEV_NOISE", reasons
    if mixed_high and not predicts:
        reasons.append("high-confidence mix remains but telemetry/previous_keep do not split classes")
        return "AMBIGUOUS", reasons
    if predicts and not mixed_high:
        reasons.append("telemetry/previous_keep split exists but the mix is mostly low-confidence")
        return "AMBIGUOUS", reasons
    reasons.append("weak remaining mix and/or weak telemetry lift")
    return "AMBIGUOUS", reasons


def _confidence_per_class(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in KEEP:
        vals = [
            float(r["jev_confidence"])
            for r in rows
            if int(r["jev_choice"]) == k and r.get("jev_confidence") is not None
        ]
        out[str(k)] = _mean_or_none(vals)
    return out


def analyze_one_variable_cell(rows: list[dict[str, Any]], cell: dict[str, Any]) -> dict[str, Any]:
    confs = [
        float(r["jev_confidence"])
        for r in rows
        if r.get("jev_confidence") is not None
    ]
    ge = {}
    for thr in (0.5, 0.7, 0.8):
        subset = [
            r
            for r in rows
            if r.get("jev_confidence") is not None and float(r["jev_confidence"]) >= thr
        ]
        ge[f"at_ge_{thr}"] = _hist_stats(subset)
    pred = cell_predictor_lifts(rows)
    stats = {
        "step": int(cell["step"]),
        "layer": int(cell["layer"]),
        "key": f"{int(cell['step'])}:{int(cell['layer'])}",
        "variability_label": cell["label"],
        "n": len(rows),
        "histogram": {
            "1": int(cell.get("count_1") or 0),
            "3": int(cell.get("count_3") or 0),
            "5": int(cell.get("count_5") or 0),
            "10": int(cell.get("count_10") or 0),
        },
        "entropy": float(cell.get("entropy") or 0.0),
        "majority": cell.get("majority"),
        "majority_frac": float(cell.get("majority_frac") or 0.0),
        "mean_confidence": _mean_or_none(confs),
        "confidence_per_class": _confidence_per_class(rows),
        "at_ge_0.5": ge["at_ge_0.5"],
        "at_ge_0.7": ge["at_ge_0.7"],
        "at_ge_0.8": ge["at_ge_0.8"],
        "predictors": pred,
    }
    label, reasons = classify_variable_cell(stats, pred)
    stats["noise_label"] = label
    stats["noise_reasons"] = reasons
    stats["thresholds"] = dict(NOISE_THRESHOLDS)
    return stats


def later_variable_keys(pair_table: dict[str, dict[str, Any]]) -> list[str]:
    keys = []
    for key, cell in pair_table.items():
        if int(cell["step"]) >= 2 and cell.get("label") in VARIABLE_LABELS:
            keys.append(key)
    return sorted(keys, key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))


def analyze_variable_cells(
    records: list[dict[str, Any]],
    pair_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per later CONTENT_SENSITIVE / HIGHLY_VARIABLE cell: mix, confidence, telemetry, noise label."""
    table = pair_table or pair_cell_stats(records)
    buckets = _cell_rows(records)
    keys = later_variable_keys(table)
    cells: dict[str, dict[str, Any]] = {}
    for key in keys:
        cell = table[key]
        rows = buckets.get((int(cell["step"]), int(cell["layer"])), [])
        cells[key] = analyze_one_variable_cell(rows, cell)
    counts = Counter(c["noise_label"] for c in cells.values())
    collapsed_07 = sum(
        1
        for c in cells.values()
        if c["at_ge_0.7"]["n"] >= 3 and c["at_ge_0.7"]["majority_frac"] >= 0.80
    )
    collapsed_08 = sum(
        1
        for c in cells.values()
        if c["at_ge_0.8"]["n"] >= 3 and c["at_ge_0.8"]["majority_frac"] >= 0.80
    )
    still_mixed_08 = sum(
        1
        for c in cells.values()
        if c["at_ge_0.8"]["n"] >= 5 and c["at_ge_0.8"]["majority_frac"] < 0.80
    )
    mean_lift = {
        name: (
            sum(c["predictors"][name] for c in cells.values()) / len(cells) if cells else 0.0
        )
        for name in (
            "previous_keep_lift",
            "rank_lift",
            "residual_lift",
            "cross_step_lift",
            "telem_all_lift",
        )
    }
    return {
        "n_variable_later": len(keys),
        "keys": keys,
        "counts": {lab: int(counts.get(lab, 0)) for lab in NOISE_LABELS},
        "cells": cells,
        "confidence_filter": {
            "n_variable": len(keys),
            "collapsed_majority_ge_0.80_at_conf_ge_0.7": collapsed_07,
            "collapsed_majority_ge_0.80_at_conf_ge_0.8": collapsed_08,
            "still_mixed_at_conf_ge_0.8": still_mixed_08,
        },
        "mean_lifts": mean_lift,
        "thresholds": dict(NOISE_THRESHOLDS),
        "rules": (
            "REAL_DYNAMIC_SIGNAL: previous_keep or telemetry leave-one-gen exact beats the "
            "cell majority by >=0.20, or mix remains at jev_confidence>=0.8 (n>=5, "
            "majority_frac<0.80) with lift>=0.10, or mix remains at >=0.7 (n>=8) with lift>=0.20. "
            "LIKELY_JEV_NOISE: not REAL SIGNAL, lift<0.10, and the mix collapses at "
            "confidence>=0.5 or >=0.7 (majority_frac>=0.80) or almost no high-conf rows. "
            "AMBIGUOUS: everything else. Aggressive hybrid treats AMBIGUOUS as majority lookup."
        ),
    }


def aggressive_hybrid_budget(
    records: list[dict[str, Any]],
    pair_table: dict[str, dict[str, Any]] | None = None,
    noise_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 1 fixed 10%; STATIC/MOSTLY_STATIC lookup; LIKELY_JEV_NOISE + AMBIGUOUS
    majority lookup; only REAL_DYNAMIC_SIGNAL stays dynamic.
    """
    eligible = eligible_records(records)
    table = pair_table or pair_cell_stats(records)
    analysis = noise_analysis or analyze_variable_cells(records, table)
    noise_by_key = {k: v["noise_label"] for k, v in analysis["cells"].items()}
    n_gen = len({str(r["generation_id"]) for r in eligible})
    step1 = 0
    static_later = 0
    noise_majority = 0
    real_signal = 0
    insufficient = 0
    for rec in eligible:
        step = int(rec["step"])
        layer = int(rec["layer"])
        if step == 1:
            step1 += 1
            continue
        key = f"{step}:{layer}"
        label = table.get(key, {}).get("label") or "INSUFFICIENT"
        if is_lookup_static(label):
            static_later += 1
            continue
        if label == "INSUFFICIENT":
            insufficient += 1
            noise_majority += 1
            continue
        noise = noise_by_key.get(key, "LIKELY_JEV_NOISE")
        if noise == "REAL_DYNAMIC_SIGNAL":
            real_signal += 1
        else:
            noise_majority += 1
    total = len(eligible)
    frozen = step1 + static_later + noise_majority
    dynamic_keys = sorted(
        k for k, lab in noise_by_key.items() if lab == "REAL_DYNAMIC_SIGNAL"
    )
    n_dynamic_cells = len(dynamic_keys)
    return {
        "n_eligible": total,
        "n_generations": n_gen,
        "decisions_per_generation": DECISIONS_PER_GENERATION,
        "step1_fixed": step1,
        "static_lookup_later": static_later,
        "noise_majority_lookup": noise_majority,
        "insufficient_as_lookup": insufficient,
        "needs_controller": real_signal,
        "frozen": frozen,
        "frac_frozen": (frozen / total) if total else 0.0,
        "frac_needs_controller": (real_signal / total) if total else 0.0,
        "variable_later_cells": analysis["n_variable_later"],
        "noise_counts": analysis["counts"],
        "remaining_dynamic_cells": dynamic_keys,
        "remaining_dynamic_of_197": n_dynamic_cells,
        "pct_eliminated_of_197": (
            ((DECISIONS_PER_GENERATION - n_dynamic_cells) / DECISIONS_PER_GENERATION) * 100.0
        ),
        "prior_hybrid_dynamic_cells": analysis["n_variable_later"],
        "thresholds": {
            "variability": dict(VARIABILITY_THRESHOLDS),
            "noise": dict(NOISE_THRESHOLDS),
        },
    }


def fill_cell_keep(
    cell: dict[str, Any],
    *,
    step: int,
    layer: int,
    force_step1_10: bool = False,
) -> tuple[int, str]:
    """Keep percent + reason for one (step, layer). Does not use STATIC min_n=5.

    Order:
    1. later layer 0 → 5 (host constraint, not a Jev vote)
    2. step 1 + force_step1_10 → 10 (author 4-step recipe)
    3. high-conf majority if n_high ≥ 2 and frac ≥ 0.75 (n=4 families allowed)
    4. all-vote majority if n ≥ 2 and frac ≥ 0.75
    5. high-conf plurality if any high-conf vote exists (mixed / uncertain)
    6. all-vote plurality if any vote exists (mixed / uncertain — not pad 10)
    7. empty cell → 10

    Uncertain cells take the plurality (tie → denser keep via ``_majority``).
    They are not silently written as 10 just because n=4.
    """
    if layer == 0 and step > 1:
        return LATER_LAYER0_KEEP, "later_layer0"
    if step == 1 and force_step1_10:
        return 10, "force_step1_10"
    t = FILL_THRESHOLDS
    high_n = int(cell.get("high_conf_n") or 0)
    high_maj = cell.get("high_conf_majority")
    high_frac = float(cell.get("high_conf_frac") or 0.0)
    n = int(cell.get("n") or 0)
    maj = cell.get("majority")
    frac = float(cell.get("majority_frac") or 0.0)
    if (
        high_n >= int(t["min_n_high_conf"])
        and high_frac >= float(t["high_conf_frac"])
        and high_maj in KEEP_SET
    ):
        return int(high_maj), "high_conf_majority"
    if n >= int(t["min_n_all"]) and frac >= float(t["all_frac"]) and maj in KEEP_SET:
        return int(maj), "all_majority"
    if high_n >= 1 and high_maj in KEEP_SET:
        return int(high_maj), "high_conf_plurality"
    if n >= 1 and maj in KEEP_SET:
        return int(maj), "all_plurality"
    return LOOKUP_DEFAULT, "empty"


def majority_keep_table(
    pair_table: dict[str, dict[str, Any]],
    n_steps: int | None = None,
    *,
    force_step1_10: bool = True,
    conservative: bool = False,
    reasons: list[list[str]] | None = None,
) -> list[list[int]]:
    """N×50 keep percents. Later layer 0 = 5.

    ``conservative=False`` (author 4-step default): step 1 = 10 when
    ``force_step1_10``; else raw cell majority or 10.

    ``conservative=True`` (HQ tables): use ``fill_cell_keep``. High-confidence
    majority at n=4 is kept (1/3/5/10). Does **not** write 10 merely because
    n<5. Pass ``force_step1_10=False`` so step 1 follows the data.
    """
    if n_steps is None:
        steps = [int(str(k).split(":")[0]) for k in pair_table]
        n_steps = max(steps) if steps else N_STEPS
    table: list[list[int]] = []
    if reasons is not None:
        reasons.clear()
    for step in range(1, int(n_steps) + 1):
        row: list[int] = []
        reason_row: list[str] = []
        for layer in range(N_LAYERS):
            cell = pair_table.get(f"{step}:{layer}") or {}
            if conservative:
                keep, why = fill_cell_keep(
                    cell, step=step, layer=layer, force_step1_10=force_step1_10
                )
            else:
                if step == 1 and force_step1_10:
                    keep, why = 10, "force_step1_10"
                elif layer == 0 and step > 1:
                    keep, why = LATER_LAYER0_KEEP, "later_layer0"
                else:
                    maj = cell.get("majority")
                    keep = int(maj) if maj in KEEP_SET else LOOKUP_DEFAULT
                    why = "all_majority" if maj in KEEP_SET else "empty"
            row.append(int(keep))
            reason_row.append(why)
        table.append(row)
        if reasons is not None:
            reasons.append(reason_row)
    return table


def overlay_keep_table(
    table: list[list[int]],
    overrides: dict[str, int],
) -> list[list[int]]:
    """Copy ``table`` and write ``overrides`` keyed ``step:layer``. Later layer 0 stays 5."""
    out = [row[:] for row in table]
    for key, keep in overrides.items():
        step_s, layer_s = str(key).split(":")
        step = int(step_s)
        layer = int(layer_s)
        if keep not in KEEP_SET:
            raise ValueError(f"override {key} keep {keep} not in {KEEP}")
        n = len(out)
        if not (1 <= step <= n and 0 <= layer < N_LAYERS):
            raise ValueError(f"override key out of range: {key}")
        out[step - 1][layer] = int(keep)
    for si in range(1, len(out)):
        out[si][0] = LATER_LAYER0_KEEP
    return out


def clip_jev_overrides(
    records: list[dict[str, Any]],
    generation_id: str,
    dynamic_keys: Iterable[str],
) -> dict[str, int]:
    """That clip's recorded jev_raw_choice on REAL SIGNAL cells only."""
    want = set(dynamic_keys)
    out: dict[str, int] = {}
    for rec in eligible_records(records):
        if str(rec["generation_id"]) != str(generation_id):
            continue
        key = f"{int(rec['step'])}:{int(rec['layer'])}"
        if key in want:
            out[key] = int(rec["jev_choice"])
    return out


def keep_table_payload(table: list[list[int]]) -> dict[str, Any]:
    return {"keep_table": table, "n_steps": len(table), "n_layers": N_LAYERS}
