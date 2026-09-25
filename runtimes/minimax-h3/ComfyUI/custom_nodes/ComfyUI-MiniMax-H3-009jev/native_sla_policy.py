# SPDX-License-Identifier: GPL-3.0-only
"""Keep-policy helpers for 009jev native SLA. Stdlib only (no torch / Comfy)."""

from __future__ import annotations

from typing import Any

N_LAYERS = 50
N_STEPS = 4  # author 4-step res_multistep recipe; HQ tables may be 15/20/32
VALID_KEEP = (1, 3, 5, 10)
MIN_TABLE_STEPS = 1
MAX_TABLE_STEPS = 100
LATER_LAYER0_KEEP = 5.0
CONFIDENCE_FLOOR = 0.3
CONST_POLICIES = {
    "const1": 1.0,
    "const3": 3.0,
    "const5": 5.0,
    "const10": 10.0,
}
TABLE_POLICY = "table"
JEV_WORKER_POLICIES = ("jev_first", "fixed5", "fixed10")
VALID_POLICIES = tuple(CONST_POLICIES) + JEV_WORKER_POLICIES + (TABLE_POLICY,)


def uses_jev_worker(policy: str) -> bool:
    """Author fixed5/fixed10 still call Jev after step 1. constN and table never do."""
    return policy in JEV_WORKER_POLICIES


def initial_keeps(policy: str) -> list[float]:
    if policy in CONST_POLICIES:
        return [CONST_POLICIES[policy]] * N_LAYERS
    if policy == "fixed10":
        return [10.0] * N_LAYERS
    if policy == TABLE_POLICY:
        raise ValueError("table policy needs keep_table; use step_keeps(normalize_keep_table(...), 1)")
    return [5.0] * N_LAYERS


def normalize_keep_table(raw: Any, n_steps: int | None = None) -> list[list[float]]:
    """Validate an N×50 keep table. Later-step layer 0 is forced to 5.

    ``n_steps`` None infers N from the table (4, 15, 20, 32, or any 1..=100).
    """
    if isinstance(raw, dict) and "keep_table" in raw:
        raw = raw["keep_table"]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"keep_table must be N rows of 50 keeps (got {type(raw).__name__})")
    got = len(raw)
    if got < MIN_TABLE_STEPS or got > MAX_TABLE_STEPS:
        raise ValueError(f"keep_table must have {MIN_TABLE_STEPS}..={MAX_TABLE_STEPS} steps (got {got})")
    if n_steps is not None and got != int(n_steps):
        raise ValueError(f"keep_table has {got} steps but sampler has {n_steps}")
    table: list[list[float]] = []
    for si, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != N_LAYERS:
            raise ValueError(f"keep_table step {si + 1} must have {N_LAYERS} layers")
        out_row: list[float] = []
        for li, val in enumerate(row):
            keep = float(int(val))
            if int(keep) not in VALID_KEEP:
                raise ValueError(f"keep_table[{si}][{li}]={val} not in {VALID_KEEP}")
            if si >= 1 and li == 0:
                keep = LATER_LAYER0_KEEP
            out_row.append(keep)
        table.append(out_row)
    return table


def step_keeps(table: list[list[float]], step_1based: int) -> list[float]:
    n = len(table)
    if not (1 <= step_1based <= n):
        raise ValueError(f"step must be 1..{n} (got {step_1based})")
    return table[step_1based - 1][:]


def parse_keep_table_from_context(context: Any) -> list[list[float]] | None:
    if not isinstance(context, dict):
        return None
    if "keep_table" not in context:
        return None
    return normalize_keep_table(context["keep_table"])


def simulate_keep_schedule(
    policy: str,
    keep_table: Any = None,
    n_steps: int | None = None,
) -> list[dict[str, Any]]:
    """CPU stand-in for Controller initialize + N step callbacks. Never calls Jev.

    Logged ``applied_layer_keep_percent`` is the table (or const), not a single
    keep unless the table is actually const. ``--sla-fixed 10`` is const10 and
    must not be used as a stand-in for a mixed table. Default N is 4 (author
    recipe); pass a longer table or ``n_steps`` for HQ euler.
    """
    if uses_jev_worker(policy):
        raise ValueError("simulate_keep_schedule does not run the Jev worker")
    table = normalize_keep_table(keep_table, n_steps=n_steps) if policy == TABLE_POLICY else None
    if policy == TABLE_POLICY:
        n = len(table)
        keeps = step_keeps(table, 1)
        begin_reason = "table"
    else:
        n = int(n_steps) if n_steps is not None else N_STEPS
        if n < MIN_TABLE_STEPS or n > MAX_TABLE_STEPS:
            raise ValueError(f"n_steps must be {MIN_TABLE_STEPS}..={MAX_TABLE_STEPS} (got {n})")
        keeps = initial_keeps(policy)
        begin_reason = "const"
    last0 = n - 1
    events: list[dict[str, Any]] = [
        {
            "event": "initial_decision",
            "reason": begin_reason,
            "initial_policy": policy,
            "applied_layer_keep_percent": keeps[:],
            "n_steps": n,
        }
    ]
    for step0 in range(n):
        ev: dict[str, Any] = {
            "event": "step",
            "step": step0 + 1,
            "reason": "table" if policy == TABLE_POLICY else ("last_step" if step0 == last0 else "const"),
            "applied_layer_keep_percent": keeps[:],
            "n_steps": n,
        }
        if table is not None and step0 < last0:
            keeps = step_keeps(table, step0 + 2)
            ev["next_layer_keep_percent"] = keeps[:]
        events.append(ev)
    return events


def apply_confidence_floor(
    choice: float,
    confidence: float,
    *,
    first_step: bool,
) -> tuple[float, bool, str | None]:
    if confidence < CONFIDENCE_FLOOR:
        applied = 10.0 if first_step else 5.0
        return applied, True, "confidence_floor"
    return float(choice), False, None
