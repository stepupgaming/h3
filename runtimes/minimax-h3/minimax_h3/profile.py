"""Lightweight phase profiler for the streamed sample path.

Uses CUDA events when the active device is CUDA so transfer vs compute
breakdowns are real device time; falls back to host perf_counter otherwise.
Accumulates across steps so a multi-step sample emits one summary.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch


@dataclass
class PhaseProfiler:
    """Named-phase accumulator. Enable with ``enabled=True``; no-op otherwise."""

    enabled: bool = False
    use_cuda: bool = False
    _ms: Dict[str, float] = field(default_factory=dict)
    _counts: Dict[str, int] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)
    steps: int = 0
    wall_s: float = 0.0
    peak_vram_bytes: int = 0
    _wall0: Optional[float] = None

    def start_wall(self) -> None:
        if not self.enabled:
            return
        self._wall0 = time.perf_counter()
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_wall(self) -> None:
        if not self.enabled or self._wall0 is None:
            return
        self.wall_s = time.perf_counter() - self._wall0
        if self.use_cuda and torch.cuda.is_available():
            self.peak_vram_bytes = int(torch.cuda.max_memory_allocated())

    def mark_step(self) -> None:
        if self.enabled:
            self.steps += 1

    @contextmanager
    def phase(self, name: str, *, sync: bool = True) -> Iterator[None]:
        """Time a named region.

        ``sync=True`` (default): full device synchronize around the region so
        CUDA work is attributed correctly (attn/mlp/etc).

        ``sync=False``: host wall time only — use for H2D *enqueue* / free
        bookkeeping so the profiler does not force a device-wide stall and
        destroy transfer/compute overlap. Those numbers are CPU overhead, not
        pure DMA time.
        """
        if not self.enabled:
            yield
            return
        if name not in self._ms:
            self._order.append(name)
            self._ms[name] = 0.0
            self._counts[name] = 0
        if self.use_cuda and torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                self._ms[name] += float(start.elapsed_time(end))
                self._counts[name] += 1
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._ms[name] += (time.perf_counter() - t0) * 1e3
                self._counts[name] += 1

    def summary(self) -> dict:
        total_ms = sum(self._ms.values())
        phases = []
        for name in self._order:
            ms = self._ms[name]
            phases.append(
                {
                    "name": name,
                    "total_ms": round(ms, 3),
                    "count": self._counts[name],
                    "ms_per_call": round(ms / max(self._counts[name], 1), 3),
                    "pct": round(100.0 * ms / total_ms, 2) if total_ms > 0 else 0.0,
                }
            )
        steps = max(self.steps, 1)
        return {
            "steps": self.steps,
            "wall_s": round(self.wall_s, 3),
            "s_per_step": round(self.wall_s / steps, 3) if self.steps else None,
            "phased_total_ms": round(total_ms, 3),
            "peak_vram_mb": round(self.peak_vram_bytes / (1024 * 1024), 2),
            "phases": phases,
        }

    def print_summary(self) -> None:
        if not self.enabled:
            return
        s = self.summary()
        print("=== profile ===")
        print(
            f"steps {s['steps']}  wall {s['wall_s']:.3f}s  "
            f"s/step {s['s_per_step']}  peak_vram {s['peak_vram_mb']} MB"
        )
        print(f"{'phase':<22} {'total_ms':>10} {'ms/call':>10} {'count':>7} {'pct':>7}")
        for p in s["phases"]:
            print(
                f"{p['name']:<22} {p['total_ms']:10.2f} {p['ms_per_call']:10.2f} "
                f"{p['count']:7d} {p['pct']:6.1f}%"
            )

    def write_json(self, path: Path, extra: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        payload = self.summary()
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote profile {path}")


# Process-wide optional profiler the model forward can see without threading
# kwargs through every call site when unset.
ACTIVE: Optional[PhaseProfiler] = None


def get_profiler() -> Optional[PhaseProfiler]:
    return ACTIVE


def set_profiler(p: Optional[PhaseProfiler]) -> None:
    global ACTIVE
    ACTIVE = p
