"""Global temporal windows: simulated weekly cycles + a final test window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recops.config import TimelineConfig


@dataclass(frozen=True)
class Windows:
    """All boundaries are days; ``[a, b)`` intervals.

    Stage-1 models for ranker training use history ``< t1``; the ranker labels live in
    ``[label_start, label_end)``; the deployed stage-1 models use history ``< t2``
    (``t2 == label_end``); the evaluation holdout is ``[h_start, h_end)``.
    """

    t1: int
    label_start: int
    label_end: int
    t2: int
    h_start: int
    h_end: int


def windows_for(h_start: int, h_end: int, tl: TimelineConfig) -> Windows:
    t2 = h_start - tl.gap_days
    t1 = t2 - tl.gap_days - tl.ranker_label_days
    return Windows(t1, t1 + tl.gap_days, t2, t2, h_start, h_end)


def plan(max_day: int, tl: TimelineConfig) -> tuple[list[Windows], Windows]:
    """Return ``(cycles, test)``. Cycle holdouts end before the test window minus the gap."""
    test_end = max_day + 1
    test_start = test_end - tl.test_days
    cycles: list[Windows] = []
    for k in range(tl.n_cycles):
        h = test_start - tl.gap_days - tl.eval_days - (tl.n_cycles - 1 - k) * tl.step_days
        cycles.append(windows_for(h, h + tl.eval_days, tl))
    test = windows_for(test_start, test_end, tl)
    if cycles[0].t1 < tl.min_history_days:
        raise ValueError(
            f"Not enough history: first cycle has only {cycles[0].t1} days before t1 "
            f"(need >= {tl.min_history_days}). Reduce n_cycles/step_days or use more data."
        )
    return cycles, test
