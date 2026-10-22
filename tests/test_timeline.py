from types import SimpleNamespace

import pytest

from recops.data.timeline import plan

TL = SimpleNamespace(
    eval_days=7, ranker_label_days=7, gap_days=1, step_days=7, n_cycles=5,
    test_days=7, train_window_days=None, min_history_days=14,
)  # fmt: skip


def test_windows_are_ordered_and_disjoint() -> None:
    cycles, test = plan(299, TL)
    assert len(cycles) == 5
    for w in cycles + [test]:
        assert w.t1 < w.label_start < w.label_end == w.t2 < w.h_start < w.h_end
        assert w.label_start - w.t1 == TL.gap_days  # gap between history and labels
        assert w.h_start - w.t2 == TL.gap_days  # gap between training data and holdout


def test_cycles_never_touch_test() -> None:
    cycles, test = plan(299, TL)
    assert test.h_end == 300 and test.h_start == 293
    assert all(c.h_end <= test.h_start - TL.gap_days for c in cycles)
    assert all(c.h_start < n.h_start for c, n in zip(cycles, cycles[1:], strict=False))


def test_not_enough_history_raises() -> None:
    with pytest.raises(ValueError):
        plan(40, TL)
