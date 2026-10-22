"""Population Stability Index and a small drift report."""

from __future__ import annotations

import numpy as np


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI between a reference and a current sample (NaN is its own bin).

    Bin edges are reference quantiles, so a constant reference collapses to one or two bins.
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    r, c = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    inner = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)[1:-1])) if len(r) else np.array([])

    def shares(x: np.ndarray, n_total: int) -> np.ndarray:
        idx = np.searchsorted(inner, x, side="right")
        counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
        counts = np.append(counts, n_total - len(x))  # NaN bin
        return np.maximum(counts / n_total, eps)

    p_ref = shares(r, len(ref))
    p_cur = shares(c, len(cur))
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def level(value: float, warn: float, alert: float) -> str:
    """Classify a PSI value."""
    if value >= alert:
        return "alert"
    return "warn" if value >= warn else "ok"
