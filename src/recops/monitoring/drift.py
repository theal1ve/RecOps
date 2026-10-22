"""Drift and quality monitoring for a production model against its reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from recops.monitoring.psi import level, psi

if TYPE_CHECKING:
    import polars as pl

    from recops.config import Config


def drift_report(
    ref_feats: pl.DataFrame | None,
    cur_feats: pl.DataFrame,
    ref_scores: np.ndarray | None,
    cur_scores: np.ndarray,
    cfg: Config,
) -> dict:
    """PSI per monitored feature and for the score distribution; alerts per config thresholds."""
    m = cfg.monitoring
    rep: dict = {"psi": {}, "levels": {}, "alerts": []}
    if ref_feats is None:
        return rep
    for col in m.psi_features:
        if col in ref_feats.columns and col in cur_feats.columns:
            val = psi(ref_feats[col].to_numpy(), cur_feats[col].to_numpy(), m.psi_bins)
            rep["psi"][col] = val
            rep["levels"][col] = level(val, m.psi_warn, m.psi_alert)
    if ref_scores is not None and len(ref_scores) and len(cur_scores):
        val = psi(ref_scores, cur_scores, m.psi_bins)
        rep["psi"]["score"] = val
        rep["levels"]["score"] = level(val, m.psi_warn, m.psi_alert)
    rep["alerts"] = [
        f"PSI alert: {k}={rep['psi'][k]:.3f}" for k, v in rep["levels"].items() if v == "alert"
    ]
    return rep


def quality_alert(current: float, reference: float, drop_threshold: float) -> tuple[bool, float]:
    """Relative drop of the primary metric vs the production reference; True if over threshold."""
    if reference <= 0:
        return False, 0.0
    drop = 1.0 - current / reference
    return drop >= drop_threshold, drop
