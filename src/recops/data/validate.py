"""Lightweight data validation for the processed event tables (plain polars checks)."""

from __future__ import annotations

import polars as pl

REQUIRED = ["uid", "item_id", "ts", "day", "is_organic", "played_pct"]


class DataValidationError(ValueError):
    """Raised when the data violates a hard expectation."""


def validate_events(ev: pl.DataFrame) -> dict:
    """Check schema, nulls and value ranges. Returns a stats dict, raises on hard failures."""
    errors: list[str] = []
    missing = [c for c in REQUIRED if c not in ev.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if ev.height == 0:
        raise DataValidationError("empty events table")
    for c in REQUIRED:
        if ev[c].null_count() > 0:
            errors.append(f"nulls in {c}: {ev[c].null_count()}")
        if not ev[c].dtype.is_integer():
            errors.append(f"{c} is not integer: {ev[c].dtype}")
    if ev["day"].min() < 0:
        errors.append("negative day")
    if ev["played_pct"].min() < 0:
        errors.append("negative played_pct")
    if set(ev["is_organic"].unique().to_list()) - {0, 1}:
        errors.append("is_organic outside {0,1}")
    if errors:
        raise DataValidationError("; ".join(errors))
    pos_rate = float((ev["played_pct"] >= 50).mean())
    stats = {
        "n_rows": ev.height,
        "n_users": int(ev["uid"].n_unique()),
        "n_items": int(ev["item_id"].n_unique()),
        "max_day": int(ev["day"].max()),
        "played_pct_max": int(ev["played_pct"].max()),
        "listen_plus_rate": pos_rate,
        "warnings": [],
    }
    if not 0.02 <= pos_rate <= 0.98:
        stats["warnings"].append(f"suspicious Listen+ rate {pos_rate:.3f}")
    return stats
