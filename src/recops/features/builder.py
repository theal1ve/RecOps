"""Point-in-time correct features.

Every builder first filters its input to ``day < as_of``, so no row from the future can
influence a feature even if the caller passes the full table. The same functions are used
for training, offline evaluation and online serving (no training/serving skew).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from recops.data.store import in_users

SOURCES = ("pop", "knn", "als")
USER_COLS = [
    "u_n_listens", "u_n_pos", "u_n_pos_7d", "u_n_items", "u_org_share",
    "u_skip_rate", "u_days_since_last", "u_n_likes",
]  # fmt: skip
ITEM_COLS = [
    "i_n_listens", "i_n_pos", "i_pos_7d", "i_pos_28d", "i_n_users", "i_org_share",
    "i_pos_rate", "i_trend", "i_age_days", "i_n_likes",
]  # fmt: skip
UI_COLS = ["ui_n_pos", "ui_n_skip", "ui_days_since"]
CAND_COLS = [f"{s}_{k}" for s in SOURCES for k in ("rank", "score")] + ["n_sources"]
FEATURE_COLS = USER_COLS + ITEM_COLS + UI_COLS + CAND_COLS


def user_features(ev: pl.DataFrame, lk: pl.DataFrame, as_of: int) -> pl.DataFrame:
    """Per-user aggregates over listens/likes strictly before ``as_of``."""
    ev = ev.filter(pl.col("day") < as_of)
    pos = pl.col("pos").cast(pl.Int64)
    g = (
        ev.group_by("uid")
        .agg(
            pl.len().alias("u_n_listens"),
            pos.sum().alias("u_n_pos"),
            pos.filter(pl.col("day") >= as_of - 7).sum().alias("u_n_pos_7d"),
            pl.col("item_id").filter(pl.col("pos")).n_unique().alias("u_n_items"),
            pl.col("is_organic").filter(pl.col("pos")).mean().alias("u_org_share"),
            pl.col("day").max().alias("u_last_day"),
        )
        .with_columns(
            (1.0 - pl.col("u_n_pos") / pl.col("u_n_listens")).alias("u_skip_rate"),
            (as_of - pl.col("u_last_day")).alias("u_days_since_last"),
        )
        .drop("u_last_day")
    )
    likes = lk.filter(pl.col("day") < as_of).group_by("uid").agg(pl.len().alias("u_n_likes"))
    return g.join(likes, on="uid", how="left").with_columns(pl.col("u_n_likes").fill_null(0))


def item_features(ev: pl.DataFrame, lk: pl.DataFrame, as_of: int) -> pl.DataFrame:
    """Per-item aggregates over listens/likes strictly before ``as_of``."""
    ev = ev.filter(pl.col("day") < as_of)
    pos = pl.col("pos").cast(pl.Int64)
    g = (
        ev.group_by("item_id")
        .agg(
            pl.len().alias("i_n_listens"),
            pos.sum().alias("i_n_pos"),
            pos.filter(pl.col("day") >= as_of - 7).sum().alias("i_pos_7d"),
            pos.filter(pl.col("day") >= as_of - 28).sum().alias("i_pos_28d"),
            pl.col("uid").filter(pl.col("pos")).n_unique().alias("i_n_users"),
            pl.col("is_organic").filter(pl.col("pos")).mean().alias("i_org_share"),
            pl.col("day").min().alias("i_first_day"),
        )
        .with_columns(
            (pl.col("i_n_pos") / pl.col("i_n_listens")).alias("i_pos_rate"),
            ((pl.col("i_pos_7d") / 7.0) / (pl.col("i_pos_28d") / 28.0 + 1e-3)).alias("i_trend"),
            (as_of - pl.col("i_first_day")).alias("i_age_days"),
        )
        .drop("i_first_day")
    )
    likes = lk.filter(pl.col("day") < as_of).group_by("item_id").agg(pl.len().alias("i_n_likes"))
    return g.join(likes, on="item_id", how="left").with_columns(pl.col("i_n_likes").fill_null(0))


def ui_features(ev: pl.DataFrame, pairs: pl.DataFrame, as_of: int) -> pl.DataFrame:
    """User x item history for the given (uid, item_id) pairs, strictly before ``as_of``."""
    ev = ev.filter(pl.col("day") < as_of)
    sub = ev.join(pairs.select("uid", "item_id").unique(), on=["uid", "item_id"], how="semi")
    pos = pl.col("pos").cast(pl.Int64)
    return sub.group_by(["uid", "item_id"]).agg(
        pos.sum().alias("ui_n_pos"),
        (pl.len() - pos.sum()).alias("ui_n_skip"),
        (as_of - pl.col("day").max()).alias("ui_days_since"),
    )


def merge_candidates(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Outer-merge per-source candidate frames into one row per (uid, item_id)."""
    parts = [f.select("uid", "item_id") for f in frames.values() if f.height]
    if not parts:
        return pl.DataFrame(schema={"uid": pl.Int64, "item_id": pl.Int64})
    out = pl.concat(parts).unique()
    for s in SOURCES:
        f = frames.get(s)
        if f is not None and f.height:
            out = out.join(
                f.select(
                    "uid", "item_id",
                    pl.col("rank").cast(pl.Float64).alias(f"{s}_rank"),
                    pl.col("score").alias(f"{s}_score"),
                ),
                on=["uid", "item_id"],
                how="left",
            )  # fmt: skip
        else:
            out = out.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(f"{s}_rank"),
                pl.lit(None, dtype=pl.Float64).alias(f"{s}_score"),
            )
    rank_cols = [f"{s}_rank" for s in SOURCES]
    return out.with_columns(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in rank_cols]).alias(
            "n_sources"
        )
    )


def restrict_sources(cands: pl.DataFrame, keep: tuple[str, ...]) -> pl.DataFrame:
    """Keep only rows found by ``keep`` sources and blank out the other sources' features."""
    if set(keep) == set(SOURCES):
        return cands
    cond = pl.any_horizontal([pl.col(f"{s}_rank").is_not_null() for s in keep])
    out = cands.filter(cond)
    drop = [s for s in SOURCES if s not in keep]
    out = out.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(f"{s}_{k}") for s in drop for k in ("rank", "score")]
    )
    return out.with_columns(
        pl.sum_horizontal([pl.col(f"{s}_rank").is_not_null().cast(pl.Int64) for s in keep]).alias(
            "n_sources"
        )
    )


def feature_frame(view, cands: pl.DataFrame) -> pl.DataFrame:
    """Join user, item and user x item features onto candidate rows (as of ``view.as_of``)."""
    if cands.height == 0:
        return cands
    users = cands["uid"].unique().to_numpy()
    ev_u = in_users(view.ev, users)
    lk_u = in_users(view.lk, users)
    uf = user_features(ev_u, lk_u, view.as_of)
    uif = ui_features(ev_u, cands, view.as_of)
    out = (
        cands.join(uf, on="uid", how="left")
        .join(view.item_feats, on="item_id", how="left")
        .join(uif, on=["uid", "item_id"], how="left")
        .with_columns(pl.col("ui_n_pos").fill_null(0), pl.col("ui_n_skip").fill_null(0))
    )
    return out


def to_matrix(df: pl.DataFrame, cols: list[str] = FEATURE_COLS) -> np.ndarray:
    """Feature columns as a float64 matrix (nulls become NaN, which CatBoost handles)."""
    return df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
