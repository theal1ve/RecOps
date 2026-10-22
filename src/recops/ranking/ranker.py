"""Stage 2: CatBoost ranker trained with a ranking loss on candidate rows."""

from __future__ import annotations

import numpy as np
import polars as pl

from recops.config import Config
from recops.data.store import in_users
from recops.features.builder import FEATURE_COLS, to_matrix


def make_train_set(
    feats: pl.DataFrame, labels: pl.DataFrame, cfg: Config, seed: int
) -> pl.DataFrame:
    """Label candidate rows and sample negatives *from the candidates* of the same user.

    Why: at inference the ranker only sees candidates, so negatives must come from the same
    distribution (hard, plausible negatives), not random catalog items. For every user we keep
    all positives and up to ``max(min_negatives, negatives_per_positive * n_pos)`` random
    non-positive candidates. Users without any positive among candidates are dropped.
    """
    lab = labels.select("uid", "item_id").unique().with_columns(pl.lit(1).alias("label"))
    df = feats.join(lab, on=["uid", "item_id"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )
    rng = np.random.default_rng(seed)
    df = df.with_columns(pl.Series("_r", rng.random(df.height)))
    df = df.with_columns(
        pl.col("label").sum().over("uid").alias("_npos"),
        pl.col("_r").rank("ordinal").over(["uid", "label"]).alias("_nr"),
    )
    limit = pl.max_horizontal(
        pl.lit(cfg.ranker.min_negatives), pl.col("_npos") * cfg.ranker.negatives_per_positive
    )
    df = df.filter((pl.col("_npos") > 0) & ((pl.col("label") == 1) | (pl.col("_nr") <= limit)))
    return df.drop("_r", "_npos", "_nr").sort("uid")


def fit_ranker(train: pl.DataFrame, cfg: Config, seed: int):
    """Fit CatBoostRanker (YetiRank by default) with a user-level validation split."""
    from catboost import CatBoostRanker, Pool

    if train.height == 0:
        raise ValueError("empty ranker training set (no positives among candidates)")
    r = cfg.ranker
    users = train["uid"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    n_valid = int(len(users) * r.valid_user_frac)
    valid_users = rng.choice(users, size=n_valid, replace=False) if n_valid >= 1 else users[:0]
    va = in_users(train, valid_users).sort("uid")
    tr = train.join(va.select("uid").unique(), on="uid", how="anti").sort("uid")

    def pool(df: pl.DataFrame) -> Pool:
        return Pool(
            to_matrix(df, FEATURE_COLS),
            label=df["label"].to_numpy(),
            group_id=df["uid"].to_numpy(),
            feature_names=list(FEATURE_COLS),
        )

    model = CatBoostRanker(
        loss_function=r.loss,
        iterations=r.iterations,
        depth=r.depth,
        learning_rate=r.learning_rate,
        random_seed=seed,
        thread_count=r.thread_count,
        eval_metric="NDCG:top=20",
        verbose=100,
    )
    if va.height:
        model.set_params(early_stopping_rounds=r.early_stopping_rounds)
        model.fit(pool(tr), eval_set=pool(va), use_best_model=True)
    else:
        model.fit(pool(tr))
    return model
