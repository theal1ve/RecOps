"""Final evaluation on the TEST window (run once), ablations, cold-start and error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from recops.config import Config
from recops.data.store import DataStore, in_users
from recops.data.timeline import plan
from recops.evaluation.metrics import ranking_metrics
from recops.features.builder import SOURCES, user_features
from recops.pipelines.core import (
    candidate_recall,
    evaluate_scored,
    pick_eval_users,
    score_users,
    targets_dict,
    to_lists,
    train_bundle,
    warm_users,
)
from recops.view import HistoryView


def cold_study(bundle, test, cfg: Config, targets: pl.DataFrame, users: np.ndarray, view) -> dict:
    """Simulated cold users: truncate a sample of users' history to the first N positives.

    Real Yambda test users are never cold (dataset card), so cold start is simulated.
    Compares the popularity fallback (what the system does) with ALS on the same tiny history.
    """
    rng = np.random.default_rng(cfg.seed)
    n = max(1, int(len(users) * cfg.evaluation.cold_sim_frac))
    sample = np.sort(rng.choice(users, size=n, replace=False))
    sdf = pl.DataFrame({"uid": sample})
    first = (
        view.ev.filter(pl.col("pos"))
        .join(sdf, on="uid", how="semi")
        .sort(["uid", "ts"])
        .group_by("uid", maintain_order=True)
        .head(cfg.evaluation.cold_sim_events)
    )
    rest = view.ev.join(sdf, on="uid", how="anti")
    ev_cold = pl.concat([rest, first.select(rest.columns)])
    vc = HistoryView(view.as_of, ev_cold, view.lk, view.item_feats, view.pop_items, view.pop_scores)
    k = max(cfg.metrics.k_values)
    tg = targets_dict(targets, sample)
    c_fb = bundle.stage1.candidates(vc, sample, cfg)  # cold -> popularity only
    c_als = bundle.stage1.candidates(vc, sample, cfg, warm_min=1)
    res = {
        "n_users": int(len(sample)),
        "history_positives": cfg.evaluation.cold_sim_events,
        "n_cold_by_rule": int(len(c_fb.cold)),
        "popularity_fallback": ranking_metrics(
            to_lists(c_fb.frames["pop"].filter(pl.col("rank") <= k)), tg, cfg.metrics.k_values
        ),
        "als_on_tiny_history": ranking_metrics(
            to_lists(c_als.frames["als"].filter(pl.col("rank") <= k)), tg, cfg.metrics.k_values
        ),
    }
    return res


def error_breakdown(
    recs: pl.DataFrame, targets: pl.DataFrame, users: np.ndarray, view, cfg: Config
) -> dict:
    """Recall@20 by target-item popularity bucket and by user-activity tercile."""
    top = (
        recs.filter(pl.col("rank") <= 20)
        .select("uid", "item_id")
        .with_columns(pl.lit(1).alias("hit"))
    )
    t = in_users(targets, users).join(top, on=["uid", "item_id"], how="left")
    t = t.with_columns(pl.col("hit").fill_null(0))
    ranks = view.item_feats.select("item_id", "i_n_pos").with_columns(
        (pl.col("i_n_pos").rank("ordinal", descending=True) / pl.len()).alias("pct")
    )
    head = cfg.evaluation.pop_head_frac
    t = t.join(ranks.select("item_id", "pct"), on="item_id", how="left").with_columns(
        pl.when(pl.col("pct").is_null())
        .then(pl.lit("unseen_in_history"))
        .when(pl.col("pct") <= head)
        .then(pl.lit("head"))
        .when(pl.col("pct") <= 0.2)
        .then(pl.lit("torso"))
        .otherwise(pl.lit("tail"))
        .alias("bucket")
    )
    by_bucket = t.group_by("bucket").agg(
        pl.len().alias("n_targets"), pl.col("hit").mean().alias("recall@20")
    )
    uf = user_features(in_users(view.ev, users), in_users(view.lk, users), view.as_of)
    per_user = t.group_by("uid").agg(pl.col("hit").mean().alias("recall@20")).join(uf, on="uid")
    q1, q2 = np.quantile(per_user["u_n_pos"].to_numpy(), [1 / 3, 2 / 3])
    per_user = per_user.with_columns(
        pl.when(pl.col("u_n_pos") <= q1)
        .then(pl.lit("low_activity"))
        .when(pl.col("u_n_pos") <= q2)
        .then(pl.lit("mid_activity"))
        .otherwise(pl.lit("high_activity"))
        .alias("tercile")
    )
    by_act = per_user.group_by("tercile").agg(
        pl.len().alias("n_users"), pl.col("recall@20").mean().alias("recall@20")
    )
    return {
        "by_target_popularity": by_bucket.to_dicts(),
        "by_user_activity": by_act.to_dicts(),
    }


def final_eval(cfg: Config, force: bool = False) -> dict:
    """Train on history < test.t2 and evaluate every system once on the test window."""
    art = Path(cfg.paths.artifacts) / "final"
    art.mkdir(parents=True, exist_ok=True)
    done = art / "test_done.json"
    if done.exists() and not force:
        raise RuntimeError(
            "TEST was already evaluated. The test window is touched once; pass --force only "
            "if you really mean to re-run (and do not tune anything based on it)."
        )
    ds = DataStore.load(cfg)
    _, test = plan(ds.max_day, cfg.timeline)
    print("test windows:", test)
    bundle = train_bundle(ds, test, cfg, cfg.seed, variants={"main": SOURCES, "als_only": ("als",)})
    view = ds.view(test.t2)
    targets = ds.labels(test.h_start, test.h_end, seen_before=test.t2)
    warm = warm_users(ds.pos_hist(test.t2), cfg.evaluation.cold_min_history)
    users = pick_eval_users(targets, warm, cfg, cfg.seed)
    sc = score_users(bundle, view, users, cfg, all_systems=True)
    metrics = evaluate_scored(sc, targets, users, view, cfg)
    catalog = bundle.stage1.catalog
    tg_items = in_users(targets, users)["item_id"].to_numpy()
    results = {
        "windows": test.__dict__,
        "n_eval_users": int(len(users)),
        "n_targets": int(tg_items.size),
        "catalog_size": int(len(catalog)),
        "cold_item_target_share": float((catalog.lookup(tg_items) < 0).mean())
        if tg_items.size
        else 0.0,
        "metrics": metrics,
        "candidate_recall": candidate_recall(sc.cands.frames, targets, users),
        "cold_start_simulated": cold_study(bundle, test, cfg, targets, users, view),
        "error_analysis": error_breakdown(sc.systems["ranker"], targets, users, view, cfg),
        "feature_importance": dict(
            zip(
                bundle.ranker.feature_names_,
                [
                    float(v)
                    for v in bundle.ranker.get_feature_importance(type="PredictionValuesChange")
                ],
                strict=True,
            )
        ),
        "primary_metric": cfg.metrics.primary,
    }
    (art / "results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    bundle.meta["holdout_primary"] = metrics["ranker"][cfg.metrics.primary]
    bundle.save(art / "bundle")
    done.write_text(json.dumps({"done": True}), encoding="utf-8")
    return results
