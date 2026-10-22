"""Core steps shared by simulation, final evaluation, Airflow and serving."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from recops.bundle import Bundle
from recops.candidates.stage1 import Candidates, Stage1, rrf, top_k
from recops.config import Config
from recops.data.store import DataStore, in_users
from recops.data.timeline import Windows
from recops.evaluation.metrics import list_stats, ranking_metrics
from recops.features.builder import (
    FEATURE_COLS,
    SOURCES,
    feature_frame,
    merge_candidates,
    restrict_sources,
    to_matrix,
)
from recops.ranking.ranker import fit_ranker, make_train_set


def warm_users(pos: pl.DataFrame, min_history: int) -> np.ndarray:
    """Users with at least ``min_history`` distinct positive items."""
    g = pos.group_by("uid").agg(pl.col("item_id").n_unique().alias("n"))
    return np.sort(g.filter(pl.col("n") >= min_history)["uid"].to_numpy().astype(np.int64))


def corrupt_history(pos: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Destroy the user-item signal by shuffling item ids (used for the 'bad candidate' demo)."""
    rng = np.random.default_rng(seed)
    return pos.with_columns(pl.Series("item_id", rng.permutation(pos["item_id"].to_numpy())))


def train_bundle(
    ds: DataStore,
    win: Windows,
    cfg: Config,
    seed: int,
    corrupt: bool = False,
    variants: dict[str, tuple[str, ...]] | None = None,
) -> Bundle:
    """Train stage 1 + ranker for one cycle.

    A) stage 1 on history < t1 -> candidates as of t1 -> labels in [label_start, label_end)
       -> sample negatives from candidates -> fit the ranker (features as of t1).
    B) stage 1 re-fit on history < t2 (deployed models; includes the labels window).
    The ranker never sees labels from the evaluation holdout (>= h_start).
    """
    window = cfg.timeline.train_window_days
    hist_a = ds.pos_hist(win.t1, window)
    if corrupt:
        hist_a = corrupt_history(hist_a, seed)
    s1a = Stage1.fit(hist_a, cfg, seed)
    view_a = ds.view(win.t1)
    users_a = warm_users(ds.pos_hist(win.t1), cfg.evaluation.cold_min_history)
    cand_a = s1a.candidates(view_a, users_a, cfg)
    merged_a = merge_candidates(cand_a.frames)
    merged_a = in_users(merged_a, cand_a.warm)
    labels = ds.labels(win.label_start, win.label_end, seen_before=win.t1)

    variants = variants or {"main": SOURCES}
    rankers = {}
    for name, sources in variants.items():
        feats = feature_frame(view_a, restrict_sources(merged_a, sources))
        train = make_train_set(feats, labels, cfg, seed)
        if corrupt:  # bad candidate: destroy the ranker's supervision signal as well
            shuffled = np.random.default_rng(seed).permutation(train["label"].to_numpy())
            train = train.with_columns(pl.Series("label", shuffled))
        rankers[name] = fit_ranker(train, cfg, seed)
        print(f"[ranker:{name}] train rows={train.height} users={train['uid'].n_unique()}")

    hist_b = ds.pos_hist(win.t2, window)
    if corrupt:
        hist_b = corrupt_history(hist_b, seed + 1)
    s1b = Stage1.fit(hist_b, cfg, seed)
    meta = {"windows": win.__dict__, "seed": seed, "corrupted": corrupt}
    extras = {k: v for k, v in rankers.items() if k != "main"}
    return Bundle(s1b, rankers["main"], meta, extra_rankers=extras)


@dataclass
class Scored:
    """Everything produced when a bundle scores a set of users at ``view.as_of``."""

    cands: Candidates
    merged: pl.DataFrame
    feats: pl.DataFrame  # warm users' candidate rows with features and ``score``
    systems: dict[str, pl.DataFrame] = field(default_factory=dict)  # name -> (uid,item_id,rank)

    @property
    def recs(self) -> pl.DataFrame:
        return self.systems["ranker"]

    @property
    def scores(self) -> np.ndarray:
        return self.feats["score"].to_numpy() if "score" in self.feats.columns else np.array([])


def _rank_cols(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("uid", "item_id", "rank")


def score_users(
    bundle: Bundle, view, users: np.ndarray, cfg: Config, all_systems: bool = False
) -> Scored:
    """Candidates -> features -> ranker for ``users``; cold users get the popularity fallback.

    With ``all_systems`` also returns the baselines (popularity, ItemKNN, ALS, RRF fusion) and
    ablation rankers, for the summary table.
    """
    k = max(cfg.metrics.k_values)
    cands = bundle.stage1.candidates(view, users, cfg)
    merged = in_users(merge_candidates(cands.frames), cands.warm)
    feats = feature_frame(view, merged)
    systems: dict[str, pl.DataFrame] = {}
    if feats.height:
        x = to_matrix(feats, FEATURE_COLS)
        feats = feats.with_columns(pl.Series("score", bundle.ranker.predict(x)))
        ranked = _rank_cols(top_k(feats, "score", k))
    else:
        ranked = pl.DataFrame(schema={"uid": pl.Int64, "item_id": pl.Int64, "rank": pl.Int64})
    pop = cands.frames["pop"]
    cold_recs = _rank_cols(
        in_users(pop, cands.cold).filter(pl.col("rank") <= k) if cands.cold.size else pop.head(0)
    )
    systems["ranker"] = pl.concat([ranked, cold_recs])
    if all_systems:
        for name, src in (("popularity", "pop"), ("itemknn", "knn"), ("als", "als")):
            systems[name] = _rank_cols(cands.frames[src].filter(pl.col("rank") <= k))
        systems["rrf"] = _rank_cols(rrf({s: cands.frames[s] for s in SOURCES}, k))
        for name, model in bundle.extra_rankers.items():
            sources = ("als",) if name == "als_only" else SOURCES
            f2 = feature_frame(view, restrict_sources(merged, sources))
            if f2.height:
                f2 = f2.with_columns(pl.Series("score", model.predict(to_matrix(f2, FEATURE_COLS))))
                systems[f"ranker_{name}"] = _rank_cols(top_k(f2, "score", k))
    return Scored(cands, merged, feats, systems)


def to_lists(recs: pl.DataFrame) -> dict[int, list[int]]:
    g = recs.sort(["uid", "rank"]).group_by("uid", maintain_order=True).agg(pl.col("item_id"))
    return dict(zip(g["uid"].to_list(), g["item_id"].to_list(), strict=True))


def targets_dict(targets: pl.DataFrame, users: np.ndarray | None = None) -> dict[int, set[int]]:
    t = targets if users is None else in_users(targets, users)
    g = t.group_by("uid").agg(pl.col("item_id"))
    return {
        u: set(items) for u, items in zip(g["uid"].to_list(), g["item_id"].to_list(), strict=True)
    }


def pick_eval_users(targets: pl.DataFrame, warm: np.ndarray, cfg: Config, seed: int) -> np.ndarray:
    """Warm users that have at least one target, capped at ``evaluation.max_users``."""
    users = np.intersect1d(np.sort(targets["uid"].unique().to_numpy()), warm)
    cap = cfg.evaluation.max_users
    if len(users) > cap:
        users = np.sort(np.random.default_rng(seed).choice(users, size=cap, replace=False))
    return users.astype(np.int64)


def evaluate_scored(
    sc: Scored, targets: pl.DataFrame, users: np.ndarray, view, cfg: Config
) -> dict[str, dict[str, float]]:
    """Metrics for every system in ``sc`` on ``users`` (Recall/NDCG/MAP/coverage/diversity/head)."""
    ks = cfg.metrics.k_values
    tg = targets_dict(targets, users)
    catalog_size = len(view.item_feats)
    n_head = max(1, int(cfg.evaluation.pop_head_frac * catalog_size))
    head = set(view.item_feats.sort("i_n_pos", descending=True).head(n_head)["item_id"].to_list())
    out: dict[str, dict[str, float]] = {}
    for name, recs in sc.systems.items():
        lists = to_lists(in_users(recs, users))
        m = ranking_metrics(lists, tg, ks)
        for k in ks:
            m.update(list_stats(lists, list(tg), k, catalog_size, head))
        out[name] = m
    return out


def candidate_recall(frames: dict[str, pl.DataFrame], targets: pl.DataFrame, users: np.ndarray):
    """Recall of each source and of every union of sources over candidate lists (not top-K).

    Also returns the unique-hit share of each source: hits found by this source and no other.
    """
    tg = in_users(targets, users).with_columns(pl.lit(1).alias("_t"))
    n_t = tg.height
    hits = {}
    for s, f in frames.items():
        h = f.join(tg, on=["uid", "item_id"], how="inner").select("uid", "item_id")
        hits[s] = set(zip(h["uid"].to_list(), h["item_id"].to_list(), strict=True))
    res: dict[str, float] = {}
    names = list(frames)
    import itertools

    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            u = set().union(*(hits[s] for s in combo))
            res["+".join(combo)] = len(u) / n_t if n_t else 0.0
    union = set().union(*hits.values()) if hits else set()
    unique = {
        s: (len(hits[s] - set().union(*(hits[o] for o in names if o != s))) / len(union))
        if union
        else 0.0
        for s in names
    }
    return {"recall": res, "unique_hit_share": unique, "n_targets": n_t}
