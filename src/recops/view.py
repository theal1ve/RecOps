"""History view: everything known at time ``as_of`` (used offline and online)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from recops.features.builder import item_features


@dataclass
class HistoryView:
    """Listens/likes strictly before ``as_of`` plus precomputed global item features and
    the "popular in the last N days" list. Online, ``ev``/``lk`` hold only one user's rows
    while ``item_feats`` and the popularity list are the shared snapshot."""

    as_of: int
    ev: pl.DataFrame
    lk: pl.DataFrame
    item_feats: pl.DataFrame
    pop_items: np.ndarray
    pop_scores: np.ndarray


def build_view(ds, as_of: int) -> HistoryView:
    cfg = ds.cfg
    ev = ds.events.filter(pl.col("day") < as_of)
    lk = ds.likes.filter(pl.col("day") < as_of)
    w = cfg.candidates.popularity.window_days
    pop = (
        ev.filter(pl.col("pos") & (pl.col("day") >= as_of - w))
        .group_by("item_id")
        .agg(pl.len().alias("c"))
        .sort(["c", "item_id"], descending=[True, False])
        .head(cfg.candidates.pop_list_size)
    )
    return HistoryView(
        as_of=as_of,
        ev=ev,
        lk=lk,
        item_feats=item_features(ev, lk, as_of),
        pop_items=pop["item_id"].to_numpy().astype(np.int64),
        pop_scores=pop["c"].to_numpy().astype(np.float64),
    )
