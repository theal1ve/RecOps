"""Synthetic event log with the same schema as processed Yambda (for offline checks/tests)."""

from __future__ import annotations

import numpy as np
import polars as pl


def make_synthetic(
    n_users: int = 300,
    n_items: int = 2000,
    n_days: int = 60,
    events_per_user_day: int = 8,
    n_clusters: int = 10,
    units_per_day: int = 17280,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Users prefer one item cluster (70% of events); item popularity is long-tailed.

    Returns ``(events, likes)`` with columns of the processed schema.
    """
    rng = np.random.default_rng(seed)
    item_cluster = rng.integers(0, n_clusters, n_items)
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.8
    pop = pop[rng.permutation(n_items)]
    user_cluster = rng.integers(0, n_clusters, n_users)

    n_ev = n_users * n_days * events_per_user_day
    uid = np.repeat(np.arange(n_users), n_days * events_per_user_day)
    day = rng.integers(0, n_days, n_ev)
    ts = day * units_per_day + rng.integers(0, units_per_day, n_ev)
    in_cluster = rng.random(n_ev) < 0.7

    item = np.empty(n_ev, dtype=np.int64)
    glob_p = pop / pop.sum()
    item[~in_cluster] = rng.choice(n_items, size=int((~in_cluster).sum()), p=glob_p)
    for c in range(n_clusters):
        idx = np.flatnonzero(in_cluster & (user_cluster[uid] == c))
        members = np.flatnonzero(item_cluster == c)
        p = pop[members] / pop[members].sum()
        item[idx] = rng.choice(members, size=len(idx), p=p)

    affinity = item_cluster[item] == user_cluster[uid]
    is_pos = rng.random(n_ev) < np.where(affinity, 0.75, 0.3)
    played = np.where(is_pos, rng.integers(50, 101, n_ev), rng.integers(1, 50, n_ev))

    events = pl.DataFrame(
        {
            "uid": uid.astype(np.int64) + 1000,
            "item_id": item.astype(np.int64) + 5000,
            "ts": ts.astype(np.int64),
            "day": day.astype(np.int64),
            "is_organic": rng.integers(0, 2, n_ev).astype(np.int64),
            "played_pct": played.astype(np.int64),
        }
    ).sort(["uid", "ts"])
    like_mask = is_pos & (rng.random(n_ev) < 0.03)
    likes = pl.DataFrame(
        {
            "uid": uid[like_mask].astype(np.int64) + 1000,
            "item_id": item[like_mask].astype(np.int64) + 5000,
            "ts": ts[like_mask].astype(np.int64),
            "day": day[like_mask].astype(np.int64),
        }
    )
    return events, likes
