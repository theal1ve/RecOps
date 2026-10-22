"""Ranking metrics (binary relevance) and list-level statistics. Pure Python/NumPy."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np


def ranking_metrics(
    recs: Mapping[int, Sequence[int]], targets: Mapping[int, set[int]], ks: Sequence[int]
) -> dict[str, float]:
    """Mean recall/precision/NDCG/MAP @k over users in ``targets`` (users without recs score 0).

    ``map@k`` normalises by ``min(|targets|, k)``; ``ndcg@k`` uses log2 discount.
    """
    out = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "precision", "ndcg", "map")}
    n = len(targets)
    if n == 0:
        return out
    disc = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    for uid, tgt in targets.items():
        ranked = list(recs.get(uid, ()))
        nt = len(tgt)
        for k in ks:
            top = ranked[:k]
            rel = np.fromiter((1.0 if i in tgt else 0.0 for i in top), dtype=float, count=len(top))
            hits = float(rel.sum())
            out[f"recall@{k}"] += hits / nt
            out[f"precision@{k}"] += hits / k
            dcg = float((rel * disc[: len(rel)]).sum())
            idcg = float(disc[: min(nt, k)].sum())
            out[f"ndcg@{k}"] += dcg / idcg
            if hits:
                prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
                out[f"map@{k}"] += float((prec * rel).sum()) / min(nt, k)
    return {key: val / n for key, val in out.items()}


def list_stats(
    recs: Mapping[int, Sequence[int]],
    users: Sequence[int],
    k: int,
    catalog_size: int,
    head_items: set[int],
) -> dict[str, float]:
    """Catalog coverage, normalised entropy of recommended items and share of head items."""
    counts: Counter[int] = Counter()
    slots = head = 0
    for u in users:
        for item in list(recs.get(u, ()))[:k]:
            counts[item] += 1
            slots += 1
            head += item in head_items
    if slots == 0 or catalog_size <= 1:
        return {f"coverage@{k}": 0.0, f"entropy@{k}": 0.0, f"head_share@{k}": 0.0}
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    entropy = float(-(p * np.log(p)).sum() / math.log(catalog_size))
    return {
        f"coverage@{k}": len(counts) / catalog_size,
        f"entropy@{k}": entropy,
        f"head_share@{k}": head / slots,
    }
