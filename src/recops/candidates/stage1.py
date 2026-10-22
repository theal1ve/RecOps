"""Stage 1: candidate generators (popularity, ItemKNN, ALS) behind one fitted object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from recops.config import Config

EMPTY_SCHEMA = {"uid": pl.Int64, "item_id": pl.Int64, "score": pl.Float64, "rank": pl.Int64}


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=EMPTY_SCHEMA)


@dataclass
class Catalog:
    """Sorted item ids <-> contiguous matrix column indices."""

    item_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        """Column index for each id, or -1 for items outside the catalog."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(self.item_ids) == 0:
            return np.full(len(ids), -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.item_ids, ids), len(self.item_ids) - 1)
        return np.where(self.item_ids[pos] == ids, pos, -1)


def build_matrices(
    pos: pl.DataFrame,
    users: np.ndarray,
    catalog: Catalog,
    alpha: float,
    max_items: int | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix | None]:
    """User x item matrices from positive listens (columns: uid, item_id, ts).

    Returns ``(confidence, binary, binary_cropped)``: ALS confidence ``1 + alpha*log1p(n)``;
    binary membership; and (if ``max_items``) a binary matrix keeping only each user's
    ``max_items`` most recent items (used to fit ItemKNN in bounded time).
    ``users`` must be sorted and unique; it defines the row order.
    """
    shape = (len(users), len(catalog))
    agg = pos.group_by(["uid", "item_id"]).agg(
        pl.len().alias("n"), pl.col("ts").max().alias("last_ts")
    )
    if agg.height == 0 or len(users) == 0 or len(catalog) == 0:
        z = sparse.csr_matrix(shape, dtype=np.float32)
        return z, z.copy(), z.copy() if max_items else None
    uid = agg["uid"].to_numpy()
    rows = np.minimum(np.searchsorted(users, uid), len(users) - 1)
    ok = (users[rows] == uid) & (catalog.lookup(agg["item_id"].to_numpy()) >= 0)
    rows = rows[ok]
    cols = catalog.lookup(agg["item_id"].to_numpy())[ok]
    n = agg["n"].to_numpy()[ok].astype(np.float64)
    last = agg["last_ts"].to_numpy()[ok]

    conf = sparse.csr_matrix(
        ((1.0 + alpha * np.log1p(n)).astype(np.float32), (rows, cols)), shape=shape
    )
    ones = np.ones(len(rows), dtype=np.float32)
    binm = sparse.csr_matrix((ones, (rows, cols)), shape=shape)
    crop = None
    if max_items:
        order = np.lexsort((-last, rows))
        r_s = rows[order]
        first = np.searchsorted(r_s, r_s, side="left")
        keep = (np.arange(len(r_s)) - first) < max_items
        sel = order[keep]
        crop = sparse.csr_matrix((ones[sel], (rows[sel], cols[sel])), shape=shape)
    return conf, binm, crop


@dataclass
class Candidates:
    """Per-source candidate frames plus the warm/cold split of the requested users."""

    frames: dict[str, pl.DataFrame]
    warm: np.ndarray
    cold: np.ndarray


class Stage1:
    """Fitted ALS + ItemKNN on a catalog, plus popularity logic (data-driven, not fitted)."""

    def __init__(self, catalog: Catalog, als, knn):
        self.catalog = catalog
        self.als = als
        self.knn = knn

    @classmethod
    def fit(cls, pos_hist: pl.DataFrame, cfg: Config, seed: int) -> Stage1:
        """Fit on positive listens (columns uid, item_id, ts, day) before the cutoff."""
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import CosineRecommender

        counts = pos_hist.group_by("item_id").agg(pl.len().alias("c"))
        counts = counts.filter(pl.col("c") >= cfg.data.min_item_count)
        catalog = Catalog(np.sort(counts["item_id"].to_numpy().astype(np.int64)))
        users = np.sort(pos_hist["uid"].unique().to_numpy().astype(np.int64))
        c = cfg.candidates
        conf, _, crop = build_matrices(
            pos_hist, users, catalog, c.als.alpha, c.itemknn.max_items_per_user
        )
        als = AlternatingLeastSquares(
            factors=c.als.factors,
            regularization=c.als.regularization,
            iterations=c.als.iterations,
            random_state=seed,
            use_gpu=False,
            calculate_training_loss=False,
        )
        als.fit(conf, show_progress=False)
        knn = CosineRecommender(K=c.itemknn.neighbors)
        knn.fit(crop, show_progress=False)
        return cls(catalog, als, knn)

    # ------------------------------------------------------------------ generation
    def _recommend(self, model, mat, users, n, exclude_seen, recalc, batch=2000) -> pl.DataFrame:
        parts: list[pl.DataFrame] = []
        for s in range(0, len(users), batch):
            rows = mat[s : s + batch].astype(np.float32).tocsr()
            kwargs = {"recalculate_user": True} if recalc else {}
            # userid=arange + the matching rows: valid for both row- and id-indexed APIs.
            ids, scores = model.recommend(
                np.arange(rows.shape[0]),
                rows,
                N=n,
                filter_already_liked_items=exclude_seen,
                **kwargs,
            )
            width = ids.shape[1]
            uid = np.repeat(users[s : s + batch], width)
            rank = np.tile(np.arange(1, width + 1), ids.shape[0])
            ids_f, sc_f = ids.ravel(), scores.ravel()
            ok = (ids_f >= 0) & np.isfinite(sc_f)
            parts.append(
                pl.DataFrame(
                    {
                        "uid": uid[ok].astype(np.int64),
                        "item_id": self.catalog.item_ids[ids_f[ok]],
                        "score": sc_f[ok].astype(np.float64),
                        "rank": rank[ok].astype(np.int64),
                    }
                )
            )
        return pl.concat(parts) if parts else empty_frame()

    def _popularity(self, view, binm, users, n, exclude_seen, batch=2000) -> pl.DataFrame:
        idx = self.catalog.lookup(view.pop_items)
        ok = idx >= 0
        items, scores, idx = view.pop_items[ok], view.pop_scores[ok], idx[ok]
        if len(items) == 0:
            return empty_frame()
        parts: list[pl.DataFrame] = []
        for s in range(0, len(users), batch):
            if exclude_seen:
                seen = binm[s : s + batch][:, idx].toarray() > 0
            else:
                seen = np.zeros((min(batch, len(users) - s), len(idx)), dtype=bool)
            keep = ~seen
            csum = np.cumsum(keep, axis=1)
            sel = keep & (csum <= n)
            r, c = np.nonzero(sel)
            parts.append(
                pl.DataFrame(
                    {
                        "uid": users[s + r].astype(np.int64),
                        "item_id": items[c].astype(np.int64),
                        "score": scores[c].astype(np.float64),
                        "rank": csum[r, c].astype(np.int64),
                    }
                )
            )
        return pl.concat(parts)

    def candidates(self, view, users: np.ndarray, cfg: Config, warm_min: int | None = None):
        """Candidates for ``users`` from the history in ``view``.

        ALS and ItemKNN are used for warm users (>= ``cold_min_history`` distinct positives);
        popularity covers everyone (it is the cold-start fallback).
        ALS user vectors are recomputed from the current history (``recalculate_user``), so a
        frozen production model can still serve users with fresh listens.
        """
        users = np.unique(np.asarray(users, dtype=np.int64))
        pos = view.ev.filter(pl.col("pos")).select("uid", "item_id", "ts")
        conf, binm, _ = build_matrices(pos, users, self.catalog, cfg.candidates.als.alpha, None)
        nnz = np.asarray(binm.getnnz(axis=1)).ravel()
        wm = cfg.evaluation.cold_min_history if warm_min is None else warm_min
        warm_mask = nnz >= wm
        widx = np.flatnonzero(warm_mask)
        n = cfg.candidates.per_source_k
        ex = cfg.evaluation.exclude_seen
        if len(widx):
            als = self._recommend(self.als, conf[widx], users[widx], n, ex, True)
            knn = self._recommend(self.knn, binm[widx], users[widx], n, ex, False)
        else:
            als = knn = empty_frame()
        pop = self._popularity(view, binm, users, n, ex)
        return Candidates({"pop": pop, "knn": knn, "als": als}, users[warm_mask], users[~warm_mask])


def top_k(df: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    """Top-``k`` rows per user by ``score_col`` with a 1-based ``rank`` column."""
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.Int64).alias("rank"))
    return (
        df.with_columns(
            pl.col(score_col)
            .rank("ordinal", descending=True)
            .over("uid")
            .cast(pl.Int64)
            .alias("rank")
        )
        .filter(pl.col("rank") <= k)
        .sort(["uid", "rank"])
    )


def rrf(frames: dict[str, pl.DataFrame], k: int, c: int = 60) -> pl.DataFrame:
    """Reciprocal Rank Fusion of source frames (no learning): the naive blending baseline."""
    parts = [f.select("uid", "item_id", "rank") for f in frames.values() if f.height]
    if not parts:
        return empty_frame()
    fused = (
        pl.concat(parts)
        .group_by(["uid", "item_id"])
        .agg((1.0 / (c + pl.col("rank"))).sum().alias("score"))
    )
    return top_k(fused, "score", k)
