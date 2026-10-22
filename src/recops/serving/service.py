"""Online recommender: frozen bundle + a snapshot of features/history (the 'feature store')."""

from __future__ import annotations

import numpy as np
import polars as pl

from recops.bundle import Bundle
from recops.candidates.stage1 import top_k
from recops.config import Config
from recops.data.store import DataStore, in_users
from recops.pipelines.core import score_users
from recops.view import HistoryView

_LK_SCHEMA = {"uid": pl.Int64, "item_id": pl.Int64, "ts": pl.Int64, "day": pl.Int64}


def _key(k) -> int:
    return int(k[0] if isinstance(k, tuple) else k)


class Recommender:
    """In-memory snapshot of per-user history (parquet -> dict) and shared item features.

    Same ``score_users`` code path as offline evaluation, so there is no train/serve skew.
    Storage choice: in-process dict of small polars frames; fine for a demo of a few thousand
    users. A real deployment would use Redis/a feature store (see README: limitations).
    """

    def __init__(
        self,
        bundle: Bundle,
        cfg: Config,
        view: HistoryView,
        user_ev: dict,
        user_lk: dict,
        version: str,
    ):
        self.bundle, self.cfg, self.version = bundle, cfg, version
        self.as_of = view.as_of
        self.item_feats = view.item_feats
        self.pop_items, self.pop_scores = view.pop_items, view.pop_scores
        self.user_ev, self.user_lk = user_ev, user_lk
        self._empty_lk = pl.DataFrame(schema=_LK_SCHEMA)

    @classmethod
    def from_store(
        cls, bundle: Bundle, ds: DataStore, as_of: int, cfg: Config, version: str = "local"
    ) -> Recommender:
        view = ds.view(as_of)
        users = np.sort(view.ev["uid"].unique().to_numpy())
        n = min(cfg.serving.snapshot_users, len(users))
        users = np.sort(np.random.default_rng(cfg.seed).choice(users, size=n, replace=False))
        ev = in_users(view.ev, users).partition_by("uid", as_dict=True)
        lk = in_users(view.lk, users).partition_by("uid", as_dict=True)
        return cls(
            bundle,
            cfg,
            view,
            {_key(k): v for k, v in ev.items()},
            {_key(k): v for k, v in lk.items()},
            version,
        )

    def _fallback(self, k: int) -> list[tuple[int, float]]:
        return [
            (int(i), float(s))
            for i, s in zip(self.pop_items[:k], self.pop_scores[:k], strict=False)
        ]

    def recommend(self, uid: int, k: int) -> tuple[list[tuple[int, float]], str]:
        """Return ``([(item_id, score)], source)``; unknown/cold users get popularity."""
        ev = self.user_ev.get(uid)
        if (
            ev is None
            or ev.filter(pl.col("pos"))["item_id"].n_unique() < self.cfg.evaluation.cold_min_history
        ):
            return self._fallback(k), "fallback_popularity"
        view = HistoryView(
            self.as_of,
            ev,
            self.user_lk.get(uid, self._empty_lk),
            self.item_feats,
            self.pop_items,
            self.pop_scores,
        )
        sc = score_users(self.bundle, view, np.array([uid], dtype=np.int64), self.cfg)
        if sc.feats.height == 0:
            return self._fallback(k), "fallback_popularity"
        top = top_k(sc.feats, "score", k)
        return list(zip(top["item_id"].to_list(), top["score"].to_list(), strict=True)), "ranker"
