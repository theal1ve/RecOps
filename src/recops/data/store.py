"""In-memory event store with temporal slicing, labels and artificial drift."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from recops.config import Config


def in_users(df: pl.DataFrame, users: np.ndarray, col: str = "uid") -> pl.DataFrame:
    """Keep rows whose ``col`` is in ``users`` (semi-join: avoids is_in API quirks)."""
    return df.join(pl.DataFrame({col: np.asarray(users, dtype=np.int64)}), on=col, how="semi")


class DataStore:
    """Holds listens (with a ``pos`` = Listen+ flag) and likes."""

    def __init__(self, events: pl.DataFrame, likes: pl.DataFrame, meta: dict, cfg: Config):
        thr = cfg.data.positive_min_played_pct
        self.events = events.with_columns((pl.col("played_pct") >= thr).alias("pos"))
        self.likes = likes
        self.meta = meta
        self.cfg = cfg
        self._views: dict[int, object] = {}

    @classmethod
    def load(cls, cfg: Config) -> DataStore:
        d = cfg.paths.data_processed
        ev = pl.read_parquet(d / "events.parquet")
        lk = pl.read_parquet(d / "likes.parquet")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(ev, lk, meta, cfg)

    @property
    def max_day(self) -> int:
        return int(self.events["day"].max())

    def pos_hist(self, as_of: int, window: int | None = None) -> pl.DataFrame:
        """Positive listens with ``day < as_of`` (optionally only the last ``window`` days)."""
        lo = 0 if window is None else as_of - window
        return self.events.filter(
            pl.col("pos") & (pl.col("day") < as_of) & (pl.col("day") >= lo)
        ).select("uid", "item_id", "ts", "day")

    def labels(self, start: int, end: int, seen_before: int) -> pl.DataFrame:
        """Unique positive (uid, item_id) pairs in ``[start, end)``.

        With ``exclude_seen`` pairs the user already liked before ``seen_before`` are removed
        (repeat listens are not counted as hits in the discovery task).
        """
        tgt = (
            self.events.filter(pl.col("pos") & (pl.col("day") >= start) & (pl.col("day") < end))
            .select("uid", "item_id")
            .unique()
        )
        if not self.cfg.evaluation.exclude_seen or tgt.height == 0:
            return tgt
        seen = (
            self.events.filter(pl.col("pos") & (pl.col("day") < seen_before))
            .select("uid", "item_id")
            .join(tgt, on=["uid", "item_id"], how="semi")
            .unique()
        )
        return tgt.join(seen, on=["uid", "item_id"], how="anti")

    def view(self, as_of: int):
        """Cached :class:`recops.view.HistoryView` for ``as_of`` (keeps the last two)."""
        from recops.view import build_view

        if as_of not in self._views:
            if len(self._views) >= 2:
                self._views.pop(next(iter(self._views)))
            self._views[as_of] = build_view(self, as_of)
        return self._views[as_of]

    def drifted(self, drift_day: int, top_items: int, seed: int) -> DataStore:
        """Return a copy where, from ``drift_day`` on, the top-N popular items are permuted.

        This is an artificial popularity/taste shift: after ``drift_day`` users' listens of
        item A appear as listens of item pi(A), which breaks models trained before it.
        """
        top = (
            self.events.filter(pl.col("pos") & (pl.col("day") < drift_day))
            .group_by("item_id")
            .agg(pl.len().alias("c"))
            .sort(["c", "item_id"], descending=[True, False])
            .head(top_items)
        )
        items = top["item_id"].to_numpy()
        rng = np.random.default_rng(seed)
        mapping = pl.DataFrame(
            {"item_id": items, "new_item_id": items[rng.permutation(len(items))]}
        )

        def remap(df: pl.DataFrame) -> pl.DataFrame:
            return (
                df.join(mapping, on="item_id", how="left")
                .with_columns(
                    pl.when(pl.col("day") >= drift_day)
                    .then(pl.coalesce(["new_item_id", "item_id"]))
                    .otherwise(pl.col("item_id"))
                    .alias("item_id")
                )
                .drop("new_item_id")
            )

        meta = dict(self.meta, drift_day=drift_day)
        return DataStore(remap(self.events), remap(self.likes), meta, self.cfg)
