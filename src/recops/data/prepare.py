"""Turn raw Yambda parquet files (or synthetic data) into the processed event tables."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from recops.config import Config
from recops.data.synthetic import make_synthetic


def _from_raw(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = cfg.paths.data_raw / cfg.data.subdir
    for name in ("listens", "likes"):
        if not (raw / f"{name}.parquet").exists():
            raise FileNotFoundError(f"{raw / (name + '.parquet')} missing: run `download` first")
    ev = (
        pl.scan_parquet(raw / "listens.parquet")
        .select(
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Int64).alias("ts"),
            pl.col("is_organic").cast(pl.Int64),
            pl.col("played_ratio_pct").cast(pl.Int64).alias("played_pct"),
        )
        .collect()
    )
    lk = (
        pl.scan_parquet(raw / "likes.parquet")
        .select(
            pl.col("uid").cast(pl.Int64),
            pl.col("item_id").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Int64).alias("ts"),
        )
        .collect()
    )
    return ev, lk


def prepare(cfg: Config) -> dict:
    """Build ``events.parquet`` / ``likes.parquet`` / ``meta.json`` in ``paths.data_processed``.

    Columns of events: uid, item_id, ts (5 s bins), day (0 = first day), is_organic, played_pct.
    """
    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    upd = cfg.data.ts_units_per_day

    if cfg.data.synthetic:
        ev, lk = make_synthetic(units_per_day=upd, seed=cfg.seed)
    else:
        ev, lk = _from_raw(cfg)
        ev = ev.with_columns((pl.col("ts") // upd).alias("day"))
        lk = lk.with_columns((pl.col("ts") // upd).alias("day"))

    if cfg.data.max_users is not None:
        users = np.sort(ev["uid"].unique().to_numpy())
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(users, size=min(cfg.data.max_users, len(users)), replace=False)
        keep_df = pl.DataFrame({"uid": np.sort(keep)})
        ev = ev.join(keep_df, on="uid", how="semi")
        lk = lk.join(keep_df, on="uid", how="semi")

    if cfg.data.last_days is not None:
        max_day = int(ev["day"].max())
        ev = ev.filter(pl.col("day") > max_day - cfg.data.last_days)
        lk = lk.filter(pl.col("day") > max_day - cfg.data.last_days)

    span = int(ev["day"].max()) - int(ev["day"].min()) + 1
    print(f"data span = {span} days under ts_units_per_day={upd}")
    if not cfg.data.synthetic and span > 400:
        print(
            "WARNING: Yambda spans ~300 days; span looks too long, ts_units_per_day is probably wrong"
        )
    day0 = int(ev["day"].min())
    ev = ev.with_columns(pl.col("day") - day0).sort(["uid", "ts"])
    lk = lk.with_columns(pl.col("day") - day0).filter(pl.col("day") >= 0)

    ev.write_parquet(out / "events.parquet")
    lk.write_parquet(out / "likes.parquet")
    meta = {
        "n_events": ev.height,
        "n_likes": lk.height,
        "n_users": int(ev["uid"].n_unique()),
        "n_items": int(ev["item_id"].n_unique()),
        "max_day": int(ev["day"].max()),
        "day0_shift": day0,
        "synthetic": cfg.data.synthetic,
        "subsampled_users": cfg.data.max_users,
        "last_days": cfg.data.last_days,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("prepared:", meta)
    return meta
