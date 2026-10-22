"""A deployable model bundle: stage-1 models + ranker + monitoring reference."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from recops.candidates.stage1 import Stage1


@dataclass
class Bundle:
    stage1: Stage1
    ranker: object  # catboost.CatBoostRanker
    meta: dict = field(default_factory=dict)
    ref_feats: pl.DataFrame | None = None
    ref_scores: np.ndarray | None = None
    extra_rankers: dict = field(default_factory=dict)  # ablation variants (not persisted)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "stage1.pkl", "wb") as f:
            pickle.dump(self.stage1, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.ranker.save_model(str(d / "ranker.cbm"))
        self.save_refs(d)

    def save_refs(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(
            json.dumps(self.meta, indent=2, default=float), encoding="utf-8"
        )
        if self.ref_feats is not None:
            self.ref_feats.write_parquet(d / "ref_feats.parquet")
        if self.ref_scores is not None:
            np.save(d / "ref_scores.npy", self.ref_scores)

    @classmethod
    def load(cls, d: Path) -> Bundle:
        from catboost import CatBoostRanker

        with open(d / "stage1.pkl", "rb") as f:
            stage1 = pickle.load(f)  # noqa: S301 - our own artifact
        ranker = CatBoostRanker()
        ranker.load_model(str(d / "ranker.cbm"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        rf = d / "ref_feats.parquet"
        rs = d / "ref_scores.npy"
        return cls(
            stage1,
            ranker,
            meta,
            pl.read_parquet(rf) if rf.exists() else None,
            np.load(rs) if rs.exists() else None,
        )
