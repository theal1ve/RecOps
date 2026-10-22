"""Typed, validated project configuration: base YAML + optional profile overlay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_DIR = Path(
    os.environ.get("RECOPS_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
)


class _Strict(BaseModel):
    """Unknown keys are errors, so a typo in YAML fails fast."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_raw: Path
    data_processed: Path
    artifacts: Path


class DataConfig(_Strict):
    dataset_repo: str
    subdir: str
    positive_min_played_pct: int = Field(ge=1, le=100)
    ts_units_per_day: int = Field(gt=0)
    max_users: int | None = Field(default=None, gt=0)
    last_days: int | None = Field(default=None, gt=0)
    min_item_count: int = Field(ge=1)
    synthetic: bool = False


class TimelineConfig(_Strict):
    eval_days: int = Field(gt=0)
    ranker_label_days: int = Field(gt=0)
    gap_days: int = Field(ge=0)
    step_days: int = Field(gt=0)
    n_cycles: int = Field(gt=0)
    test_days: int = Field(gt=0)
    train_window_days: int | None = Field(default=None, gt=0)
    min_history_days: int = Field(ge=0)


class EvaluationConfig(_Strict):
    exclude_seen: bool
    cold_min_history: int = Field(ge=1)
    max_users: int = Field(gt=0)
    pop_head_frac: float = Field(gt=0, lt=1)
    cold_sim_frac: float = Field(gt=0, le=1)
    cold_sim_events: int = Field(ge=1)


class PopularityConfig(_Strict):
    window_days: int = Field(gt=0)


class ItemKNNConfig(_Strict):
    neighbors: int = Field(gt=0)
    max_items_per_user: int = Field(gt=0)


class ALSConfig(_Strict):
    factors: int = Field(gt=0)
    regularization: float = Field(ge=0)
    iterations: int = Field(gt=0)
    alpha: float = Field(ge=0)


class CandidatesConfig(_Strict):
    per_source_k: int = Field(gt=0)
    pop_list_size: int = Field(gt=0)
    popularity: PopularityConfig
    itemknn: ItemKNNConfig
    als: ALSConfig


class RankerConfig(_Strict):
    loss: str
    iterations: int = Field(gt=0)
    depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    negatives_per_positive: int = Field(gt=0)
    min_negatives: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    valid_user_frac: float = Field(gt=0, lt=1)
    thread_count: int


class MetricsConfig(_Strict):
    k_values: list[int]
    primary: str


class PromotionConfig(_Strict):
    min_relative_improvement: float = Field(ge=0)


class MonitoringConfig(_Strict):
    psi_warn: float = Field(gt=0)
    psi_alert: float = Field(gt=0)
    quality_drop_alert: float = Field(gt=0)
    psi_bins: int = Field(ge=2)
    reference_sample: int = Field(gt=0)
    psi_features: list[str]


class SimulationConfig(_Strict):
    drift_cycle: int | None = None
    drift_top_items: int = Field(gt=0)
    bad_candidate_cycle: int | None = None


class MlflowConfig(_Strict):
    experiment: str
    model_name: str
    alias: str


class ServingConfig(_Strict):
    host: str
    port: int = Field(gt=0, lt=65536)
    snapshot_users: int = Field(gt=0)
    latency_requests: int = Field(gt=0)


class Config(_Strict):
    seed: int
    paths: PathsConfig
    data: DataConfig
    timeline: TimelineConfig
    evaluation: EvaluationConfig
    candidates: CandidatesConfig
    ranker: RankerConfig
    metrics: MetricsConfig
    promotion: PromotionConfig
    monitoring: MonitoringConfig
    simulation: SimulationConfig
    mlflow: MlflowConfig
    serving: ServingConfig

    @property
    def primary_metric(self) -> str:
        return self.metrics.primary


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(profile: str = "full", workdir: str | Path | None = None) -> Config:
    """Load ``configs/base.yaml`` merged with ``configs/profiles/<profile>.yaml``.

    If ``workdir`` is given, all paths are placed under it (used on Kaggle).
    """
    raw = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    if profile != "base":
        overlay = yaml.safe_load(
            (CONFIG_DIR / "profiles" / f"{profile}.yaml").read_text(encoding="utf-8")
        )
        raw = _deep_merge(raw, overlay or {})
    if workdir is not None:
        wd = Path(workdir)
        raw["paths"] = {
            "data_raw": str(wd / "data" / "raw"),
            "data_processed": str(wd / "data" / "processed"),
            "artifacts": str(wd / "artifacts"),
        }
    return Config.model_validate(raw)
