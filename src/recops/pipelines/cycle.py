"""One simulated production cycle (train -> evaluate -> compare -> promote) and the simulation."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from recops.bundle import Bundle
from recops.config import Config
from recops.data.store import DataStore
from recops.data.timeline import Windows, plan
from recops.data.validate import validate_events
from recops.monitoring.drift import drift_report, quality_alert
from recops.pipelines.core import (
    candidate_recall,
    evaluate_scored,
    pick_eval_users,
    score_users,
    train_bundle,
    warm_users,
)
from recops.promotion import decide
from recops.registry import Registry


def _sample_ref(sc, cfg: Config, seed: int):
    feats = sc.feats
    n = min(cfg.monitoring.reference_sample, feats.height)
    idx = np.sort(np.random.default_rng(seed).choice(feats.height, size=n, replace=False))
    sub = feats.select(pl.all().gather(idx.tolist()))
    return sub, sub["score"].to_numpy()


def train_step(ds: DataStore, win: Windows, k: int, cfg: Config) -> Path:
    """Airflow-friendly step: validate the data slice, train candidate bundle, save to disk."""
    validate_events(ds.events.filter(ds.events["day"] < win.t2))
    bad = k == cfg.simulation.bad_candidate_cycle
    t0 = time.time()
    bundle = train_bundle(ds, win, cfg, cfg.seed + k, corrupt=bad)
    bundle.meta["train_seconds"] = time.time() - t0
    out = Path(cfg.paths.artifacts) / "cycles" / f"cycle_{k}" / "bundle"
    bundle.save(out)
    return out


def decide_step(
    ds: DataStore, win: Windows, k: int, cfg: Config, reg: Registry, bundle_dir: Path
) -> dict:
    """Evaluate candidate vs production on the same holdout, monitor, decide, log, promote."""
    prim = cfg.metrics.primary
    cand = Bundle.load(bundle_dir)
    view = ds.view(win.t2)
    targets = ds.labels(win.h_start, win.h_end, seen_before=win.t2)
    warm = warm_users(ds.pos_hist(win.t2), cfg.evaluation.cold_min_history)
    users = pick_eval_users(targets, warm, cfg, cfg.seed + k)

    sc_c = score_users(cand, view, users, cfg, all_systems=True)
    m_c = evaluate_scored(sc_c, targets, users, view, cfg)
    cand_metric = m_c["ranker"][prim]

    prod = reg.load_production()
    prod_metric = None
    monitoring: dict = {"alerts": []}
    m_p = None
    if prod is not None:
        sc_p = score_users(prod, view, users, cfg)
        m_p = evaluate_scored(sc_p, targets, users, view, cfg)
        prod_metric = m_p["ranker"][prim]
        monitoring = drift_report(prod.ref_feats, sc_p.feats, prod.ref_scores, sc_p.scores, cfg)
        ref = prod.meta.get("holdout_primary")
        if ref:
            alert, drop = quality_alert(prod_metric, ref, cfg.monitoring.quality_drop_alert)
            monitoring["quality_drop_vs_reference"] = drop
            if alert:
                monitoring["alerts"].append(
                    f"QUALITY alert: production {prim} {prod_metric:.4f} vs reference {ref:.4f}"
                )

    promote, reason = decide(cand_metric, prod_metric, cfg.promotion.min_relative_improvement)
    cand.meta["holdout_primary"] = cand_metric
    cand.ref_feats, cand.ref_scores = _sample_ref(sc_c, cfg, cfg.seed + k)
    cand.save_refs(bundle_dir)

    record = {
        "cycle": k,
        "windows": win.__dict__,
        "drifted_holdout": cfg.simulation.drift_cycle is not None
        and k >= cfg.simulation.drift_cycle,
        "corrupted_candidate": bool(cand.meta.get("corrupted")),
        "n_eval_users": int(len(users)),
        "primary_metric": prim,
        "candidate": cand_metric,
        "production": prod_metric,
        "promoted": promote,
        "reason": reason,
        "systems_candidate": m_c,
        "production_metrics": m_p["ranker"] if m_p else None,
        "candidate_recall": candidate_recall(sc_c.cands.frames, targets, users),
        "monitoring": monitoring,
    }
    out_dir = Path(cfg.paths.artifacts) / "cycles"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"cycle_{k}.json"
    rec_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    flat = {f"cand_{n}": v for n, v in m_c["ranker"].items()}
    if m_p:
        flat.update({f"prod_{n}": v for n, v in m_p["ranker"].items()})
    for f, v in monitoring.get("psi", {}).items():
        flat[f"psi_{f}"] = v
    version = reg.log_cycle(
        k,
        params={"cycle": k, **win.__dict__, "corrupted": cand.meta.get("corrupted")},
        metrics=flat,
        tags={"decision": "promoted" if promote else "rejected", "reason": reason},
        files=[rec_path],
        bundle_dir=bundle_dir if promote else None,
    )
    record["registered_version"] = version
    rec_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    shutil.rmtree(bundle_dir.parent, ignore_errors=True)  # bundle lives in the registry now
    return record


def simulate(cfg: Config) -> list[dict]:
    """Run all cycles on (optionally drifted) data. Test window is never used here."""
    ds_clean = DataStore.load(cfg)
    cycles, _ = plan(ds_clean.max_day, cfg.timeline)
    ds = ds_clean
    if cfg.simulation.drift_cycle is not None:
        day = cycles[cfg.simulation.drift_cycle].h_start
        ds = ds_clean.drifted(day, cfg.simulation.drift_top_items, cfg.seed)
        print(
            f"artificial drift starts at day {day} (holdout of cycle {cfg.simulation.drift_cycle})"
        )
        del ds_clean
    reg = Registry(cfg)
    records = []
    for k, win in enumerate(cycles):
        print(f"\n=== cycle {k}: {win} ===")
        bundle_dir = train_step(ds, win, k, cfg)
        rec = decide_step(ds, win, k, cfg, reg, bundle_dir)
        print(
            f"cycle {k}: candidate={rec['candidate']:.4f} production={rec['production']} "
            f"promoted={rec['promoted']} | {rec['reason']} | alerts={rec['monitoring']['alerts']}"
        )
        records.append(rec)
    summary = Path(cfg.paths.artifacts) / "simulation.json"
    summary.write_text(json.dumps(records, indent=2, default=float), encoding="utf-8")
    return records
