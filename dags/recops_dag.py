"""Airflow DAG: one weekly cycle. UNTESTED on Airflow here (Kaggle has no Docker/Airflow).

Run locally via docker-compose. The same functions are executed by ``recops.cli simulate``
on Kaggle. Trigger with conf: {"cycle": 0}. In production the cycle index would be derived
from the logical date; here the data is simulated time (weekly windows).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROFILE = os.environ.get("RECOPS_PROFILE", "full")
WORKDIR = os.environ.get("RECOPS_WORKDIR", "/opt/recops_run")


def _ctx(cycle: int):
    from recops.config import load_config
    from recops.data.store import DataStore
    from recops.data.timeline import plan

    cfg = load_config(PROFILE, WORKDIR)
    ds = DataStore.load(cfg)
    cycles, _ = plan(ds.max_day, cfg.timeline)
    return cfg, ds, cycles[cycle]


def _cycle(**kw) -> int:
    return int(kw["dag_run"].conf.get("cycle", 0))


def validate(**kw):
    from recops.cli import _validate
    from recops.config import load_config

    _validate(load_config(PROFILE, WORKDIR))


def train(**kw):
    from recops.pipelines.cycle import train_step

    cfg, ds, win = _ctx(_cycle(**kw))
    return str(train_step(ds, win, _cycle(**kw), cfg))


def decide(ti=None, **kw):
    from recops.pipelines.cycle import decide_step
    from recops.registry import Registry

    k = _cycle(**kw)
    cfg, ds, win = _ctx(k)
    bundle_dir = Path(ti.xcom_pull(task_ids="train_candidate"))
    return decide_step(ds, win, k, cfg, Registry(cfg), bundle_dir)["promoted"]


with DAG(
    "recops_cycle",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["recops"],
) as dag:
    t_val = PythonOperator(task_id="validate_data", python_callable=validate)
    t_train = PythonOperator(task_id="train_candidate", python_callable=train)
    t_dec = PythonOperator(task_id="evaluate_compare_promote", python_callable=decide)
    t_val >> t_train >> t_dec
