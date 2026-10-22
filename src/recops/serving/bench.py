"""Latency / throughput measurement of the API (in-process, via Starlette TestClient)."""

from __future__ import annotations

import time

import numpy as np

from recops.serving.app import create_app
from recops.serving.service import Recommender


def run_latency(rec: Recommender, n_requests: int, k: int = 20, seed: int = 0) -> dict:
    """Sequential POST /recommend timings. Includes HTTP-stack overhead of TestClient but not
    network. Measured on whatever machine runs it (Kaggle CPU), not a production host."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app(rec))
    users = list(rec.user_ev)
    rng = np.random.default_rng(seed)
    picks = rng.choice(users, size=n_requests, replace=True)
    for u in picks[:5]:  # warmup
        client.post("/recommend", json={"user_id": int(u), "k": k})
    lat = []
    t_all = time.perf_counter()
    for u in picks:
        t0 = time.perf_counter()
        r = client.post("/recommend", json={"user_id": int(u), "k": k})
        lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text
    total = time.perf_counter() - t_all
    t0 = time.perf_counter()
    r = client.post("/recommend", json={"user_id": -1, "k": k})
    fallback_ms = (time.perf_counter() - t0) * 1000
    assert client.get("/health").status_code == 200
    lat_arr = np.array(lat)
    return {
        "n_requests": int(n_requests),
        "k": k,
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "mean_ms": float(lat_arr.mean()),
        "throughput_rps_sequential": float(n_requests / total),
        "fallback_ms": fallback_ms,
        "fallback_source": r.json()["source"],
        "note": "sequential in-process TestClient; single worker; CPU of the host that ran it",
    }
