"""End-to-end smoke on synthetic data: needs polars, implicit, catboost, mlflow (CI job 'smoke')."""

import json

import pytest

for mod in ("polars", "implicit", "catboost", "mlflow"):
    pytest.importorskip(mod)

pytestmark = pytest.mark.slow


def test_full_pipeline_on_synthetic(tmp_path) -> None:
    from recops.cli import run
    from recops.config import load_config

    cfg = load_config("synthetic", workdir=tmp_path)
    for step in ("prepare", "validate", "simulate", "final", "bench", "report"):
        run(step, cfg)
    sim = json.loads((tmp_path / "artifacts" / "simulation.json").read_text())
    assert sim[0]["promoted"] and sim[0]["production"] is None  # bootstrap
    assert sim[2]["corrupted_candidate"] and not sim[2]["promoted"]  # bad model is rejected
    fin = json.loads((tmp_path / "artifacts" / "final" / "results.json").read_text())
    m = fin["metrics"]
    assert m["als"]["ndcg@20"] > m["popularity"]["ndcg@20"]  # personalisation beats popularity
    assert (tmp_path / "artifacts" / "report.md").exists()
    with pytest.raises(RuntimeError):  # test window is evaluated once
        run("final", cfg)
