"""Command line entry point: ``python -m recops.cli <command> --profile <p> --workdir <dir>``."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit recommends this
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from recops.config import load_config  # noqa: E402

COMMANDS = [
    "download",
    "prepare",
    "validate",
    "simulate",
    "final",
    "bench",
    "report",
    "serve",
    "all",
]


def _validate(cfg) -> None:
    import polars as pl

    from recops.data.validate import validate_events

    stats = validate_events(pl.read_parquet(Path(cfg.paths.data_processed) / "events.parquet"))
    out = Path(cfg.paths.artifacts) / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("validation ok:", stats)


def _bench(cfg) -> None:
    from recops.bundle import Bundle
    from recops.data.store import DataStore
    from recops.data.timeline import plan
    from recops.serving.bench import run_latency
    from recops.serving.service import Recommender

    art = Path(cfg.paths.artifacts)
    ds = DataStore.load(cfg)
    _, test = plan(ds.max_day, cfg.timeline)
    bundle = Bundle.load(art / "final" / "bundle")
    rec = Recommender.from_store(bundle, ds, test.t2, cfg, version="final")
    res = run_latency(rec, cfg.serving.latency_requests)
    (art / "latency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("latency:", res)


def run(command: str, cfg, force: bool = False) -> None:
    if command == "download":
        if cfg.data.synthetic:
            print("synthetic profile: nothing to download")
            return
        from recops.data.download import download_yambda

        download_yambda(cfg)
    elif command == "prepare":
        from recops.data.prepare import prepare

        prepare(cfg)
    elif command == "validate":
        _validate(cfg)
    elif command == "simulate":
        from recops.pipelines.cycle import simulate

        simulate(cfg)
    elif command == "final":
        from recops.pipelines.final import final_eval

        final_eval(cfg, force=force)
    elif command == "bench":
        _bench(cfg)
    elif command == "report":
        from recops.report import build_report

        print("report:", build_report(cfg))
    elif command == "serve":
        import uvicorn

        os.environ.setdefault("RECOPS_PROFILE", "full")
        uvicorn.run(
            "recops.serving.app:factory", factory=True, host=cfg.serving.host, port=cfg.serving.port
        )
    elif command == "all":
        for c in ("download", "prepare", "validate", "simulate", "final", "bench", "report"):
            print(f"\n######## {c} ########")
            run(c, cfg, force)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="recops")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--profile", default="full", help="full | smoke | synthetic")
    p.add_argument("--workdir", default=None, help="root for data/ and artifacts/")
    p.add_argument("--force", action="store_true", help="re-run final test evaluation")
    a = p.parse_args(argv)
    cfg = load_config(a.profile, a.workdir)
    run(a.command, cfg, a.force)


if __name__ == "__main__":
    main()
