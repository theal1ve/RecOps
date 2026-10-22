"""FastAPI service: POST /recommend, GET /health."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recops.serving.service import Recommender


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=10, ge=1, le=50)


class ScoredItem(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    source: str
    model_version: str
    items: list[ScoredItem]


def create_app(rec: Recommender) -> FastAPI:
    """Build the app around a ready :class:`Recommender`."""
    app = FastAPI(title="RecOps")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": rec.version, "users_in_snapshot": len(rec.user_ev)}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        items, source = rec.recommend(req.user_id, req.k)
        return RecommendResponse(
            user_id=req.user_id,
            source=source,
            model_version=rec.version,
            items=[ScoredItem(item_id=i, score=s) for i, s in items],
        )

    return app


def factory() -> FastAPI:
    """``uvicorn --factory recops.serving.app:factory``.

    Env: RECOPS_PROFILE, RECOPS_WORKDIR; RECOPS_BUNDLE_DIR (optional, else the MLflow
    Registry ``production`` alias is loaded).
    """
    from recops.bundle import Bundle
    from recops.config import load_config
    from recops.data.store import DataStore
    from recops.data.timeline import plan

    cfg = load_config(os.environ.get("RECOPS_PROFILE", "full"), os.environ.get("RECOPS_WORKDIR"))
    bundle_dir = os.environ.get("RECOPS_BUNDLE_DIR")
    if bundle_dir:
        bundle, version = Bundle.load(Path(bundle_dir)), "local-bundle"
    else:
        from recops.registry import Registry

        reg = Registry(cfg)
        mv = reg.production_version()
        if mv is None:
            raise RuntimeError("no production model in the registry")
        bundle, version = reg.load_production(), f"registry-v{mv.version}"
    ds = DataStore.load(cfg)
    _, test = plan(ds.max_day, cfg.timeline)
    return create_app(Recommender.from_store(bundle, ds, test.t2, cfg, version))
