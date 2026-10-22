import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("polars")

from fastapi.testclient import TestClient  # noqa: E402

from recops.serving.app import create_app  # noqa: E402


class FakeRecommender:
    version = "test"
    user_ev = {1: object()}

    def recommend(self, uid: int, k: int):
        if uid == 1:
            return [(i, 1.0 / (i + 1)) for i in range(k)], "ranker"
        return [(7, 3.0)], "fallback_popularity"


def test_health_and_recommend() -> None:
    c = TestClient(create_app(FakeRecommender()))
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/recommend", json={"user_id": 1, "k": 3}).json()
    assert r["source"] == "ranker" and len(r["items"]) == 3
    assert (
        c.post("/recommend", json={"user_id": 99, "k": 3}).json()["source"] == "fallback_popularity"
    )


def test_validation_errors() -> None:
    c = TestClient(create_app(FakeRecommender()))
    assert c.post("/recommend", json={"user_id": 1, "k": 0}).status_code == 422
    assert c.post("/recommend", json={"k": 5}).status_code == 422
