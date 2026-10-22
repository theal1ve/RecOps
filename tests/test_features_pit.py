"""Point-in-time correctness: rows from the future must not change any feature."""

import polars as pl
import pytest

pytest.importorskip("polars")

from recops.features.builder import item_features, ui_features, user_features  # noqa: E402


def _events(rows):
    return pl.DataFrame(
        rows, schema=["uid", "item_id", "ts", "day", "is_organic", "played_pct"], orient="row"
    ).with_columns((pl.col("played_pct") >= 50).alias("pos"))


BASE = [
    (1, 10, 100, 1, 1, 90), (1, 11, 200, 2, 0, 20), (2, 10, 300, 3, 1, 70), (1, 10, 400, 4, 1, 60),
]  # fmt: skip
FUTURE = [(1, 10, 900, 10, 1, 99), (3, 12, 950, 11, 0, 80), (1, 13, 999, 12, 1, 100)]
LIKES = pl.DataFrame({"uid": [1], "item_id": [10], "ts": [150], "day": [1]})
LIKES_FUT = pl.concat(
    [LIKES, pl.DataFrame({"uid": [1], "item_id": [10], "ts": [990], "day": [11]})]
)
AS_OF = 6


def test_user_features_ignore_future() -> None:
    a = user_features(_events(BASE), LIKES, AS_OF).sort("uid")
    b = user_features(_events(BASE + FUTURE), LIKES_FUT, AS_OF).sort("uid")
    assert a.equals(b)


def test_item_features_ignore_future() -> None:
    a = item_features(_events(BASE), LIKES, AS_OF).sort("item_id")
    b = item_features(_events(BASE + FUTURE), LIKES_FUT, AS_OF).sort("item_id")
    assert a.equals(b)


def test_ui_features_ignore_future_and_count_correctly() -> None:
    pairs = pl.DataFrame({"uid": [1, 1], "item_id": [10, 13]})
    a = ui_features(_events(BASE), pairs, AS_OF).sort(["uid", "item_id"])
    b = ui_features(_events(BASE + FUTURE), pairs, AS_OF).sort(["uid", "item_id"])
    assert a.equals(b)
    row = a.filter(pl.col("item_id") == 10).row(0, named=True)
    assert row["ui_n_pos"] == 2 and row["ui_n_skip"] == 0 and row["ui_days_since"] == AS_OF - 4
    assert a.filter(pl.col("item_id") == 13).height == 0  # item 13 only appears in the future


def test_values_are_as_expected() -> None:
    uf = user_features(_events(BASE), LIKES, AS_OF).filter(pl.col("uid") == 1).row(0, named=True)
    assert uf["u_n_listens"] == 3 and uf["u_n_pos"] == 2 and uf["u_n_likes"] == 1
    assert abs(uf["u_skip_rate"] - 1 / 3) < 1e-9 and uf["u_days_since_last"] == 2
