import math

from recops.evaluation.metrics import list_stats, ranking_metrics


def test_hand_computed_values() -> None:
    recs = {1: [10, 11, 12], 2: [20, 21, 22]}
    tg = {1: {11, 99}, 2: {20}}
    m = ranking_metrics(recs, tg, [3])
    u1_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(m["recall@3"], 0.75)
    assert math.isclose(m["ndcg@3"], (u1_ndcg + 1.0) / 2)
    assert math.isclose(m["map@3"], (0.25 + 1.0) / 2)


def test_missing_recs_score_zero_and_perfect_is_one() -> None:
    tg = {1: {5}}
    assert ranking_metrics({}, tg, [10])["ndcg@10"] == 0.0
    assert ranking_metrics({1: [5]}, tg, [10])["ndcg@10"] == 1.0


def test_list_stats() -> None:
    s = list_stats({1: [10, 11, 12], 2: [20, 21, 22]}, [1, 2], 2, 100, {10, 20})
    assert s["coverage@2"] == 0.04 and s["head_share@2"] == 0.5
