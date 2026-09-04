import math

import pytest

from rss.metrics import evaluate, mrr, ndcg_at_k, recall_at_k

TOY_RANKING = ["d5", "d2", "d9", "d1", "d7", "d3", "d8", "d4", "d6", "d10", "d11"]


def test_gold_at_rank_1():
    ranking = ["gold", "d2", "d3"]
    assert ndcg_at_k(ranking, "gold", k=10) == pytest.approx(1.0)
    assert recall_at_k(ranking, "gold", k=10) == 1.0
    assert mrr(ranking, "gold") == pytest.approx(1.0)


def test_gold_at_rank_5():
    # rank 5 -> DCG = 1/log2(6), IDCG = 1/log2(2)
    expected_ndcg = (1 / math.log2(6)) / (1 / math.log2(2))
    assert ndcg_at_k(TOY_RANKING, "d7", k=10) == pytest.approx(expected_ndcg)
    assert recall_at_k(TOY_RANKING, "d7", k=10) == 1.0
    assert mrr(TOY_RANKING, "d7") == pytest.approx(1 / 5)


def test_k_cutoff_excludes_beyond_k():
    # d7 is at rank 5; k=3 should not see it
    assert ndcg_at_k(TOY_RANKING, "d7", k=3) == 0.0
    assert recall_at_k(TOY_RANKING, "d7", k=3) == 0.0
    # mrr has no k cutoff -- it should still find it
    assert mrr(TOY_RANKING, "d7") == pytest.approx(1 / 5)


def test_gold_absent_from_ranking():
    assert ndcg_at_k(TOY_RANKING, "not-there", k=10) == 0.0
    assert recall_at_k(TOY_RANKING, "not-there", k=10) == 0.0
    assert mrr(TOY_RANKING, "not-there") == 0.0


def test_empty_ranking():
    assert ndcg_at_k([], "gold", k=10) == 0.0
    assert recall_at_k([], "gold", k=10) == 0.0
    assert mrr([], "gold") == 0.0


def test_evaluate_aggregates_mean_and_per_query():
    run = {
        "q1": ["gold1", "d2", "d3"],       # perfect
        "q2": ["d1", "d2", "gold2", "d4"], # rank 3
        "q3": ["d1", "d2", "d3"],          # miss
    }
    qrels = {"q1": "gold1", "q2": "gold2", "q3": "gold3"}

    result = evaluate(run, qrels, k_values=[1, 10])

    assert result["n_queries"] == 3
    assert set(result["per_query"].keys()) == {"q1", "q2", "q3"}

    # q1: perfect at rank 1
    assert result["per_query"]["q1"]["ndcg@10"] == pytest.approx(1.0)
    assert result["per_query"]["q1"]["recall@1"] == 1.0

    # q2: found at rank 3 -- inside k=10 but not k=1
    assert result["per_query"]["q2"]["recall@1"] == 0.0
    assert result["per_query"]["q2"]["recall@10"] == 1.0
    assert result["per_query"]["q2"]["mrr"] == pytest.approx(1 / 3)

    # q3: gold not in the ranking at all
    assert result["per_query"]["q3"]["ndcg@10"] == 0.0
    assert result["per_query"]["q3"]["mrr"] == 0.0

    # mean is the simple average across the three queries
    expected_mean_mrr = (1.0 + (1 / 3) + 0.0) / 3
    assert result["mean"]["mrr"] == pytest.approx(expected_mean_mrr)


def test_evaluate_empty_run_returns_empty_aggregates():
    result = evaluate({}, {}, k_values=[10])
    assert result == {"per_query": {}, "mean": {}, "n_queries": 0}
