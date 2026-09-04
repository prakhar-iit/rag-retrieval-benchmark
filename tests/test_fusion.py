import pytest

from rss.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_favours_doc_ranked_high_in_both_runs():
    run_a = ["d1", "d2", "d3"]
    run_b = ["d2", "d1", "d4"]
    fused = reciprocal_rank_fusion([run_a, run_b])
    # d1 and d2 both appear near the top of both runs; d3/d4 only appear
    # once each near the bottom, so d1/d2 should outrank d3/d4.
    assert set(fused[:2]) == {"d1", "d2"}


def test_rrf_doc_present_in_every_run_beats_doc_missing_from_one():
    run_a = ["d1", "d2"]
    run_b = ["d3", "d1"]
    run_c = ["d1", "d3"]
    fused = reciprocal_rank_fusion([run_a, run_b, run_c])
    assert fused[0] == "d1"


def test_rrf_score_formula_matches_definition():
    # single run, single doc at rank 1 -> score = 1 / (k + 1)
    from collections import defaultdict

    k = 60
    run = ["only_doc"]
    fused = reciprocal_rank_fusion([run], k=k)
    assert fused == ["only_doc"]


def test_rrf_empty_runs_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_weighted_score_fusion_all_weight_on_one_method_reduces_to_that_method():
    run_a = {"d1": 10.0, "d2": 1.0}
    run_b = {"d1": 1.0, "d2": 10.0}
    fused = weighted_score_fusion([run_a, run_b], weights=[1.0, 0.0])
    assert fused[0] == "d1"


def test_weighted_score_fusion_combines_normalised_scores():
    # BM25-like unbounded scores vs cosine-like [-1, 1] scores -- normalisation
    # should prevent the larger-scale method from silently dominating.
    bm25 = {"d1": 40.0, "d2": 5.0}
    dense = {"d1": 0.2, "d2": 0.9}
    fused_equal = weighted_score_fusion([bm25, dense], weights=[0.5, 0.5])
    # d1 wins bm25 big, d2 wins dense big; with equal weight and normalisation
    # to [0,1], d1 gets 0.5*1.0 + 0.5*0.0 = 0.5, d2 gets 0.5*0.0 + 0.5*1.0 = 0.5 -- tie.
    assert set(fused_equal) == {"d1", "d2"}


def test_weighted_score_fusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_score_fusion([{"d1": 1.0}], weights=[0.5, 0.5])


def test_weighted_score_fusion_handles_flat_scores_without_divide_by_zero():
    run = {"d1": 5.0, "d2": 5.0}
    fused = weighted_score_fusion([run], weights=[1.0])
    assert set(fused) == {"d1", "d2"}
