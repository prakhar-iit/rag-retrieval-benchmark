import pytest

from rss.rerank import load_reranker, rerank


class _FakeCrossEncoder:
    """Duck-typed stand-in for sentence_transformers.CrossEncoder: scores
    each (query, doc_text) pair by how many words they share, so tests can
    assert on a specific, predictable ranking."""

    def __init__(self):
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.predict_calls.append(list(pairs))
        scores = []
        for query, doc_text in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc_text.lower().split())
            scores.append(float(len(q_words & d_words)))
        return scores


_CANDIDATES = [
    ("d1", "cats are small domesticated mammals"),
    ("d2", "the eiffel tower is a landmark in paris"),
    ("d3", "dogs are loyal domesticated companions"),
]


def test_rerank_orders_by_score_best_first():
    model = _FakeCrossEncoder()
    result = rerank(model, "domesticated mammals cats", _CANDIDATES)
    assert result[0] == "d1"  # shares "domesticated", "mammals", "cats"


def test_rerank_returns_all_doc_ids_reordered():
    model = _FakeCrossEncoder()
    result = rerank(model, "domesticated animals", _CANDIDATES)
    assert set(result) == {"d1", "d2", "d3"}
    assert len(result) == 3


def test_rerank_respects_top_n_cap():
    model = _FakeCrossEncoder()
    result = rerank(model, "paris landmark", _CANDIDATES, top_n=2)
    assert len(result) == 2
    # only the first 2 candidates (d1, d2) should ever be eligible -- d3 is
    # dropped outright, not scored-and-excluded (checked directly below).
    assert set(result) <= {"d1", "d2"}


def test_rerank_top_n_only_scores_the_capped_candidates():
    model = _FakeCrossEncoder()
    rerank(model, "query", _CANDIDATES, top_n=2)
    assert len(model.predict_calls[0]) == 2


def test_rerank_empty_candidates_returns_empty():
    model = _FakeCrossEncoder()
    assert rerank(model, "anything", []) == []


def test_rerank_single_candidate():
    model = _FakeCrossEncoder()
    result = rerank(model, "cats", [("d1", "cats are mammals")])
    assert result == ["d1"]


def test_load_reranker_raises_when_local_snapshot_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reranker("BAAI/does-not-exist", local_dir=str(tmp_path / "no_such_dir"))
