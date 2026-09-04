from rss.index import bm25_index_size_bytes, build_bm25, search_bm25

_CORPUS = [
    ["cats", "are", "small", "domesticated", "carnivorous", "mammals"],
    ["dogs", "are", "loyal", "domesticated", "mammals", "and", "good", "companions"],
    ["stochastic", "gradient", "descent", "optimizes", "neural", "network", "weights"],
    ["reinforcement", "learning", "agents", "optimize", "a", "reward", "signal"],
    ["the", "eiffel", "tower", "is", "a", "landmark", "in", "paris", "france"],
]


def test_build_bm25_returns_index_scorable_for_all_docs():
    index = build_bm25(_CORPUS)
    scores = index.get_scores(["mammals"])
    assert len(scores) == len(_CORPUS)


def test_search_bm25_ranks_exact_term_match_first():
    index = build_bm25(_CORPUS)
    top = search_bm25(index, ["gradient", "descent"], k=3)
    assert top[0] == 2  # the SGD doc


def test_search_bm25_returns_k_results_best_first():
    index = build_bm25(_CORPUS)
    top = search_bm25(index, ["domesticated", "mammals"], k=2)
    assert len(top) == 2
    # both the cats and dogs docs mention "domesticated mammals" -- one of
    # them should rank first, and it should not be the unrelated docs.
    assert set(top) <= {0, 1}


def test_search_bm25_k_larger_than_corpus_returns_all():
    index = build_bm25(_CORPUS)
    top = search_bm25(index, ["paris"], k=100)
    assert len(top) == len(_CORPUS)
    assert top[0] == 4  # the Eiffel Tower doc


def test_search_bm25_no_matching_terms_still_returns_k():
    index = build_bm25(_CORPUS)
    top = search_bm25(index, ["zzz_nonexistent_term"], k=3)
    assert len(top) == 3


def test_bm25_index_size_bytes_scales_with_corpus():
    small = build_bm25(_CORPUS[:2])
    large = build_bm25(_CORPUS)
    assert bm25_index_size_bytes(large) > bm25_index_size_bytes(small)
    assert bm25_index_size_bytes(small) > 0


import numpy as np

from rss.index import (
    build_qdrant,
    delete_qdrant_index,
    qdrant_index_size_bytes,
    search,
)

# 5 near-orthogonal toy vectors in 4D so cosine similarity cleanly separates
# "this is doc i" from "this is not doc i" -- avoids flaky nearest-neighbour
# assertions on random vectors.
_VECTORS = np.array(
    [
        [1.0, 0.01, 0.0, 0.0],
        [0.0, 1.0, 0.01, 0.0],
        [0.0, 0.0, 1.0, 0.01],
        [0.01, 0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0, 0.0],  # deliberately close to doc 0 and doc 1 both
    ],
    dtype=np.float32,
)
_PAYLOADS = [{"doc_id": f"doc{i}"} for i in range(5)]


def test_build_qdrant_and_search_finds_exact_match(tmp_path):
    index = build_qdrant("toy", _VECTORS, _PAYLOADS, path=str(tmp_path / "qdrant"))
    top = search(index, _VECTORS[2], k=3)
    assert top[0] == 2


def test_search_returns_k_results(tmp_path):
    index = build_qdrant("toy", _VECTORS, _PAYLOADS, path=str(tmp_path / "qdrant"))
    top = search(index, _VECTORS[0], k=3)
    assert len(top) == 3


def test_build_qdrant_rejects_payload_count_mismatch(tmp_path):
    try:
        build_qdrant("toy", _VECTORS, _PAYLOADS[:2], path=str(tmp_path / "qdrant"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_qdrant_without_payloads(tmp_path):
    index = build_qdrant("toy", _VECTORS, None, path=str(tmp_path / "qdrant"))
    top = search(index, _VECTORS[4], k=1)
    assert top[0] == 4


def test_qdrant_index_size_bytes_positive_after_build(tmp_path):
    path = str(tmp_path / "qdrant")
    build_qdrant("toy", _VECTORS, _PAYLOADS, path=path)
    assert qdrant_index_size_bytes(path) > 0


def test_build_qdrant_rebuild_at_same_path_overwrites(tmp_path):
    path = str(tmp_path / "qdrant")
    build_qdrant("toy", _VECTORS, _PAYLOADS, path=path)
    smaller = build_qdrant("toy", _VECTORS[:2], _PAYLOADS[:2], path=path)
    top = search(smaller, _VECTORS[0], k=5)
    # only 2 points exist post-rebuild, even though the first build had 5
    assert len(top) == 2


def test_delete_qdrant_index_removes_directory(tmp_path):
    import os

    path = str(tmp_path / "qdrant")
    build_qdrant("toy", _VECTORS, _PAYLOADS, path=path)
    assert os.path.exists(path)
    delete_qdrant_index(path)
    assert not os.path.exists(path)
