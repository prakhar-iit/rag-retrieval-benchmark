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
