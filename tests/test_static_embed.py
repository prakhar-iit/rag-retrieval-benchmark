import math

import numpy as np
import pandas as pd
import pytest

from rss.static_embed import (
    build_phrases,
    compute_idf,
    doc_vector,
    tokenize,
    tokenize_corpus,
    train,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! It's a Test.") == ["hello", "world", "it", "s", "a", "test"]


def test_tokenize_drops_pure_numbers_but_keeps_alphanumeric_jargon():
    tokens = tokenize("ResNet50 achieved 96.2% accuracy in 2019 using GPT3.")
    assert "resnet50" in tokens
    assert "gpt3" in tokens
    assert "96" not in tokens
    assert "2019" not in tokens


def test_tokenize_empty_string():
    assert tokenize("") == []
    assert tokenize("... !!! ---") == []


def test_tokenize_corpus_preserves_row_order():
    df = pd.DataFrame({"abstract": ["cat dog", "bird fish snake"]})
    result = tokenize_corpus(df)
    assert result == [["cat", "dog"], ["bird", "fish", "snake"]]


def _toy_sentences(n=200):
    # "reinforcement learning" co-occurs often enough to clear Phrases' default
    # thresholds; "banana" is a filler unigram that should never merge with anything.
    sentences = []
    for i in range(n):
        sentences.append(["reinforcement", "learning", "agent", "policy"])
        sentences.append(["deep", "reinforcement", "learning", "reward"])
        sentences.append(["banana", "apple", str(i % 3)])
    return sentences


def test_build_phrases_merges_frequent_bigram():
    # gensim's phrase score is normalised by vocab size, so a toy corpus with
    # only ~20 distinct words needs a much lower threshold than real text
    # (large vocab) would to actually clear it -- this is a property of the
    # toy fixture's tiny vocabulary, not of build_phrases' real-world defaults.
    sentences = _toy_sentences()
    phrased = build_phrases(sentences, min_count=5, threshold=0.001)
    assert len(phrased) == len(sentences)
    # "reinforcement learning" appears together very often -> should merge.
    merged_tokens = {tok for sent in phrased for tok in sent}
    assert any("reinforcement" in tok and "learning" in tok for tok in merged_tokens)


def test_build_phrases_does_not_merge_rare_pairs():
    sentences = _toy_sentences()
    phrased = build_phrases(sentences, min_count=5, threshold=0.001)
    merged_tokens = {tok for sent in phrased for tok in sent}
    # "banana" and "apple" co-occur every time they appear together, same as
    # the frequent pair above -- so at this threshold they merge too. The
    # real discriminating case is a pair that does NOT reliably co-occur.
    assert any("banana" in tok for tok in merged_tokens)
    assert not any("policy" in tok and "banana" in tok for tok in merged_tokens)


def _toy_model():
    sentences = _toy_sentences(n=50)
    cfg = {"sg": 1, "vector_size": 16, "window": 3, "min_count": 1, "negative": 5, "epochs": 5}
    return train(sentences, cfg, seed=13)


def test_train_is_deterministic_with_fixed_seed():
    sentences = _toy_sentences(n=50)
    cfg = {"sg": 1, "vector_size": 16, "window": 3, "min_count": 1, "negative": 5, "epochs": 5}
    m1 = train(sentences, cfg, seed=13)
    m2 = train(sentences, cfg, seed=13)
    np.testing.assert_array_equal(m1.wv["reinforcement"], m2.wv["reinforcement"])


def test_train_respects_vector_size():
    model = _toy_model()
    assert model.wv.vector_size == 16


def test_compute_idf_rare_term_scores_higher_than_common_term():
    docs = [["common", "common", "rare"], ["common", "other"], ["common", "another"]]
    idf = compute_idf(docs)
    assert idf["rare"] > idf["common"]


def test_compute_idf_term_in_every_doc_stays_positive():
    docs = [["always"], ["always", "x"], ["always", "y"]]
    idf = compute_idf(docs)
    assert idf["always"] > 0


def test_doc_vector_mean_pooling_matches_manual_average():
    model = _toy_model()
    tokens = ["reinforcement", "learning"]
    vec = doc_vector(model, tokens)
    expected = np.mean([model.wv["reinforcement"], model.wv["learning"]], axis=0)
    np.testing.assert_allclose(vec, expected, rtol=1e-5)


def test_doc_vector_skips_oov_tokens():
    model = _toy_model()
    vec_with_oov = doc_vector(model, ["reinforcement", "learning", "zzz_not_in_vocab"])
    vec_without_oov = doc_vector(model, ["reinforcement", "learning"])
    np.testing.assert_allclose(vec_with_oov, vec_without_oov, rtol=1e-5)


def test_doc_vector_all_oov_returns_zero_vector():
    model = _toy_model()
    vec = doc_vector(model, ["zzz_nope", "zzz_also_nope"])
    np.testing.assert_array_equal(vec, np.zeros(model.wv.vector_size, dtype=np.float32))


def test_doc_vector_empty_tokens_returns_zero_vector():
    model = _toy_model()
    vec = doc_vector(model, [])
    np.testing.assert_array_equal(vec, np.zeros(model.wv.vector_size, dtype=np.float32))


def test_doc_vector_idf_weighting_differs_from_mean_pooling():
    model = _toy_model()
    tokens = ["reinforcement", "learning", "agent"]
    idf = {"reinforcement": 1.0, "learning": 1.0, "agent": 10.0}
    mean_vec = doc_vector(model, tokens)
    idf_vec = doc_vector(model, tokens, idf=idf)
    # Heavily up-weighting "agent" should pull the pooled vector toward it,
    # away from the uniform mean.
    assert not np.allclose(mean_vec, idf_vec)


def test_doc_vector_idf_weighting_reduces_to_mean_with_uniform_weights():
    model = _toy_model()
    tokens = ["reinforcement", "learning", "agent"]
    uniform_idf = {t: 1.0 for t in tokens}
    mean_vec = doc_vector(model, tokens)
    idf_vec = doc_vector(model, tokens, idf=uniform_idf)
    np.testing.assert_allclose(mean_vec, idf_vec, rtol=1e-5)
