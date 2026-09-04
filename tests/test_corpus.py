import pandas as pd
import pytest

from rss.corpus import (
    _make_doc_id,
    _sample_and_id,
    freeze,
    load_frozen,
    load_local_and_sample,
)


def _toy_df(n=50):
    return pd.DataFrame(
        {
            "abstract": [
                f"This is a synthetic abstract number {i} about topic {i % 7}, "
                f"written with enough words to clear the minimum length filter."
                for i in range(n)
            ],
            "title": [f"Paper {i}" for i in range(n)],
        }
    )


def test_doc_id_is_stable_and_content_derived():
    assert _make_doc_id("hello world") == _make_doc_id("hello world")
    assert _make_doc_id("hello world") != _make_doc_id("goodbye world")
    assert _make_doc_id("hello world").startswith("arxiv_")


def test_sample_and_id_deterministic_for_fixed_seed():
    df = _toy_df(50)
    a = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13)
    b = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13)
    assert list(a["doc_id"]) == list(b["doc_id"])
    assert len(a) == 10
    assert a["doc_id"].is_unique


def test_sample_and_id_different_seed_gives_different_sample():
    df = _toy_df(50)
    a = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13)
    c = _sample_and_id(df, text_field="abstract", sample_size=10, seed=99)
    assert list(a["doc_id"]) != list(c["doc_id"])


def test_sample_size_larger_than_corpus_keeps_everything():
    df = _toy_df(5)
    out = _sample_and_id(df, text_field="abstract", sample_size=1000, seed=13)
    assert len(out) == 5


def test_drops_empty_and_duplicate_abstracts():
    df = pd.DataFrame({"abstract": ["real one", "", "  ", "real one", "another real one"]})
    out = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13, min_words=0)
    assert len(out) == 2
    assert set(out["abstract"]) == {"real one", "another real one"}


def test_drops_stray_unnamed_index_columns():
    df = pd.DataFrame({
        "Unnamed: 0.1": [0, 1],
        "Unnamed: 0": [0.0, 1.0],
        "abstract": ["first real abstract here", "second real abstract here"],
        "title": ["A", "B"],
    })
    out = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13)
    assert not any(c.startswith("Unnamed:") for c in out.columns)
    assert set(out.columns) == {"doc_id", "abstract", "title"}


def test_drops_degenerate_and_too_short_abstracts():
    df = pd.DataFrame({
        "abstract": [
            "This is a perfectly normal abstract with plenty of real content in it about topic X.",
            "This preprint has been withdrawn by the author for revision.",
            "Some content of the article needs to be kept secret.",
            "Too short.",
        ]
    })
    out = _sample_and_id(df, text_field="abstract", sample_size=10, seed=13, min_words=15)
    assert len(out) == 1
    assert "normal abstract" in out.iloc[0]["abstract"]


def test_missing_text_field_raises_with_helpful_message():
    df = pd.DataFrame({"summary": ["x"]})
    with pytest.raises(ValueError, match="not found"):
        _sample_and_id(df, text_field="abstract", sample_size=10, seed=13)


def test_load_local_and_sample_csv(tmp_path):
    df = _toy_df(20)
    csv_path = tmp_path / "corpus.csv"
    df.to_csv(csv_path, index=False)

    out = load_local_and_sample(str(csv_path), sample_size=5, seed=13, text_field="abstract")
    assert len(out) == 5
    assert "doc_id" in out.columns


def test_load_local_and_sample_unrecognized_extension(tmp_path):
    bad_path = tmp_path / "corpus.txt"
    bad_path.write_text("not a real corpus")
    with pytest.raises(ValueError, match="Unrecognized file type"):
        load_local_and_sample(str(bad_path), sample_size=5, seed=13)


def test_freeze_and_load_frozen_roundtrip(tmp_path):
    df = _sample_and_id(_toy_df(30), text_field="abstract", sample_size=10, seed=13)
    out_path = tmp_path / "nested" / "corpus_sample.parquet"

    freeze(df, str(out_path))
    assert out_path.exists()

    reloaded = load_frozen(str(out_path))
    pd.testing.assert_frame_equal(reloaded, df)
