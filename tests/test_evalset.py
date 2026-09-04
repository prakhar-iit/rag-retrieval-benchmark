import pandas as pd
import pytest

from rss.evalset import generate_queries, load_jsonl, save_jsonl, select_docs_for_queries, spot_check, split


def _toy_corpus(n=100):
    return pd.DataFrame({"doc_id": [f"doc_{i:03d}" for i in range(n)], "abstract": [f"abstract {i}" for i in range(n)]})


def test_select_docs_deterministic_for_fixed_seed():
    df = _toy_corpus()
    a = select_docs_for_queries(df, n_queries=10, seed=13)
    b = select_docs_for_queries(df, n_queries=10, seed=13)
    assert list(a["doc_id"]) == list(b["doc_id"])
    assert len(a) == 10


def test_select_docs_is_sorted_by_doc_id():
    df = _toy_corpus()
    out = select_docs_for_queries(df, n_queries=10, seed=13)
    assert list(out["doc_id"]) == sorted(out["doc_id"])


def test_select_docs_raises_if_n_queries_exceeds_corpus():
    df = _toy_corpus(5)
    with pytest.raises(ValueError, match="exceeds corpus size"):
        select_docs_for_queries(df, n_queries=10, seed=13)


def test_generate_queries_builds_records_with_full_coverage():
    df = _toy_corpus()
    selected = select_docs_for_queries(df, n_queries=5, seed=13)
    queries = {doc_id: f"What does {doc_id} show?" for doc_id in selected["doc_id"]}

    records = generate_queries(df, n_queries=5, seed=13, queries=queries)

    assert len(records) == 5
    assert {r["gold_doc_id"] for r in records} == set(selected["doc_id"])
    assert all(r["query"].startswith("What does") for r in records)
    assert all(r["split"] is None for r in records)
    assert all(r["qid"] == f"q_{r['gold_doc_id']}" for r in records)


def test_generate_queries_raises_on_missing_coverage():
    df = _toy_corpus()
    selected = select_docs_for_queries(df, n_queries=5, seed=13)
    incomplete = {doc_id: "some question" for doc_id in list(selected["doc_id"])[:3]}
    with pytest.raises(ValueError, match="have no authored query yet"):
        generate_queries(df, n_queries=5, seed=13, queries=incomplete)


def test_generate_queries_raises_on_empty_query_text():
    df = _toy_corpus()
    selected = select_docs_for_queries(df, n_queries=5, seed=13)
    queries = {doc_id: "   " for doc_id in selected["doc_id"]}
    with pytest.raises(ValueError, match="Empty query"):
        generate_queries(df, n_queries=5, seed=13, queries=queries)


def test_spot_check_samples_without_replacement_and_is_reproducible():
    records = [{"qid": f"q{i}"} for i in range(50)]
    a = spot_check(records, n=20, seed=7)
    b = spot_check(records, n=20, seed=7)
    assert len(a) == 20
    assert [r["qid"] for r in a] == [r["qid"] for r in b]
    assert len({r["qid"] for r in a}) == 20  # no duplicates


def test_spot_check_caps_at_available_records():
    records = [{"qid": f"q{i}"} for i in range(5)]
    out = spot_check(records, n=20, seed=7)
    assert len(out) == 5


def test_split_is_disjoint_and_covers_everything():
    records = [{"qid": f"q{i}"} for i in range(200)]
    out = split(records, holdout_frac=0.5, seed=13)

    finetune = {r["qid"] for r in out if r["split"] == "finetune"}
    holdout = {r["qid"] for r in out if r["split"] == "holdout"}

    assert finetune.isdisjoint(holdout)
    assert finetune | holdout == {r["qid"] for r in records}
    assert len(holdout) == 100  # exactly holdout_frac of 200


def test_split_does_not_mutate_input():
    records = [{"qid": "q0", "split": None}]
    split(records, holdout_frac=0.5, seed=13)
    assert records[0]["split"] is None


def test_split_is_reproducible_for_fixed_seed():
    records = [{"qid": f"q{i}"} for i in range(50)]
    a = split(records, holdout_frac=0.3, seed=13)
    b = split(records, holdout_frac=0.3, seed=13)
    assert [r["split"] for r in a] == [r["split"] for r in b]


def test_save_and_load_jsonl_roundtrip(tmp_path):
    records = [
        {"qid": "q1", "query": "does it work?", "gold_doc_id": "doc_1", "split": "finetune"},
        {"qid": "q2", "query": "does it scale?", "gold_doc_id": "doc_2", "split": "holdout"},
    ]
    path = tmp_path / "nested" / "evalset.jsonl"
    save_jsonl(records, str(path))
    assert path.exists()
    reloaded = load_jsonl(str(path))
    assert reloaded == records
