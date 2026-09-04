"""Phase 0: build the evaluation set.

CRITICAL DESIGN NOTE
--------------------
Queries are LLM-GENERATED natural research questions, not phrases extracted from
the documents. Extraction leaks surface tokens into the query and hands the
comparison to BM25 by construction, which invalidates the entire study.

In this project the "LLM" doing the generating is Claude, composing questions
directly (no scripted API call configured) -- see scripts/build_evalset.py for
the workflow: it selects which docs get a question, then an already-authored
{doc_id: query_text} mapping is merged in and validated for full coverage.

Each record: {"qid", "query", "gold_doc_id", "split"}
`split` is "finetune" or "holdout" and the two MUST be disjoint (Phase 2b).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def select_docs_for_queries(df: pd.DataFrame, n_queries: int, seed: int) -> pd.DataFrame:
    """Deterministically choose which `n_queries` docs get a question authored
    for them. Sorted by doc_id afterward so the selection has a stable,
    diffable order regardless of the corpus's on-disk row order."""
    if n_queries > len(df):
        raise ValueError(f"n_queries={n_queries} exceeds corpus size {len(df)}")
    return df.sample(n=n_queries, random_state=seed).sort_values("doc_id").reset_index(drop=True)


def generate_queries(
    df: pd.DataFrame, n_queries: int, seed: int, queries: Mapping[str, str]
) -> list[dict]:
    """Combine the deterministic doc selection with an already-authored
    {doc_id: query_text} mapping, and validate full coverage.

    `queries` is supplied by whoever/whatever generated the questions --
    here, Claude composing them directly in batches. Keeping selection and
    authoring separate means the selection logic is testable without needing
    400 real questions on hand.
    """
    selected = select_docs_for_queries(df, n_queries, seed)
    missing = set(selected["doc_id"]) - set(queries.keys())
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} selected doc_ids have no authored query yet, e.g. {preview}")

    records = []
    for _, row in selected.iterrows():
        query_text = queries[row["doc_id"]].strip()
        if not query_text:
            raise ValueError(f"Empty query for doc_id={row['doc_id']}")
        records.append(
            {
                "qid": f"q_{row['doc_id']}",
                "query": query_text,
                "gold_doc_id": row["doc_id"],
                "split": None,  # assigned by split() in Task 0.4
            }
        )
    return records


def spot_check(records: Sequence[dict], n: int, seed: int | None = None) -> list[dict]:
    """Sample n records for manual review. Reject rate is recorded by whoever
    reviews the printed sample -- this function's job is just to draw a fair,
    reproducible sample, not to judge quality itself."""
    rng = random.Random(seed)
    return rng.sample(list(records), min(n, len(records)))


def split(records: Sequence[dict], holdout_frac: float, seed: int) -> list[dict]:
    """Disjoint finetune/holdout split (Task 0.4). Returns new records with
    `split` filled in; does not mutate the input."""
    records = [dict(r) for r in records]
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_holdout = round(len(records) * holdout_frac)
    holdout_idx = set(idx[:n_holdout])
    for i, r in enumerate(records):
        r["split"] = "holdout" if i in holdout_idx else "finetune"
    return records


def save_jsonl(records: Sequence[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
