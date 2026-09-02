"""Phase 0: build the evaluation set.

CRITICAL DESIGN NOTE
--------------------
Queries are LLM-GENERATED natural research questions, not phrases extracted from
the documents. Extraction leaks surface tokens into the query and hands the
comparison to BM25 by construction, which invalidates the entire study.

Each record: {"qid", "query", "gold_doc_id", "split"}
`split` is "finetune" or "holdout" and the two MUST be disjoint (Phase 2b).
"""
from __future__ import annotations


def generate_queries(df, n_queries: int, seed: int):
    """Generate one research question per sampled abstract."""
    raise NotImplementedError("Task 0.3")


def spot_check(records, n: int):
    """Print n records for manual review; return the reject rate."""
    raise NotImplementedError("Task 0.3")


def split(records, holdout_frac: float, seed: int):
    """Disjoint finetune/holdout split."""
    raise NotImplementedError("Task 0.4")
