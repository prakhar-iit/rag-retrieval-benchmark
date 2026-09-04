"""Phase 0: retrieval metrics.

Recall@k matters more than precision for RAG: a generator can ignore an
irrelevant chunk, but it cannot invent a missing one.

Every query in this project has exactly one gold document (see evalset.py --
one LLM-generated question per sampled abstract), so IDCG is always the
single-relevant-document case: 1 / log2(1 + 1) = 1.0. nDCG@k below is
written generally (dividing by that IDCG) so it stays correct if a later
phase ever moves to multi-relevant qrels.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def _rank_of(ranked_ids: Sequence[str], gold_id: str) -> int | None:
    """1-indexed rank of gold_id in ranked_ids, or None if absent."""
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def ndcg_at_k(ranked_ids: Sequence[str], gold_id: str, k: int = 10) -> float:
    rank = _rank_of(ranked_ids, gold_id)
    if rank is None or rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(1 + 1)  # single relevant doc, best case at rank 1
    return dcg / idcg


def recall_at_k(ranked_ids: Sequence[str], gold_id: str, k: int = 10) -> float:
    rank = _rank_of(ranked_ids, gold_id)
    return 1.0 if rank is not None and rank <= k else 0.0


def mrr(ranked_ids: Sequence[str], gold_id: str) -> float:
    rank = _rank_of(ranked_ids, gold_id)
    return 0.0 if rank is None else 1.0 / rank


def evaluate(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, str],
    k_values: Sequence[int],
) -> dict:
    """Aggregate metrics over a full run.

    run:   {qid: [doc_id, doc_id, ...]}  ranked, best first
    qrels: {qid: gold_doc_id}
    k_values: e.g. [1, 5, 10, 20]

    Returns per-query scores AND the mean -- averages hide the failure modes
    that Phase 3b (automatic failure taxonomy) exists to find, so callers
    should keep `per_query` around to slice by topic/cluster later, not just
    read `mean`.
    """
    per_query: dict[str, dict[str, float]] = {}

    for qid, gold_id in qrels.items():
        ranked_ids = run.get(qid, [])
        scores: dict[str, float] = {"mrr": mrr(ranked_ids, gold_id)}
        for k in k_values:
            scores[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, gold_id, k)
            scores[f"recall@{k}"] = recall_at_k(ranked_ids, gold_id, k)
        per_query[qid] = scores

    metric_names = next(iter(per_query.values())).keys() if per_query else []
    mean = {
        name: (sum(s[name] for s in per_query.values()) / len(per_query))
        for name in metric_names
    }

    return {"per_query": per_query, "mean": mean, "n_queries": len(per_query)}
