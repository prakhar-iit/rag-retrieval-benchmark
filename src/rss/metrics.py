"""Phase 0: retrieval metrics.

Recall@k matters more than precision for RAG: a generator can ignore an
irrelevant chunk, but it cannot invent a missing one.
"""
from __future__ import annotations
from typing import Sequence


def ndcg_at_k(ranked_ids: Sequence[str], gold_id: str, k: int = 10) -> float:
    raise NotImplementedError("Task 0.5")


def recall_at_k(ranked_ids: Sequence[str], gold_id: str, k: int = 10) -> float:
    raise NotImplementedError("Task 0.5")


def mrr(ranked_ids: Sequence[str], gold_id: str) -> float:
    raise NotImplementedError("Task 0.5")


def evaluate(run, qrels, k_values):
    """Aggregate metrics over a full run. Also report per-slice, not just the mean --
    averages hide the failure modes that Phase 3b exists to find."""
    raise NotImplementedError("Task 0.5")
