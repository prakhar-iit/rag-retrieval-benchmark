"""Phase 2c: cross-encoder reranking.

A cross-encoder scores query and document JOINTLY -- there is no independent
document vector, so it cannot be pre-indexed. That is why it runs as a second
stage over top-N candidates, and why it is both the most accurate and the most
expensive stage in the pipeline.
"""
from __future__ import annotations


def rerank(model_name: str, query: str, candidates, top_n: int = 50):
    """Rerank top_n candidates. Record added p95 latency alongside the nDCG lift."""
    raise NotImplementedError("Task 2c.1")
