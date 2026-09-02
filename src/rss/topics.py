"""Phase 3: topic modelling as diagnostics, not as an end in itself.

3a  Does embedding choice change what you discover? (NPMI, c_v, outlier rate)
3b  Automatic failure taxonomy -- cluster FAILING queries, LLM-label the clusters.
    Every relevance team does this by hand in a spreadsheet and calls it error analysis.
3c  Corpus coverage gaps -- query-topic density vs document-topic density.
    High query volume + low document density = a gap no retrieval tuning can fix.
"""
from __future__ import annotations


def fit_bertopic(embeddings, documents):
    raise NotImplementedError("Task 3a")


def coherence(topic_model, documents):
    """NPMI and c_v."""
    raise NotImplementedError("Task 3a")


def failure_taxonomy(failed_queries, embeddings):
    """Cluster failures, then LLM-label each cluster."""
    raise NotImplementedError("Task 3b")


def coverage_gaps(doc_topics, query_topics):
    raise NotImplementedError("Task 3c")
