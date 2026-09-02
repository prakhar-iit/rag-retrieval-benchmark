"""Phase 2: vector index (Qdrant) and the lexical index.

Qdrant rather than FAISS alone: FAISS is an index, Qdrant is a vector database,
and the claim being evidenced is 'vector DB'.
"""
from __future__ import annotations


def build_bm25(corpus_tokens):
    raise NotImplementedError("Task 2.1")


def build_qdrant(name: str, vectors, payloads, url: str):
    """Create a collection and upsert. Record build time and on-disk size (Task 2.7)."""
    raise NotImplementedError("Task 2.4")


def search(index, query_vector, k: int):
    raise NotImplementedError("Task 2.4")
