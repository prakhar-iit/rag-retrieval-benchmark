"""Phase 2: vector index (Qdrant) and the lexical index.

Qdrant rather than FAISS alone: FAISS is an index, Qdrant is a vector database,
and the claim being evidenced is 'vector DB'.
"""
from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np


def build_bm25(corpus_tokens: Sequence[Sequence[str]]):
    """rank_bm25 BM25Okapi index over pre-tokenized documents.

    Uses the SAME tokenizer as the Word2Vec baselines (rss.static_embed.tokenize)
    so preprocessing is held constant across methods -- differences in the
    eval numbers reflect the retrieval method, not the tokenizer.
    """
    from rank_bm25 import BM25Okapi

    return BM25Okapi(list(corpus_tokens))


def search_bm25(index, query_tokens: Sequence[str], k: int) -> list[int]:
    """Return the top-k corpus positions for `query_tokens`, best first.

    Positions index into whatever `corpus_tokens` sequence `index` was built
    from -- callers map position -> doc_id themselves (index.py stays
    id-scheme-agnostic on purpose, same as build_qdrant/search below).
    """
    scores = index.get_scores(list(query_tokens))
    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return list(top_idx[np.argsort(-scores[top_idx])])


def bm25_index_size_bytes(index) -> int:
    """Approximate in-memory index size via pickle -- BM25Okapi has no native
    on-disk format, so this is the fairest apples-to-apples size comparison
    against Qdrant's reported on-disk collection size (Task 2.7)."""
    return len(pickle.dumps(index))


def build_qdrant(name: str, vectors, payloads, url: str):
    """Create a collection and upsert. Record build time and on-disk size (Task 2.7)."""
    raise NotImplementedError("Task 2.4")


def search(index, query_vector, k: int):
    raise NotImplementedError("Task 2.4")
