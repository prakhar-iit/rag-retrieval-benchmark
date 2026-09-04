"""Phase 2: vector index (Qdrant) and the lexical index.

Qdrant rather than FAISS alone: FAISS is an index, Qdrant is a vector database,
and the claim being evidenced is 'vector DB'.

Deviation from configs/default.yaml's `index.url: http://localhost:6333`: this
dev environment has no docker, so there's no way to run a Qdrant server. Qdrant's
Python client has a fully-supported embedded/local mode (`QdrantClient(path=...)`)
that persists the same collection format to a local directory with no server
process at all -- same client API, same on-disk collection, no docker dependency.
That's what build_qdrant/search use here; index.path in the config carries the
directory, index.url is kept for documentation of the "real" deployment target.
"""
from __future__ import annotations

import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Mapping, Sequence

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


_DISTANCE_MAP = {"cosine": "COSINE", "dot": "DOT", "euclid": "EUCLID"}


@dataclass
class QdrantIndex:
    """A built Qdrant collection: the client (bound to its on-disk directory
    in local mode) plus the collection name, so `search` has one argument
    to carry around instead of two -- mirrors how `build_bm25` returns a
    single opaque index object for `search_bm25`."""

    client: "object"
    collection: str
    path: str


def build_qdrant(
    name: str,
    vectors,
    payloads: Sequence[Mapping] | None,
    path: str,
    distance: str = "cosine",
) -> QdrantIndex:
    """Create a local (embedded, docker-free) Qdrant collection and upsert
    `vectors` with optional `payloads` (one dict per vector, e.g. {"doc_id": ...}).

    `path` is a directory -- QdrantClient(path=...) local mode owns it as its
    on-disk collection storage (RocksDB-backed), not a single file. Rebuilding
    an existing collection at the same path deletes and recreates it, so this
    function is safe to call repeatedly (e.g. once per config in Task 2.7's
    scaling-curve runs) without accumulating stale collections.

    Callers time this call and measure `path`'s on-disk size afterward for the
    systems numbers in Task 2.7 (see qdrant_index_size_bytes).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    vectors = np.asarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    if payloads is not None and len(payloads) != n:
        raise ValueError(f"got {len(payloads)} payloads for {n} vectors")

    client = QdrantClient(path=path)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance[_DISTANCE_MAP[distance]]),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload=dict(payloads[i]) if payloads is not None else {},
        )
        for i in range(n)
    ]
    client.upsert(collection_name=name, points=points)
    return QdrantIndex(client=client, collection=name, path=path)


def search(index: QdrantIndex, query_vector, k: int) -> list[int]:
    """Return the top-k point ids for `query_vector` against `index`, best first.

    Ids are whatever integer ids `build_qdrant` assigned (0..n-1, positional --
    same id-scheme-agnostic convention as search_bm25's corpus positions).
    """
    hits = index.client.query_points(
        collection_name=index.collection,
        query=np.asarray(query_vector, dtype=np.float32).tolist(),
        limit=k,
    ).points
    return [hit.id for hit in hits]


def qdrant_index_size_bytes(path: str) -> int:
    """On-disk size of a local Qdrant collection directory (Task 2.7) -- the
    real counterpart to bm25_index_size_bytes's pickle-based approximation,
    since Qdrant local mode actually persists to disk rather than living
    only in memory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return total


def delete_qdrant_index(path: str) -> None:
    """Remove a local Qdrant collection directory entirely (cleanup between
    scaling-curve runs at different corpus sizes -- Task 2.7d)."""
    shutil.rmtree(path, ignore_errors=True)
