"""Phase 2c: cross-encoder reranking.

A cross-encoder scores query and document JOINTLY -- there is no independent
document vector, so it cannot be pre-indexed. That is why it runs as a second
stage over top-N candidates, and why it is both the most accurate and the most
expensive stage in the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple


def load_reranker(model_name: str, local_dir: str = "models/hf"):
    """Load a local cross-encoder snapshot (e.g. BAAI/bge-reranker-base) for
    repeated use across many `rerank()` calls.

    Loading takes real time (weight deserialization), so callers should
    load once per eval run and pass the result to `rerank` per query --
    NOT reload per query, which would dominate the latency measurement
    Task 2c cares about (added p95 vs the first-stage retriever alone).

    Like rss.dense_embed.encode, this loads from a LOCAL snapshot directory
    rather than downloading automatically, since huggingface.co is
    unreachable from this dev environment.
    """
    local_path = Path(local_dir) / model_name.split("/")[-1]
    if not local_path.exists():
        raise FileNotFoundError(
            f"no local snapshot at {local_path} -- fetch it first with "
            f"scripts/fetch_hf_models.py on a network that can reach huggingface.co"
        )

    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(local_path))


def rerank(model, query: str, candidates: Sequence[Tuple[str, str]], top_n: int = 50) -> list[str]:
    """Rerank `candidates` -- a first-stage retriever's (doc_id, doc_text)
    results, best-first -- against `query` with a loaded cross-encoder
    (from `load_reranker`). Returns doc_ids in the cross-encoder's order,
    best first.

    `top_n` caps how many of `candidates` actually get the expensive
    cross-encoder pass: only the FIRST `top_n` (the first-stage retriever's
    own best guesses) are scored, and anything beyond that is dropped
    entirely rather than appended unscored at the end -- scoring the full
    corpus this way for every query would be far too slow, which is the
    whole reason reranking is a second stage over a small candidate set
    instead of a replacement for the first-stage retriever.

    `model` is any object with a `.predict(list[(query, doc_text)])` method
    returning per-pair scores, higher-is-better (CrossEncoder's own
    interface) -- duck-typed so tests can pass a fake instead of a real
    cross-encoder.
    """
    candidates = list(candidates)[:top_n]
    if not candidates:
        return []

    pairs = [(query, text) for _doc_id, text in candidates]
    scores = model.predict(pairs)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i][0] for i in order]
