"""Phase 2: hybrid retrieval.

Lexical and dense methods fail on DIFFERENT queries -- that is the whole
argument for fusing them. RRF is the usual default because it needs no score
calibration across methods.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def reciprocal_rank_fusion(runs: Sequence[Sequence], k: int = 60) -> list:
    """RRF over N ranked lists of doc ids: score(d) = sum(1 / (k + rank(d))),
    rank is 1-indexed position within each run. Returns doc ids sorted by
    fused score, best first.

    A doc absent from a run simply contributes 0 from that run rather than
    being penalised -- this asymmetry (missing != worst-possible-rank) is
    why RRF needs no cross-method score calibration, unlike
    `weighted_score_fusion` below.
    """
    scores: dict = defaultdict(float)
    for run in runs:
        for rank, doc_id in enumerate(run, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def weighted_score_fusion(runs_with_scores: Sequence[Mapping], weights: Sequence[float]) -> list:
    """Normalised weighted sum of per-method similarity/relevance scores.

    `runs_with_scores` is one {doc_id: score} dict per method, same length
    and order as `weights`. Each method's scores are min-max normalised to
    [0, 1] INDEPENDENTLY before combining -- otherwise a method whose raw
    scores happen to run larger (BM25's unbounded scores vs cosine's
    [-1, 1]) would dominate the sum regardless of its weight. That
    normalisation is exactly the score-calibration step RRF is designed to
    avoid; running both methods and comparing is the point of Task 2.6.
    """
    if len(runs_with_scores) != len(weights):
        raise ValueError(
            f"got {len(runs_with_scores)} runs but {len(weights)} weights -- must match"
        )

    combined: dict = defaultdict(float)
    for run, weight in zip(runs_with_scores, weights):
        if not run:
            continue
        values = list(run.values())
        lo, hi = min(values), max(values)
        spread = hi - lo
        for doc_id, score in run.items():
            normalized = (score - lo) / spread if spread > 0 else 0.0
            combined[doc_id] += weight * normalized
    return sorted(combined, key=lambda d: combined[d], reverse=True)
