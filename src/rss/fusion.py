"""Phase 2: hybrid retrieval.

Lexical and dense methods fail on DIFFERENT queries -- that is the whole
argument for fusing them. RRF is the usual default because it needs no score
calibration across methods.
"""
from __future__ import annotations


def reciprocal_rank_fusion(runs, k: int = 60):
    """RRF over N ranked lists: score = sum(1 / (k + rank))."""
    raise NotImplementedError("Task 2.6")


def weighted_score_fusion(runs, weights):
    """Normalised weighted sum. Requires score calibration -- compare against RRF."""
    raise NotImplementedError("Task 2.6")
