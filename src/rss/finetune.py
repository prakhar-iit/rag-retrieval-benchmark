"""Phase 2b: fine-tune a sentence-transformer on domain pairs.

Answers: should you train your own embedding model for RAG?

Expected arc, with numbers at each step:
  1. Word2Vec from scratch, in-domain      -> beats out-of-domain Word2Vec
  2. Generic pretrained sentence-transformer -> likely beats BOTH
  3. That model fine-tuned on domain pairs   -> best of all

Lesson: training from scratch is almost never right; fine-tuning an existing
model on domain pairs frequently is.

MUST fine-tune and evaluate on DISJOINT slices or the result is meaningless.
"""
from __future__ import annotations


def finetune(base_model: str, pairs, epochs: int = 1):
    """MultipleNegativesRankingLoss -- in-batch negatives, the same contrastive
    objective that made SBERT work."""
    raise NotImplementedError("Task 2b.1")
