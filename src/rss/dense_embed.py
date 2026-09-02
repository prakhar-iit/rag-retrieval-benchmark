"""Phase 2: dense embeddings, and Matryoshka truncation.

MRL is a TRAINING OBJECTIVE, not a compression trick: it front-loads
information so that any prefix of the vector is itself a valid embedding.
That is why truncating an MRL-trained model barely moves quality while
truncating a non-MRL model degrades badly -- running both is the experiment.
"""
from __future__ import annotations
import numpy as np


def encode(model_name: str, texts, batch_size: int = 64) -> np.ndarray:
    """Encode texts; cache to disk keyed by (model, corpus hash)."""
    raise NotImplementedError("Task 2.2 / 2.3")


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Slice to `dim` dimensions and RENORMALISE.

    Cosine similarity assumes unit vectors; slicing breaks the norm.
    Forgetting this is the classic MRL bug and it fails silently.
    """
    raise NotImplementedError("Task 2.5")


def assert_unit_norm(vectors: np.ndarray, tol: float = 1e-5) -> None:
    """Guard against the bug above."""
    raise NotImplementedError("Task 2.5")
