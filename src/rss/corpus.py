"""Phase 0: load, sample and freeze the arXiv corpus.

One abstract = one document. The sample is frozen to disk so every method
in the study sees byte-identical inputs.
"""
from __future__ import annotations


def load_and_sample(dataset: str, sample_size: int, seed: int):
    """Load the HF dataset, sample `sample_size` abstracts, return a DataFrame."""
    raise NotImplementedError("Task 0.1 / 0.2")


def freeze(df, path: str) -> None:
    """Write the frozen sample. Never re-sample after this point."""
    raise NotImplementedError("Task 0.2")
