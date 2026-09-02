"""Phase 1: Word2Vec trained from scratch, in-domain.

The finding under test: for static embeddings, does DOMAIN beat SCALE?
Three models on one eval set -- arXiv-trained, out-of-domain (OpinRank),
and pretrained GoogleNews.

Phrase detection matters here. Scientific text is full of multiword concepts
(reinforcement_learning, attention_mechanism); unigram-only vectors mangle them
and make the 2013 baseline look worse than it is.
"""
from __future__ import annotations


def build_phrases(sentences):
    """gensim Phrases -> bigrams/trigrams. Run BEFORE training."""
    raise NotImplementedError("Task 1.2")


def train(sentences, cfg):
    """Train Word2Vec with the config's hyperparameters."""
    raise NotImplementedError("Task 1.3")


def doc_vector(model, tokens, idf=None):
    """Mean pooling, or IDF-weighted pooling when `idf` is supplied.

    Mean pooling is a weak sentence representation -- that weakness is the
    reason sentence-transformers exist, and showing it is part of the point.
    """
    raise NotImplementedError("Task 1.5")
