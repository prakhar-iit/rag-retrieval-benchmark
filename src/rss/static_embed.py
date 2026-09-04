"""Phase 1: Word2Vec trained from scratch, in-domain.

The finding under test: for static embeddings, does DOMAIN beat SCALE?
Three models on one eval set -- arXiv-trained, out-of-domain (OpinRank),
and pretrained GoogleNews.

Phrase detection matters here. Scientific text is full of multiword concepts
(reinforcement_learning, attention_mechanism); unigram-only vectors mangle them
and make the 2013 baseline look worse than it is.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Tokens must start with a letter, so pure numbers ("2019", "96.2") are dropped
# but alphanumeric domain jargon (resnet50, gpt3, bert) survives intact.
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, stripping punctuation.

    Deliberately simple (a single regex, no external tokenizer dependency) --
    consistent with the rest of this repo's minimal-dependency style. Numbers
    and punctuation carry no useful signal for word-co-occurrence embeddings.
    """
    return _TOKEN_RE.findall(text.lower())


def tokenize_corpus(df: pd.DataFrame, text_field: str = "abstract") -> list[list[str]]:
    """Tokenize every document in `df[text_field]`, preserving row order (so
    the result lines up positionally with df's doc_id column)."""
    return [tokenize(t) for t in df[text_field]]


def build_phrases(
    sentences: Sequence[Sequence[str]],
    min_count: int = 5,
    threshold: int = 10,
) -> list[list[str]]:
    """gensim Phrases -> bigrams/trigrams. Run BEFORE training.

    Two passes: the first pass merges frequent unigram pairs into bigrams
    (`reinforcement` + `learning` -> `reinforcement_learning`); the second
    pass runs Phrases again over the bigram-merged sentences, so a frequent
    bigram + adjacent unigram can merge into a trigram
    (`reinforcement_learning` + `agent` -> `reinforcement_learning_agent`).
    Returns the phrase-merged sentences, ready for Word2Vec.
    """
    from gensim.models.phrases import Phrases, Phraser

    bigram_model = Phrases(sentences, min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram_model)
    bigrammed = [bigram_phraser[s] for s in sentences]

    trigram_model = Phrases(bigrammed, min_count=min_count, threshold=threshold)
    trigram_phraser = Phraser(trigram_model)
    return [trigram_phraser[s] for s in bigrammed]


def train(sentences: Sequence[Sequence[str]], cfg: Mapping, seed: int = 13):
    """Train Word2Vec with the config's hyperparameters (configs/default.yaml
    `word2vec:` section).

    Note on determinism: gensim's Word2Vec is only bit-for-bit reproducible
    with `workers=1` (multi-threaded SGD updates race regardless of seed).
    Given this corpus (~20K abstracts, a few million tokens) trains in well
    under a minute even single-threaded, we default to workers=1 so results
    are exactly reproducible across runs, matching the rest of this project's
    "same seed, same output" discipline.
    """
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=sentences,
        sg=cfg.get("sg", 1),
        vector_size=cfg.get("vector_size", 300),
        window=cfg.get("window", 5),
        min_count=cfg.get("min_count", 5),
        negative=cfg.get("negative", 10),
        epochs=cfg.get("epochs", 10),
        workers=cfg.get("workers", 1),
        seed=seed,
    )


def compute_idf(tokenized_docs: Iterable[Sequence[str]]) -> dict[str, float]:
    """Smoothed IDF over a tokenized corpus: idf(t) = log(N / (1 + df(t))) + 1.

    The "+1" smoothing (as in scikit-learn's TfidfVectorizer) keeps weights
    positive and finite even for terms that appear in every document, and
    avoids a zero weight blowing up mean pooling for short documents.
    """
    tokenized_docs = list(tokenized_docs)
    n_docs = len(tokenized_docs)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))
    return {term: math.log(n_docs / (1 + df)) + 1.0 for term, df in doc_freq.items()}


def doc_vector(
    keyed_vectors, tokens: Sequence[str], idf: Mapping[str, float] | None = None
) -> np.ndarray:
    """Pool token vectors into one document vector.

    `keyed_vectors` is a gensim KeyedVectors instance -- for a Word2Vec model
    trained in this module that's `model.wv`; a standalone pretrained
    KeyedVectors load (e.g. GoogleNews vectors) is passed directly, since it
    has no enclosing Word2Vec model. Both support `in` / `[...]` the same way,
    which is all this function needs -- that's why task 1.6's three-way
    comparison (in-domain, out-of-domain, pretrained) can share one code path.

    Mean pooling when `idf` is None (every in-vocabulary token weighted
    equally); IDF-weighted pooling when `idf` is supplied (rare/informative
    tokens weighted more than common ones). Out-of-vocabulary tokens are
    skipped; a document with no in-vocabulary tokens returns the zero vector.

    Mean pooling is a weak sentence representation -- that weakness is the
    reason sentence-transformers exist, and showing it is part of the point.
    """
    dim = keyed_vectors.vector_size
    vecs = []
    weights = []
    for t in tokens:
        if t in keyed_vectors:
            vecs.append(keyed_vectors[t])
            weights.append(idf.get(t, 1.0) if idf is not None else 1.0)

    if not vecs:
        return np.zeros(dim, dtype=np.float32)

    vecs_arr = np.asarray(vecs, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32)
    if weights_arr.sum() == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.average(vecs_arr, axis=0, weights=weights_arr).astype(np.float32)
