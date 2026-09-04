#!/usr/bin/env python3
"""Phase 1, task 1.6 (out-of-domain leg): train a Word2Vec model on OpinRank
hotel reviews, using the exact same pipeline and hyperparameters as the
in-domain arXiv model, so the only variable that differs is domain.

Source: the OpinRank hotel-review excerpt Kavita Ganesan (the dataset's own
author) distributes for the classic gensim Word2Vec tutorial --
https://github.com/kavgan/nlp-in-practice/blob/master/word2vec/reviews_data.txt.gz
(tab-separated: date, review title, review text; ~256K reviews, ~42M words).
The original OpinRank hosting (kavita-ganesan.com/entity-ranking-data-code)
is no longer reachable from this environment.

Sampled down to the SAME document count as the arXiv corpus (corpus.sample_size,
same seed) rather than trained on the full 256K reviews: training on a
comparably-sized corpus isolates domain as the only variable between this
model and the arXiv one. Training on all 256K would confound domain
mismatch with a ~12x scale difference, muddying the "does domain beat scale"
finding this comparison exists to test.

Usage:
    python scripts/train_word2vec_opinrank.py [--config configs/default.yaml]
                                               [--reviews data/opinrank/reviews_data.txt.gz]

Writes: models/word2vec_opinrank.model
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rss.static_embed import build_phrases, tokenize_corpus, train  # noqa: E402

_MIN_WORDS = 15


def load_reviews(path: str) -> pd.DataFrame:
    """Parse the tab-separated (date, title, review) file into a DataFrame
    with one row per review, filtering empty/very short reviews the same way
    corpus._sample_and_id filters degenerate arXiv abstracts."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            review = parts[2].strip()
            if review:
                rows.append(review)
    df = pd.DataFrame({"review": rows})
    df = df.drop_duplicates(subset=["review"])
    df = df[df["review"].str.split().map(len) >= _MIN_WORDS]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reviews", default="data/opinrank/reviews_data.txt.gz")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    seed = config["seed"]
    sample_size = config["corpus"]["sample_size"]
    w2v_cfg = config["word2vec"]

    print(f"Loading reviews from {args.reviews}")
    df = load_reviews(args.reviews)
    print(f"  {len(df)} reviews after dropping empty/short/duplicate ones")

    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(df)} reviews (seed={seed}), matching the arXiv corpus size")

    print("Tokenizing...")
    t0 = time.time()
    sentences = tokenize_corpus(df, text_field="review")
    n_tokens = sum(len(s) for s in sentences)
    print(f"  {n_tokens:,} tokens across {len(sentences)} documents ({time.time() - t0:.1f}s)")

    if w2v_cfg.get("use_phrases", True):
        print("Detecting bigram/trigram phrases...")
        t0 = time.time()
        sentences = build_phrases(sentences)
        n_tokens_after = sum(len(s) for s in sentences)
        print(f"  {n_tokens:,} unigram tokens -> {n_tokens_after:,} tokens after phrase merging "
              f"({time.time() - t0:.1f}s)")

    print(f"Training Word2Vec: {dict(w2v_cfg)}")
    t0 = time.time()
    model = train(sentences, w2v_cfg, seed=seed)
    print(f"  Trained in {time.time() - t0:.1f}s. Vocab size: {len(model.wv)}")

    print("\n=== Nearest neighbours (out-of-domain sanity check) ===")
    for term in ["transformer", "attention", "diffusion", "hotel", "room"]:
        if term in model.wv:
            neighbours = model.wv.most_similar(term, topn=5)
            formatted = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
            print(f"  {term}: {formatted}")
        else:
            print(f"  {term}: not in vocabulary")

    out_path = Path("models") / "word2vec_opinrank.model"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
