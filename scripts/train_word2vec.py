#!/usr/bin/env python3
"""Phase 1, tasks 1.1-1.5: tokenize, phrase-detect, train Word2Vec in-domain,
sanity-check nearest neighbours, and save an IDF table for doc-vector pooling.

Usage:
    python scripts/train_word2vec.py [--config configs/default.yaml]

Reads the frozen corpus (configs/default.yaml corpus.frozen_path), trains a
single in-domain Word2Vec model on it, and writes:
    models/word2vec_arxiv.model   (gensim KeyedVectors-backed model)
    models/idf_arxiv.json         (term -> IDF weight, for doc_vector pooling)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.static_embed import build_phrases, compute_idf, tokenize_corpus, train  # noqa: E402

_SANITY_TERMS = ["transformer", "attention", "diffusion", "embedding", "gradient"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    seed = config["seed"]
    corpus_cfg = config["corpus"]
    w2v_cfg = config["word2vec"]

    print(f"Loading frozen corpus from {corpus_cfg['frozen_path']}")
    df = load_frozen(corpus_cfg["frozen_path"])
    print(f"Loaded {len(df)} documents")

    print("Tokenizing...")
    t0 = time.time()
    sentences = tokenize_corpus(df, text_field=corpus_cfg["text_field"])
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

    print("\n=== Sanity check: nearest neighbours ===")
    for term in _SANITY_TERMS:
        if term in model.wv:
            neighbours = model.wv.most_similar(term, topn=8)
            formatted = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
            print(f"  {term}: {formatted}")
        else:
            print(f"  {term}: not in vocabulary (min_count={w2v_cfg.get('min_count')})")

    print("\nComputing IDF table...")
    idf = compute_idf(sentences)
    print(f"  {len(idf):,} terms")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "word2vec_arxiv.model"
    idf_path = out_dir / "idf_arxiv.json"
    model.save(str(model_path))
    idf_path.write_text(json.dumps(idf))
    print(f"\nSaved model to {model_path}")
    print(f"Saved IDF table to {idf_path}")


if __name__ == "__main__":
    main()
