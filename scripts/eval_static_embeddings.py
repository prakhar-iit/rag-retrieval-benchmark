#!/usr/bin/env python3
"""Phase 1, task 1.6: three-way Word2Vec comparison on the eval set.

Compares in-domain (arXiv-trained), out-of-domain (OpinRank-trained) and
pretrained (GoogleNews) Word2Vec vectors -- each with mean pooling AND
IDF-weighted pooling -- by using them to retrieve arXiv abstracts for the
400 eval queries, and scoring with the Phase 0 metrics harness.

Retrieval here is brute-force cosine similarity over the full 20K-document
corpus (a 20000x400 matmul -- trivial at this corpus size). This is NOT the
Phase 2 index; Qdrant/ANN indexing is 2.4, deliberately out of scope here.

Usage:
    python scripts/eval_static_embeddings.py [--config configs/default.yaml]

Writes: results/phase1_word2vec_comparison.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402
from rss.metrics import evaluate  # noqa: E402
from rss.static_embed import doc_vector, tokenize  # noqa: E402


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard docs/queries with zero vectors (all-OOV)
    return mat / norms


def _build_matrix(keyed_vectors, token_lists, idf, dim) -> np.ndarray:
    mat = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        mat[i] = doc_vector(keyed_vectors, tokens, idf=idf)
    return mat


def _run_and_score(doc_mat, query_mat, doc_ids, qids, gold_map, k_values, top_k=20):
    doc_mat = _l2_normalize(doc_mat)
    query_mat = _l2_normalize(query_mat)
    sims = query_mat @ doc_mat.T  # (n_queries, n_docs)

    run = {}
    for i, qid in enumerate(qids):
        top_idx = np.argpartition(-sims[i], top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[i][top_idx])]
        run[qid] = [doc_ids[j] for j in top_idx]

    return evaluate(run, gold_map, k_values)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]

    print("Loading arXiv corpus and eval set...")
    df = load_frozen(corpus_cfg["frozen_path"])
    records = load_jsonl(evalset_cfg["path"])
    print(f"  {len(df)} documents, {len(records)} eval queries")

    doc_ids = df["doc_id"].tolist()
    qids = [r["qid"] for r in records]
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}

    print("Tokenizing documents and queries...")
    doc_tokens = [tokenize(t) for t in df[corpus_cfg["text_field"]]]
    query_tokens = [tokenize(r["query"]) for r in records]

    idf = json.loads(Path("models/idf_arxiv.json").read_text())

    print("Loading models...")
    from gensim.models import Word2Vec, KeyedVectors

    models = {}
    t0 = time.time()
    models["in_domain_arxiv"] = Word2Vec.load("models/word2vec_arxiv.model").wv
    models["out_of_domain_opinrank"] = Word2Vec.load("models/word2vec_opinrank.model").wv
    models["pretrained_googlenews"] = KeyedVectors.load_word2vec_format(
        "models/word2vec-google-news-300.gz", binary=True, limit=500_000
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    results = {}
    for model_name, kv in models.items():
        dim = kv.vector_size
        for pooling_name, use_idf in [("mean", False), ("idf_weighted", True)]:
            t0 = time.time()
            pool_idf = idf if use_idf else None
            doc_mat = _build_matrix(kv, doc_tokens, pool_idf, dim)
            query_mat = _build_matrix(kv, query_tokens, pool_idf, dim)
            scores = _run_and_score(doc_mat, query_mat, doc_ids, qids, gold_map, k_values)
            key = f"{model_name}/{pooling_name}"
            results[key] = scores["mean"] | {"n_queries": scores["n_queries"]}
            print(f"  {key}: nDCG@10={scores['mean']['ndcg@10']:.4f}  "
                  f"Recall@10={scores['mean']['recall@10']:.4f}  "
                  f"MRR={scores['mean']['mrr']:.4f}  ({time.time() - t0:.1f}s)")

    print("\n=== Summary (nDCG@10 / Recall@10 / MRR) ===")
    for key, scores in results.items():
        print(f"  {key:38s}  {scores['ndcg@10']:.4f}  {scores['recall@10']:.4f}  {scores['mrr']:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_word2vec_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
