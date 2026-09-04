#!/usr/bin/env python3
"""Phase 2, task 2.1: BM25 baseline on the eval set.

Uses the same tokenizer as the Word2Vec baselines (rss.static_embed.tokenize)
so preprocessing is held constant across methods.

Usage:
    python scripts/eval_bm25.py [--config configs/default.yaml]

Writes: results/phase2_bm25.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402
from rss.index import bm25_index_size_bytes, build_bm25, search_bm25  # noqa: E402
from rss.metrics import evaluate  # noqa: E402
from rss.static_embed import tokenize  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]
    top_k = max(k_values)

    print("Loading arXiv corpus and eval set...")
    df = load_frozen(corpus_cfg["frozen_path"])
    records = load_jsonl(evalset_cfg["path"])
    doc_ids = df["doc_id"].tolist()
    qids = [r["qid"] for r in records]
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}
    print(f"  {len(df)} documents, {len(records)} eval queries")

    print("Tokenizing...")
    doc_tokens = [tokenize(t) for t in df[corpus_cfg["text_field"]]]
    query_tokens = [tokenize(r["query"]) for r in records]

    print("Building BM25 index...")
    t0 = time.time()
    index = build_bm25(doc_tokens)
    build_time = time.time() - t0
    index_size = bm25_index_size_bytes(index)
    print(f"  built in {build_time:.2f}s, ~{index_size / 1e6:.1f}MB (pickled, approximate)")

    print("Querying...")
    t0 = time.time()
    latencies = []
    run = {}
    for qid, q_tokens in zip(qids, query_tokens):
        q0 = time.perf_counter()
        top_idx = search_bm25(index, q_tokens, k=top_k)
        latencies.append((time.perf_counter() - q0) * 1000)
        run[qid] = [doc_ids[i] for i in top_idx]
    total_query_time = time.time() - t0

    scores = evaluate(run, gold_map, k_values)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\nnDCG@10={scores['mean']['ndcg@10']:.4f}  "
          f"Recall@10={scores['mean']['recall@10']:.4f}  MRR={scores['mean']['mrr']:.4f}")
    print(f"Query latency: p50={p50:.2f}ms  p95={p95:.2f}ms  "
          f"({len(qids)} queries in {total_query_time:.2f}s total)")

    out = {
        "mean": scores["mean"],
        "n_queries": scores["n_queries"],
        "index_build_time_s": build_time,
        "index_size_bytes_approx": index_size,
        "query_latency_ms": {"p50": p50, "p95": p95},
    }
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_bm25.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
