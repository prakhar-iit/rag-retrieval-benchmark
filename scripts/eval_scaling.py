#!/usr/bin/env python3
"""Phase 2, task 2d.1: scaling curve -- index build time, size, and query
latency for BM25, the best dense model, and hybrid fusion, at 5K/10K/20K docs.

This is a SYSTEMS experiment, not a quality experiment: task 2d.1 asks for
build time / index size / p95 latency, not nDCG. Subsampling the corpus to
5K/10K docs would make quality metrics meaningless anyway (most eval queries'
gold documents would simply be missing from the smaller subset, which tests
"is the answer even in this shard" rather than "how good is retrieval") --
so quality is deliberately out of scope here; 2.1-2.7 already cover it at
the full 20K.

Each size subsamples the first N rows of the already-frozen, already-shuffled
20K corpus (df.iloc[:N]) and the matching prefix of all-mpnet-base-v2's
already-cached full-dimension vectors -- no re-encoding needed. Latency is
measured over a fixed 50-query sample (not the full 400) since this script
runs one size per invocation and a full 400-query pass at every stage would
risk exceeding a single shell call's time budget on top of the ~144s dense
index build at 20K; 50 real queries is enough to read a stable p50/p95 for
a systems number, though the ABSOLUTE latencies at N=20K should already
roughly match 2.1-2.7's own 400-query measurements as a sanity check.

The index is always rebuilt fresh (delete then build) rather than reused
across invocations -- see TASKS.md 2c.2 for why a "reuse if the directory
exists" shortcut is NOT safe without verifying the reused index actually
has data.

Usage:
    python scripts/eval_scaling.py --size 5000
    python scripts/eval_scaling.py --size 10000
    python scripts/eval_scaling.py --size 20000

Writes/merges into: results/phase2_scaling.json (keyed by size)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.dense_embed import encode_checkpointed, get_task_prefix  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402
from rss.fusion import weighted_score_fusion  # noqa: E402
from rss.index import (  # noqa: E402
    bm25_index_size_bytes,
    build_bm25,
    build_qdrant,
    delete_qdrant_index,
    qdrant_index_size_bytes,
    search_bm25_with_scores,
    search_with_scores,
)
from rss.static_embed import tokenize  # noqa: E402

DENSE_MODEL = "all-mpnet-base-v2"
FUSION_DEPTH = 100
BM25_WEIGHT = 0.5
LATENCY_SAMPLE = 50  # real eval queries used for latency measurement, not the full 400 -- see module docstring
RESULTS_PATH = REPO_ROOT / "results" / "phase2_scaling.json"


def _load_cached_dense(dirname: str, texts: list[str], kind: str, chunk_size: int):
    cache_dir = REPO_ROOT / "models" / "embed_cache" / dirname
    vectors = encode_checkpointed(None, texts, cache_dir, kind, time_budget=0, chunk_size=chunk_size)
    if vectors is None:
        raise SystemExit(f"{cache_dir} is missing {kind} chunks -- run scripts/encode_locally.py first")
    return vectors


def _percentiles(latencies_ms):
    lat = sorted(latencies_ms)
    return {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, required=True, help="number of docs to subsample (e.g. 5000, 10000, 20000)")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]

    print(f"Loading arXiv corpus and eval set, subsampling to N={args.size}...")
    df_full = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    if args.size > len(df_full):
        raise SystemExit(f"--size {args.size} exceeds corpus size {len(df_full)}")
    df = df_full.iloc[: args.size].reset_index(drop=True)
    doc_ids = df["doc_id"].tolist()
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    print(f"  {len(df)} documents (of {len(df_full)}), {len(records)} eval queries available")

    rng = random.Random(args.seed)
    sample_records = rng.sample(records, min(LATENCY_SAMPLE, len(records)))
    print(f"  measuring latency over a {len(sample_records)}-query sample (see module docstring)")

    print("Tokenizing for BM25...")
    doc_tokens = [tokenize(t) for t in df[corpus_cfg["text_field"]]]
    sample_query_tokens = [tokenize(r["query"]) for r in sample_records]

    doc_prefix = get_task_prefix(DENSE_MODEL, "docs")
    query_prefix = get_task_prefix(DENSE_MODEL, "queries")
    doc_placeholder = [doc_prefix + "x"] * len(df_full)  # full-corpus cache, sliced below
    query_placeholder = [query_prefix + "x"] * len(records)
    doc_vectors_full = _load_cached_dense(DENSE_MODEL, doc_placeholder, "docs", args.chunk_size)
    query_vectors_full = _load_cached_dense(DENSE_MODEL, query_placeholder, "queries", args.chunk_size)
    doc_vectors = doc_vectors_full[: args.size]
    qid_to_vec = dict(zip([r["qid"] for r in records], query_vectors_full))
    sample_query_vectors = [qid_to_vec[r["qid"]] for r in sample_records]

    # --- BM25 ---
    print("Building BM25 index...")
    t0 = time.time()
    bm25_index = build_bm25(doc_tokens)
    bm25_build_time = time.time() - t0
    bm25_size = bm25_index_size_bytes(bm25_index)

    bm25_latencies = []
    for q_tokens in sample_query_tokens:
        t0 = time.perf_counter()
        search_bm25_with_scores(bm25_index, q_tokens, k=20)
        bm25_latencies.append((time.perf_counter() - t0) * 1000)
    print(f"  BM25: build {bm25_build_time:.2f}s, {bm25_size / 1e6:.1f}MB, latency {_percentiles(bm25_latencies)}")

    # --- Dense (all-mpnet-base-v2) ---
    index_path = str(REPO_ROOT / "models" / "qdrant" / f"scaling_{DENSE_MODEL}_{args.size}")
    delete_qdrant_index(index_path)  # always rebuild fresh -- see module docstring
    print(f"Building dense Qdrant index ({args.size} vectors)...")
    payloads = [{"doc_id": d} for d in doc_ids]
    t0 = time.time()
    dense_index = build_qdrant(f"scaling_{DENSE_MODEL}", doc_vectors, payloads, path=index_path)
    dense_build_time = time.time() - t0
    dense_size = qdrant_index_size_bytes(index_path)

    dense_latencies = []
    for qvec in sample_query_vectors:
        t0 = time.perf_counter()
        search_with_scores(dense_index, qvec, k=20)
        dense_latencies.append((time.perf_counter() - t0) * 1000)
    print(f"  Dense: build {dense_build_time:.2f}s, {dense_size / 1e6:.1f}MB, latency {_percentiles(dense_latencies)}")

    # --- Hybrid (weighted fusion, bm25_weight=0.5) -- real end-to-end latency ---
    print("Measuring hybrid end-to-end query latency (bm25 search + dense search + fusion)...")
    hybrid_latencies = []
    for q_tokens, qvec in zip(sample_query_tokens, sample_query_vectors):
        t0 = time.perf_counter()
        bm25_scored = search_bm25_with_scores(bm25_index, q_tokens, k=FUSION_DEPTH)
        dense_scored = search_with_scores(dense_index, qvec, k=FUSION_DEPTH)
        bm25_scores_by_id = {doc_ids[pos]: score for pos, score in bm25_scored}
        dense_scores_by_id = {doc_ids[pos]: score for pos, score in dense_scored}
        weighted_score_fusion([bm25_scores_by_id, dense_scores_by_id], weights=[BM25_WEIGHT, 1.0 - BM25_WEIGHT])
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)
    print(f"  Hybrid (end-to-end): latency {_percentiles(hybrid_latencies)}")

    delete_qdrant_index(index_path)

    entry = {
        "n_docs": args.size,
        "n_latency_queries": len(sample_records),
        "bm25": {
            "build_time_s": bm25_build_time,
            "index_size_bytes_approx": bm25_size,
            "query_latency_ms": _percentiles(bm25_latencies),
        },
        "dense": {
            "model": DENSE_MODEL,
            "build_time_s": dense_build_time,
            "index_size_bytes": dense_size,
            "query_latency_ms": _percentiles(dense_latencies),
        },
        "hybrid": {
            "bm25_weight": BM25_WEIGHT,
            "query_latency_ms_end_to_end": _percentiles(hybrid_latencies),
        },
    }

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results[str(args.size)] = entry
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved (merged) results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
