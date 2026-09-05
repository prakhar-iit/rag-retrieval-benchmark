#!/usr/bin/env python3
"""Phase 2, task 2c.2: cross-encoder reranking, before/after.

"Before" is the best first-stage retriever this project has: hybrid weighted
fusion at bm25_weight=0.5 (Task 2.6's own best result, 0.9885 nDCG@10) over a
100-candidate BM25 + all-mpnet-base-v2 pool -- reranking on top of the
already-best baseline is the honest test (reranking a weak baseline would
overstate the gain, since there'd be more obvious headroom to claim).

"After" takes that same fused ranking's top `rerank.top_n` (50, from
configs/default.yaml) candidates, scores them jointly with a cross-encoder
(BAAI/bge-reranker-base via rss.rerank), and re-sorts by the cross-encoder's
own score. Added latency is measured as just the rerank() call itself --
the incremental cost a production pipeline would pay on top of whatever
first-stage retrieval it already does, not first-stage retrieval time again.

A cross-encoder pass over 50 candidates takes ~2s/query on this CPU-only dev
box -- 400 queries would be ~15 minutes, well past a single shell call's time
budget. So, like scripts/eval_dense.py and eval_mrl_sweep.py, this script is
checkpointed and resumable: per-query results land in a JSON checkpoint file
after each query, and the dense Qdrant index (the slow-to-build part) is left
on disk between calls rather than torn down and rebuilt every invocation --
run this script repeatedly; it picks up from the last completed query.

Usage:
    python scripts/eval_rerank.py [--config configs/default.yaml] [--chunk-size 500] [--time-budget 150]

Writes: results/phase2_rerank.json (final), results/.rerank_checkpoint.json (interim)
"""
from __future__ import annotations

import argparse
import json
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
    build_bm25,
    build_qdrant,
    delete_qdrant_index,
    search_bm25_with_scores,
    search_with_scores,
)
from rss.metrics import evaluate  # noqa: E402
from rss.rerank import load_reranker, rerank  # noqa: E402
from rss.static_embed import tokenize  # noqa: E402

DENSE_MODEL = "all-mpnet-base-v2"
FUSION_DEPTH = 100  # same pool depth as eval_hybrid.py, for the same reason
BM25_WEIGHT = 0.5  # Task 2.6's own best weighted-fusion point
CHECKPOINT_PATH = REPO_ROOT / "results" / ".rerank_checkpoint.json"


def _load_cached_dense(dirname: str, texts: list[str], kind: str, chunk_size: int):
    cache_dir = REPO_ROOT / "models" / "embed_cache" / dirname
    vectors = encode_checkpointed(None, texts, cache_dir, kind, time_budget=0, chunk_size=chunk_size)
    if vectors is None:
        raise SystemExit(
            f"{cache_dir} is missing {kind} chunks for chunk_size={chunk_size} -- "
            f"run scripts/encode_locally.py --model {dirname} first"
        )
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--chunk-size", type=int, default=500, help="must match the cache's own chunk_size")
    parser.add_argument("--time-budget", type=float, default=150.0, help="seconds of query-loop work per call")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]
    top_k = max(k_values)
    rerank_cfg = config["rerank"]
    rerank_top_n = rerank_cfg["top_n"]
    reranker_model = rerank_cfg["model"]

    print(f"Loading arXiv corpus and eval set (dense side: {DENSE_MODEL}, reranker: {reranker_model})...")
    df = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    doc_ids = df["doc_id"].tolist()
    doc_texts_by_id = dict(zip(doc_ids, df[corpus_cfg["text_field"]]))
    qids = [r["qid"] for r in records]
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}
    print(f"  {len(df)} documents, {len(records)} eval queries")

    print("Tokenizing for BM25...")
    doc_tokens = [tokenize(t) for t in df[corpus_cfg["text_field"]]]

    doc_prefix = get_task_prefix(DENSE_MODEL, "docs")
    query_prefix = get_task_prefix(DENSE_MODEL, "queries")
    doc_placeholder = [doc_prefix + "x"] * len(df)
    query_placeholder = [query_prefix + "x"] * len(records)
    doc_vectors = _load_cached_dense(DENSE_MODEL, doc_placeholder, "docs", args.chunk_size)
    query_vectors = _load_cached_dense(DENSE_MODEL, query_placeholder, "queries", args.chunk_size)
    print(f"  loaded {len(doc_vectors)} cached doc vectors, {len(query_vectors)} cached query vectors")

    print("Building BM25 index...")
    bm25_index = build_bm25(doc_tokens)

    # NOTE: earlier versions of this script tried to reuse an existing Qdrant
    # directory across resumed calls without verifying it actually held data --
    # a killed mid-build process can leave a valid-looking directory (meta.json
    # + .lock) with ZERO points in it, and querying THAT empty-but-"existing"
    # collection silently makes the dense side of every fusion contribute
    # nothing (weighted_score_fusion just falls through to BM25's own ranking).
    # That happened once already: a "before" run here came back numerically
    # identical to BM25-only's own nDCG@10 to floating-point noise (diff
    # ~2e-16), which is how the bug was caught -- see TASKS.md 2c.2. Simplest
    # correct fix: always rebuild fresh, same as eval_hybrid.py/eval_dense.py
    # do, rather than trying to cheaply reuse a directory whose contents were
    # never verified.
    index_path = str(REPO_ROOT / "models" / "qdrant" / f"rerank_{DENSE_MODEL}")
    delete_qdrant_index(index_path)
    print("Building dense Qdrant index...")
    payloads = [{"doc_id": d} for d in doc_ids]
    dense_index = build_qdrant(f"rerank_{DENSE_MODEL}", doc_vectors, payloads, path=index_path)

    print(f"Loading cross-encoder {reranker_model}...")
    t0 = time.time()
    reranker = load_reranker(reranker_model, local_dir=str(REPO_ROOT / "models" / "hf"))
    print(f"  loaded in {time.time() - t0:.1f}s")

    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text())
        before_run = checkpoint["before_run"]
        after_run = checkpoint["after_run"]
        rerank_latencies = checkpoint["rerank_latencies_ms"]
        done_qids = set(before_run.keys())
        print(f"  resuming from checkpoint: {len(done_qids)}/{len(qids)} queries already done")
    else:
        before_run = {}
        after_run = {}
        rerank_latencies = []
        done_qids = set()

    query_by_qid = {qid: (q_tokens, qvec, record) for qid, q_tokens, qvec, record in zip(
        qids, [tokenize(r["query"]) for r in records], query_vectors, records
    )}

    print(f"Querying (fusion pool {FUSION_DEPTH}, reranking top {rerank_top_n}, evaluating at top-{top_k})...")
    t_start = time.time()
    for qid in qids:
        if qid in done_qids:
            continue
        if time.time() - t_start > args.time_budget:
            print(f"  time budget reached ({len(before_run)}/{len(qids)} done) -- re-run this script to continue")
            CHECKPOINT_PATH.write_text(json.dumps({
                "before_run": before_run, "after_run": after_run, "rerank_latencies_ms": rerank_latencies,
            }))
            return

        q_tokens, qvec, record = query_by_qid[qid]
        bm25_scored = search_bm25_with_scores(bm25_index, q_tokens, k=FUSION_DEPTH)
        dense_scored = search_with_scores(dense_index, qvec, k=FUSION_DEPTH)
        bm25_scores_by_id = {doc_ids[pos]: score for pos, score in bm25_scored}
        dense_scores_by_id = {doc_ids[pos]: score for pos, score in dense_scored}

        fused = weighted_score_fusion(
            [bm25_scores_by_id, dense_scores_by_id], weights=[BM25_WEIGHT, 1.0 - BM25_WEIGHT]
        )
        before_run[qid] = fused[:top_k]

        candidates = [(doc_id, doc_texts_by_id[doc_id]) for doc_id in fused[:rerank_top_n]]
        t0 = time.perf_counter()
        reranked = rerank(reranker, record["query"], candidates, top_n=rerank_top_n)
        rerank_latencies.append((time.perf_counter() - t0) * 1000)
        after_run[qid] = reranked[:top_k]

        # checkpoint every 10 queries so a mid-batch interruption loses little
        if len(before_run) % 10 == 0:
            CHECKPOINT_PATH.write_text(json.dumps({
                "before_run": before_run, "after_run": after_run, "rerank_latencies_ms": rerank_latencies,
            }))

    before_scores = evaluate(before_run, gold_map, k_values)
    after_scores = evaluate(after_run, gold_map, k_values)
    rerank_latencies_sorted = sorted(rerank_latencies)
    p50 = rerank_latencies_sorted[len(rerank_latencies_sorted) // 2]
    p95 = rerank_latencies_sorted[int(len(rerank_latencies_sorted) * 0.95)]

    print(f"\nBefore rerank (hybrid weighted, bm25_weight={BM25_WEIGHT}): nDCG@10={before_scores['mean']['ndcg@10']:.4f}")
    print(f"After rerank ({reranker_model}, top_n={rerank_top_n}):        nDCG@10={after_scores['mean']['ndcg@10']:.4f}")
    print(f"Added rerank latency: p50={p50:.1f}ms  p95={p95:.1f}ms")

    out = {
        "before": {"method": f"hybrid_weighted_bm25={BM25_WEIGHT}", "mean": before_scores["mean"], "n_queries": before_scores["n_queries"]},
        "after": {
            "method": reranker_model,
            "mean": after_scores["mean"],
            "n_queries": after_scores["n_queries"],
            "rerank_top_n": rerank_top_n,
            "added_latency_ms": {"p50": p50, "p95": p95},
        },
        "config": {"fusion_depth": FUSION_DEPTH, "bm25_weight": BM25_WEIGHT, "rerank_top_n": rerank_top_n, "dense_model": DENSE_MODEL},
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_rerank.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {out_path}")

    CHECKPOINT_PATH.unlink(missing_ok=True)
    delete_qdrant_index(index_path)


if __name__ == "__main__":
    main()
