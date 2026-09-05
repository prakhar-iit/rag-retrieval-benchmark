#!/usr/bin/env python3
"""Phase 2, task 2.6: hybrid retrieval -- fuse BM25 with the best dense model.

Two fusion methods are compared, not just one, per configs/default.yaml's
`hybrid` block and the design already laid out in rss.fusion's docstrings:

  - Reciprocal rank fusion (RRF): needs no score calibration across methods,
    combines two RANKED LISTS via 1/(k+rank).
  - Weighted score fusion: needs each method's raw scores min-max normalised
    to [0, 1] first (BM25's unbounded scores vs. cosine's [-1, 1] would
    otherwise let BM25 dominate any sum regardless of weight), then combines
    via a weighted sum -- swept across configs/default.yaml's
    `hybrid.weight_sweep`, interpreted here as the WEIGHT ON BM25 (dense gets
    1 - weight), so weight=0.0 is pure dense and weight=1.0 is pure BM25.

The dense side is `all-mpnet-base-v2`, the corrected "best dense model" pick
from Task 2.3 (NOT nomic -- nomic's own headline number needed its required
task-instruction prefix to be read correctly, and once read correctly mpnet
scored higher). Its vectors are read back from the already-completed
embed_cache (see scripts/encode_locally.py / eval_dense.py) rather than
re-encoded -- same "trust the resumable cache" pattern as eval_mrl_sweep.py.

Both methods fuse over a candidate pool deeper than the final top-k (see
FUSION_DEPTH) -- fusing only over each method's already-truncated top-k
would hide exactly the cross-method complementarity RRF/weighted fusion
exist to capture (a doc BM25 ranks 15th and dense ranks 3rd should be able
to surface in the fused top-10, which requires seeing rank 15 in the first
place).

Usage:
    python scripts/eval_hybrid.py [--config configs/default.yaml] [--chunk-size 500]

Writes: results/phase2_hybrid.json
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
from rss.fusion import reciprocal_rank_fusion, weighted_score_fusion  # noqa: E402
from rss.index import (  # noqa: E402
    build_bm25,
    build_qdrant,
    delete_qdrant_index,
    search_bm25_with_scores,
    search_with_scores,
)
from rss.metrics import evaluate  # noqa: E402
from rss.static_embed import tokenize  # noqa: E402

DENSE_MODEL = "all-mpnet-base-v2"

# Candidate pool depth for fusion -- deeper than the final top_k (20, from
# configs/default.yaml's metrics.k_values) so a doc that's mediocre-but-present
# in one method's ranking can still be pulled up by the other. 100 is a
# common default for first-stage retrieval depth in hybrid/rerank pipelines;
# there's nothing sacred about it here beyond "clearly deeper than what we
# score against."
FUSION_DEPTH = 100


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
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]
    top_k = max(k_values)
    rrf_k = config["hybrid"]["rrf_k"]
    weight_sweep = config["hybrid"]["weight_sweep"]

    print(f"Loading arXiv corpus and eval set (dense side: {DENSE_MODEL})...")
    df = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    doc_ids = df["doc_id"].tolist()
    qids = [r["qid"] for r in records]
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}
    print(f"  {len(df)} documents, {len(records)} eval queries")

    print("Tokenizing for BM25...")
    doc_tokens = [tokenize(t) for t in df[corpus_cfg["text_field"]]]
    query_tokens = [tokenize(r["query"]) for r in records]

    doc_prefix = get_task_prefix(DENSE_MODEL, "docs")
    query_prefix = get_task_prefix(DENSE_MODEL, "queries")
    # content doesn't matter for a cache read-back, only length + prefix-ness
    # (encode_checkpointed with model=None never calls the model -- it only
    # reassembles chunks that already exist on disk from 2.2's real encode)
    doc_placeholder = [doc_prefix + "x"] * len(df)
    query_placeholder = [query_prefix + "x"] * len(records)
    doc_vectors = _load_cached_dense(DENSE_MODEL, doc_placeholder, "docs", args.chunk_size)
    query_vectors = _load_cached_dense(DENSE_MODEL, query_placeholder, "queries", args.chunk_size)
    print(f"  loaded {len(doc_vectors)} cached doc vectors, {len(query_vectors)} cached query vectors")

    print("Building BM25 index...")
    bm25_index = build_bm25(doc_tokens)

    print("Building dense Qdrant index...")
    index_path = str(REPO_ROOT / "models" / "qdrant" / f"hybrid_{DENSE_MODEL}")
    delete_qdrant_index(index_path)
    payloads = [{"doc_id": d} for d in doc_ids]
    dense_index = build_qdrant(f"hybrid_{DENSE_MODEL}", doc_vectors, payloads, path=index_path)

    print(f"Querying (pool depth {FUSION_DEPTH}, evaluating at top-{top_k})...")
    bm25_only_run: dict[str, list[str]] = {}
    dense_only_run: dict[str, list[str]] = {}
    rrf_run: dict[str, list[str]] = {}
    weighted_runs: dict[float, dict[str, list[str]]] = {w: {} for w in weight_sweep}
    rrf_latencies = []
    weighted_latencies: dict[float, list[float]] = {w: [] for w in weight_sweep}

    for qid, q_tokens, qvec in zip(qids, query_tokens, query_vectors):
        bm25_scored = search_bm25_with_scores(bm25_index, q_tokens, k=FUSION_DEPTH)
        dense_scored = search_with_scores(dense_index, qvec, k=FUSION_DEPTH)

        bm25_ranked_ids = [doc_ids[pos] for pos, _score in bm25_scored]
        dense_ranked_ids = [doc_ids[pos] for pos, _score in dense_scored]
        bm25_only_run[qid] = bm25_ranked_ids[:top_k]
        dense_only_run[qid] = dense_ranked_ids[:top_k]

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_ranked_ids, dense_ranked_ids], k=rrf_k)
        rrf_latencies.append((time.perf_counter() - t0) * 1000)
        rrf_run[qid] = fused[:top_k]

        bm25_scores_by_id = {doc_ids[pos]: score for pos, score in bm25_scored}
        dense_scores_by_id = {doc_ids[pos]: score for pos, score in dense_scored}
        for w in weight_sweep:
            t0 = time.perf_counter()
            fused_w = weighted_score_fusion([bm25_scores_by_id, dense_scores_by_id], weights=[w, 1.0 - w])
            weighted_latencies[w].append((time.perf_counter() - t0) * 1000)
            weighted_runs[w][qid] = fused_w[:top_k]

    def _summarize(run, latencies_ms=None):
        scores = evaluate(run, gold_map, k_values)
        out = {"mean": scores["mean"], "n_queries": scores["n_queries"]}
        if latencies_ms:
            lat = sorted(latencies_ms)
            out["fusion_latency_ms"] = {
                "p50": lat[len(lat) // 2],
                "p95": lat[int(len(lat) * 0.95)],
            }
        return out

    results = {
        "bm25_only": _summarize(bm25_only_run),
        "dense_only": _summarize(dense_only_run),
        "rrf": _summarize(rrf_run, rrf_latencies),
        "weighted": {
            f"bm25_weight={w}": _summarize(weighted_runs[w], weighted_latencies[w]) for w in weight_sweep
        },
        "config": {"fusion_depth": FUSION_DEPTH, "rrf_k": rrf_k, "weight_sweep": weight_sweep, "dense_model": DENSE_MODEL},
    }

    print(f"\nBM25 only:    nDCG@10={results['bm25_only']['mean']['ndcg@10']:.4f}")
    print(f"Dense only:   nDCG@10={results['dense_only']['mean']['ndcg@10']:.4f}")
    print(f"RRF (k={rrf_k}): nDCG@10={results['rrf']['mean']['ndcg@10']:.4f}")
    for w in weight_sweep:
        r = results["weighted"][f"bm25_weight={w}"]
        print(f"Weighted (bm25_w={w}): nDCG@10={r['mean']['ndcg@10']:.4f}")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_hybrid.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")

    delete_qdrant_index(index_path)


if __name__ == "__main__":
    main()
