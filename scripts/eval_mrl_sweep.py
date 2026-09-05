#!/usr/bin/env python3
"""Phase 2, task 2.5: MRL truncation sweep.

MRL is a TRAINING OBJECTIVE, not a compression trick: it front-loads
information so any prefix of the vector is itself a valid embedding. This
sweeps nomic-embed-text-v1.5 (MRL-trained) down through
dense.truncation_dims (768/512/256/128/64), renormalising after each slice
(rss.dense_embed.truncate -- also asserts unit norm, guarding the classic
silent MRL bug). Runs the SAME sweep on all-mpnet-base-v2 (NOT MRL-trained)
as a control: truncating nomic should barely move nDCG, truncating mpnet
should degrade visibly. Running both is what makes this an experiment
rather than an assertion.

Reuses embeddings already encoded+cached by scripts/eval_dense.py /
scripts/encode_locally.py (models/embed_cache/<model>/*.npy) -- does not
re-encode; loads the cache directly (model=None is safe here specifically
because every chunk is expected to already exist, so encode_checkpointed
never reaches the "encode a missing chunk" branch that would need a real
model).

Resumable: writes results/phase2_mrl_sweep.json after each (model, dim)
completes, and skips combinations already present in that file on re-run.

Usage:
    python scripts/eval_mrl_sweep.py [--chunk-size 500]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.dense_embed import assert_unit_norm, encode_checkpointed, get_task_prefix, truncate  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402
from rss.index import build_qdrant, delete_qdrant_index, qdrant_index_size_bytes, search  # noqa: E402
from rss.metrics import evaluate  # noqa: E402

# (dirname, is_mrl_trained)
MODELS = [
    ("nomic-embed-text-v1.5", True),
    ("all-mpnet-base-v2", False),
]

RESULTS_PATH = REPO_ROOT / "results" / "phase2_mrl_sweep.json"


def _load_cached(dirname: str, texts, kind: str, chunk_size: int):
    cache_dir = REPO_ROOT / "models" / "embed_cache" / dirname
    vectors = encode_checkpointed(None, texts, cache_dir, kind, time_budget=0, chunk_size=chunk_size)
    if vectors is None:
        raise SystemExit(
            f"{cache_dir} is missing {kind} chunks (or was written with a different --chunk-size) -- "
            "run scripts/eval_dense.py or scripts/encode_locally.py for this model first"
        )
    return vectors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--chunk-size", type=int, default=500, help="must match the --chunk-size the cache was written with")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]
    top_k = max(k_values)
    truncation_dims = config["dense"]["truncation_dims"]

    df = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    doc_ids = df["doc_id"].tolist()
    qids = [r["qid"] for r in records]
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}
    n_docs, n_queries = len(doc_ids), len(records)

    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"Resuming: {len(results)} (model, dim) combos already done")

    for dirname, is_mrl in MODELS:
        # Text content doesn't matter here -- only its LENGTH (for chunk-count
        # math), since every chunk is expected to already exist on disk. Still
        # apply the same prefix convention used at encode time for clarity.
        doc_prefix = get_task_prefix(dirname, "docs")
        query_prefix = get_task_prefix(dirname, "queries")
        doc_placeholder = [doc_prefix + "x"] * n_docs
        query_placeholder = [query_prefix + "x"] * n_queries

        print(f"Loading cached embeddings for {dirname}...")
        doc_vectors = _load_cached(dirname, doc_placeholder, "docs", args.chunk_size)
        query_vectors = _load_cached(dirname, query_placeholder, "queries", args.chunk_size)
        native_dim = doc_vectors.shape[1]

        for dim in truncation_dims:
            key = f"{dirname}@{dim}"
            if key in results:
                print(f"  {key}: already done, skipping")
                continue
            if dim > native_dim:
                print(f"  {key}: skipped ({dirname} is only {native_dim}-dim)")
                continue

            trunc_docs = truncate(doc_vectors, dim)
            trunc_queries = truncate(query_vectors, dim)
            assert_unit_norm(trunc_docs)
            assert_unit_norm(trunc_queries)

            index_path = str(REPO_ROOT / "models" / "qdrant" / f"mrl_{dirname}_{dim}")
            delete_qdrant_index(index_path)
            payloads = [{"doc_id": d} for d in doc_ids]

            t0 = time.time()
            index = build_qdrant(f"mrl_{dirname}_{dim}", trunc_docs, payloads, path=index_path)
            build_time = time.time() - t0
            index_size = qdrant_index_size_bytes(index_path)

            latencies = []
            run: dict[str, list[str]] = {}
            for qid, qvec in zip(qids, trunc_queries):
                q0 = time.perf_counter()
                top_ids = search(index, qvec, k=top_k)
                latencies.append((time.perf_counter() - q0) * 1000)
                run[qid] = [doc_ids[i] for i in top_ids]
            delete_qdrant_index(index_path)  # sweeping 9 combos -- don't keep 9 indices on disk

            scores = evaluate(run, gold_map, k_values)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]

            results[key] = {
                "model": dirname,
                "is_mrl_trained": is_mrl,
                "dim": dim,
                "native_dim": native_dim,
                "mean": scores["mean"],
                "index_build_time_s": build_time,
                "index_size_bytes": index_size,
                "query_latency_ms": {"p50": p50, "p95": p95},
            }
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
            print(
                f"  {key}: nDCG@10={scores['mean']['ndcg@10']:.4f}  "
                f"build={build_time:.1f}s  size={index_size / 1e6:.1f}MB  "
                f"p50={p50:.1f}ms"
            )

    print(f"\nSaved {len(results)} combos to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
