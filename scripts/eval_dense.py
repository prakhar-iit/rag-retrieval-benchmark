#!/usr/bin/env python3
"""Phase 2, tasks 2.2/2.3: dense sentence-transformer models on the eval set.

Encoding 20K abstracts on this machine's CPU is slow enough to exceed a
single shell call's time budget, so encoding is checkpointed to disk in
small chunks (models/embed_cache/<model>/docs_NNNN.npy) rather than done in
one `model.encode()` call -- run this script repeatedly; it resumes from
wherever the previous run left off (whatever chunk files already exist) and
only builds the index + evaluates once every chunk for both docs and
queries is present. This is why it doesn't just call
rss.dense_embed.encode() (which caches the WHOLE encode, all-or-nothing --
fine for a fast call, not for a multi-call slow one).

Usage:
    python scripts/eval_dense.py --model all-MiniLM-L6-v2 [--time-budget 150] [--chunk-size 200]

`--model` is the local directory name under models/hf/ (see
scripts/fetch_hf_models.py). Writes: results/phase2_dense_<model>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.dense_embed import encode_checkpointed, get_task_prefix  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402
from rss.index import build_qdrant, delete_qdrant_index, qdrant_index_size_bytes, search  # noqa: E402
from rss.metrics import evaluate  # noqa: E402

# Small enough that even a slow model's single chunk finishes well within
# one call's time budget -- the risk being sized against is a chunk that's
# STILL encoding when the call gets cut off, which would waste that chunk's
# work entirely (nothing is saved until model.encode() returns).
CHUNK_SIZE = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="dirname under models/hf/, e.g. all-MiniLM-L6-v2")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--time-budget", type=float, default=150.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="docs per checkpoint -- lower for slower models so a killed call loses less work",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]
    k_values = config["metrics"]["k_values"]
    top_k = max(k_values)

    local_path = REPO_ROOT / "models" / "hf" / args.model
    if not local_path.exists():
        raise SystemExit(f"no local snapshot at {local_path} -- run scripts/fetch_hf_models.py first")

    print(f"Loading arXiv corpus and eval set for {args.model}...")
    df = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    doc_ids = df["doc_id"].tolist()
    doc_prefix = get_task_prefix(args.model, "docs")
    query_prefix = get_task_prefix(args.model, "queries")
    doc_texts = [doc_prefix + t for t in df[corpus_cfg["text_field"]]]
    qids = [r["qid"] for r in records]
    query_texts = [query_prefix + r["query"] for r in records]
    if doc_prefix or query_prefix:
        print(f"  applying task prefixes: docs={doc_prefix!r} queries={query_prefix!r}")
    gold_map = {r["qid"]: r["gold_doc_id"] for r in records}
    print(f"  {len(df)} documents, {len(records)} eval queries")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(local_path), trust_remote_code=True)

    cache_dir = REPO_ROOT / "models" / "embed_cache" / args.model

    t0 = time.time()
    doc_vectors = encode_checkpointed(
        model, doc_texts, cache_dir, "docs", time_budget=args.time_budget, chunk_size=args.chunk_size
    )
    if doc_vectors is None:
        print("  [docs] time budget reached -- re-run this script to continue")
        return
    doc_encode_time = time.time() - t0

    remaining_budget = max(args.time_budget - doc_encode_time, 10.0)
    t0 = time.time()
    query_vectors = encode_checkpointed(
        model, query_texts, cache_dir, "queries", time_budget=remaining_budget, chunk_size=args.chunk_size
    )
    if query_vectors is None:
        print("  [queries] time budget reached -- re-run this script to continue")
        return
    query_encode_time = time.time() - t0

    print(
        f"Encoded {len(doc_vectors)} docs in {doc_encode_time:.1f}s "
        f"({len(doc_vectors) / max(doc_encode_time, 1e-9):.1f} docs/s), "
        f"{len(query_vectors)} queries in {query_encode_time:.1f}s"
    )

    index_path = str(REPO_ROOT / "models" / "qdrant" / f"dense_{args.model}")
    delete_qdrant_index(index_path)  # clean rebuild -- see build_qdrant's rebuild-at-same-path note
    payloads = [{"doc_id": d} for d in doc_ids]

    print("Building Qdrant index...")
    t0 = time.time()
    index = build_qdrant(f"dense_{args.model}", doc_vectors, payloads, path=index_path)
    build_time = time.time() - t0
    index_size = qdrant_index_size_bytes(index_path)
    print(f"  built in {build_time:.2f}s, {index_size / 1e6:.1f}MB on disk")

    print("Querying...")
    latencies = []
    run: dict[str, list[str]] = {}
    for qid, qvec in zip(qids, query_vectors):
        q0 = time.perf_counter()
        top_ids = search(index, qvec, k=top_k)
        latencies.append((time.perf_counter() - q0) * 1000)
        run[qid] = [doc_ids[i] for i in top_ids]

    scores = evaluate(run, gold_map, k_values)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    print(
        f"\n{args.model}: nDCG@10={scores['mean']['ndcg@10']:.4f}  "
        f"Recall@10={scores['mean']['recall@10']:.4f}  MRR={scores['mean']['mrr']:.4f}"
    )
    print(f"Query latency: p50={p50:.2f}ms  p95={p95:.2f}ms")

    out = {
        "model": args.model,
        "mean": scores["mean"],
        "n_queries": scores["n_queries"],
        "doc_encode_time_s": doc_encode_time,
        "doc_encode_docs_per_s": len(doc_vectors) / max(doc_encode_time, 1e-9),
        "index_build_time_s": build_time,
        "index_size_bytes": index_size,
        "query_latency_ms": {"p50": p50, "p95": p95},
        "vector_dim": int(doc_vectors.shape[1]),
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase2_dense_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
