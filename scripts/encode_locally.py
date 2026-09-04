#!/usr/bin/env python3
"""Run the SLOW part of Phase 2's dense-model eval (Tasks 2.2/2.3) -- corpus
and query encoding -- on real hardware instead of the sandboxed dev VM.

That VM measured ~2.6-3 docs/sec on all-mpnet-base-v2 (4 CPUs, no GPU, ~3.8GB
RAM); encoding the full 20K-doc corpus there would take dozens of round
trips. A real Mac's CPU (and Metal/MPS acceleration, if Apple Silicon)
should be dramatically faster.

Writes to the EXACT SAME cache format scripts/eval_dense.py expects
(models/embed_cache/<model>/{docs,queries}_NNNN.npy) -- both scripts share
rss.dense_embed.encode_checkpointed. So after this finishes, re-running

    python3 scripts/eval_dense.py --model <model>

back in the dev VM picks the vectors straight up: no re-encoding, it just
builds the Qdrant index and evaluates (fast), which keeps those systems
numbers (build time, query latency) measured in the same environment as
the other methods (BM25, all-MiniLM-L6-v2) for a fair comparison.

Usage (from a real Terminal, NOT the sandboxed dev VM):
    cd ~/Documents/rag-retrieval-benchmark
    python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
    pip install -r requirements.txt
    python3 scripts/encode_locally.py --model all-mpnet-base-v2
    python3 scripts/encode_locally.py --model nomic-embed-text-v1.5

No time limit here (there's no per-call timeout on a real Terminal), but
still checkpointed per chunk -- safe to Ctrl-C or let your laptop sleep and
resume later; it picks up from whatever chunks already exist.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rss.corpus import load_frozen  # noqa: E402
from rss.dense_embed import encode_checkpointed  # noqa: E402
from rss.evalset import load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="dirname under models/hf/, e.g. all-mpnet-base-v2")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--chunk-size", type=int, default=500, help="docs per checkpoint file")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    evalset_cfg = config["evalset"]

    local_path = REPO_ROOT / "models" / "hf" / args.model
    if not local_path.exists():
        raise SystemExit(f"no local snapshot at {local_path} -- run scripts/fetch_hf_models.py first")

    print(f"Loading corpus and eval set for {args.model}...")
    df = load_frozen(str(REPO_ROOT / corpus_cfg["frozen_path"]))
    records = load_jsonl(str(REPO_ROOT / evalset_cfg["path"]))
    doc_texts = df[corpus_cfg["text_field"]].tolist()
    query_texts = [r["query"] for r in records]
    print(f"  {len(doc_texts)} documents, {len(query_texts)} eval queries")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(local_path), trust_remote_code=True)
    cache_dir = REPO_ROOT / "models" / "embed_cache" / args.model

    t0 = time.time()
    doc_vectors = encode_checkpointed(
        model, doc_texts, cache_dir, "docs", time_budget=float("inf"), chunk_size=args.chunk_size
    )
    dt = time.time() - t0
    print(f"Docs encoded in {dt:.1f}s ({len(doc_texts) / max(dt, 1e-9):.1f} docs/s)")

    t0 = time.time()
    encode_checkpointed(
        model, query_texts, cache_dir, "queries", time_budget=float("inf"), chunk_size=args.chunk_size
    )
    print(f"Queries encoded in {time.time() - t0:.1f}s")

    print(f"\nDone -- {cache_dir} now has every chunk for {args.model}.")
    print("Back in the dev VM (or here, if you'd rather), run:")
    print(f"  python3 scripts/eval_dense.py --model {args.model}")
    print("which will pick these up directly and just build the index + evaluate.")


if __name__ == "__main__":
    main()
