"""Fetch the 4 HuggingFace models Phase 2 needs, into models/hf/<name>/.

Run this on a machine/network that can actually reach huggingface.co --
huggingface.co, hf-mirror.com, storage.googleapis.com, and
cdn-lfs.huggingface.co are all unreachable from the sandboxed dev
environment this project is otherwise built in (see TASKS.md's Phase 2
notes), so this script is meant to be run manually, once, from a normal
Terminal with normal internet access -- not from that sandbox.

Usage:
    pip install huggingface_hub
    python3 scripts/fetch_hf_models.py

Run from the repo root; it writes into ./models/hf/<model>/, which is
exactly where rss.dense_embed.encode() and rss.rerank look for a local
snapshot (Path(local_dir) / model_name.split("/")[-1]).
"""
from __future__ import annotations

from pathlib import Path

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "BAAI/bge-reranker-base",
]


def main() -> None:
    from huggingface_hub import snapshot_download

    dest_root = Path("models/hf")
    dest_root.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        local_dir = dest_root / model_name.split("/")[-1]
        print(f"Fetching {model_name} -> {local_dir} ...")
        snapshot_download(repo_id=model_name, local_dir=str(local_dir))
        print(f"  done: {local_dir}")

    print("\nAll 4 models fetched. If this repo lives in a folder connected to")
    print("Claude, the download should be picked up automatically on the next check.")


if __name__ == "__main__":
    main()
