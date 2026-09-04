"""Fetch the HuggingFace models Phase 2 needs, into models/hf/<name>/.

Run this on a machine/network that can actually reach huggingface.co --
huggingface.co, hf-mirror.com, storage.googleapis.com, and
cdn-lfs.huggingface.co are all unreachable from the sandboxed dev
environment this project is otherwise built in (see TASKS.md's Phase 2
notes), so this script is meant to be run manually from a normal Terminal
with normal internet access -- not from that sandbox.

Usage:
    pip install huggingface_hub
    python3 scripts/fetch_hf_models.py

Run from the repo root; it writes into ./models/hf/<name>/, which is
exactly where rss.dense_embed.encode(), rss.rerank, and CrossEncoder/
SentenceTransformer loads elsewhere in this repo look for a local snapshot.

Only weight/config/tokenizer files are pulled (`allow_patterns` below) --
NOT the onnx/openvino/tf/rust duplicate weight formats every HF repo also
ships, which is what made the first version of this script pull ~10GB for
~2GB of actually-needed files. Safe to re-run: snapshot_download skips
files that already match on disk.
"""
from __future__ import annotations

from pathlib import Path

# (repo_id, local dirname). nomic-bert-2048 isn't a model you load directly --
# nomic-embed-text-v1.5/config.json's auto_map points AT it for its custom
# architecture code (NomicBertModel), so sentence-transformers/transformers
# fetches it at load time unless it's already present locally. Same
# local_dir convention as the others so rss.dense_embed.encode() need not
# special-case it.
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text-v1.5"),
    ("nomic-ai/nomic-bert-2048", "nomic-bert-2048"),
    ("BAAI/bge-reranker-base", "bge-reranker-base"),
]

# Weights + everything needed to load and run the model; excludes the
# onnx/, openvino/, *.h5 (tf), *.ot (rust) duplicate formats every repo
# ships alongside model.safetensors.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
]


def main() -> None:
    from huggingface_hub import snapshot_download

    dest_root = Path("models/hf")
    dest_root.mkdir(parents=True, exist_ok=True)

    for model_name, dirname in MODELS:
        local_dir = dest_root / dirname
        print(f"Fetching {model_name} -> {local_dir} ...")
        snapshot_download(
            repo_id=model_name, local_dir=str(local_dir), allow_patterns=ALLOW_PATTERNS
        )
        print(f"  done: {local_dir}")

    print("\nAll models fetched. If this repo lives in a folder connected to")
    print("Claude, the download should be picked up automatically on the next check.")


if __name__ == "__main__":
    main()
