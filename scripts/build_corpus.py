#!/usr/bin/env python3
"""Phase 0, tasks 0.1 + 0.2: load the corpus, inspect it, sample, freeze.

Usage:
    python scripts/build_corpus.py                # pulls from HuggingFace Hub
    python scripts/build_corpus.py --local FILE    # reads a file already on disk
                                                    # (csv/parquet/jsonl/json)

The --local path exists because huggingface.co is not reachable from every
network this project runs on. If `python scripts/build_corpus.py` fails to
connect, fetch the dataset from a machine that *can* reach the Hub, e.g.:

    pip install datasets
    python -c "
    from datasets import load_dataset
    load_dataset('CShorten/ML-ArXiv-Papers', split='train').to_parquet('ml_arxiv_papers.parquet')
    "

...then copy ml_arxiv_papers.parquet into this repo (outside data/, which is
gitignored but fine to use) and run:

    python scripts/build_corpus.py --local ml_arxiv_papers.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rss.corpus import freeze, load_and_sample, load_local_and_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--local", default=None, help="Path to a local corpus file instead of the HF Hub")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    corpus_cfg = config["corpus"]
    seed = config["seed"]

    if args.local:
        print(f"Loading local file: {args.local}")
        df = load_local_and_sample(
            args.local,
            sample_size=corpus_cfg["sample_size"],
            seed=seed,
            text_field=corpus_cfg["text_field"],
        )
    else:
        print(f"Loading from HuggingFace Hub: {corpus_cfg['dataset']}")
        try:
            df = load_and_sample(
                corpus_cfg["dataset"],
                sample_size=corpus_cfg["sample_size"],
                seed=seed,
                text_field=corpus_cfg["text_field"],
            )
        except Exception as e:
            print(f"\nCould not reach the HuggingFace Hub ({e!r}).")
            print("See the --local option in this script's docstring for a workaround.")
            raise SystemExit(1)

    print(f"\nSampled {len(df)} documents.")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    text_field = corpus_cfg["text_field"]
    lengths = df[text_field].str.split().map(len)
    print(f"\n{text_field} length (words): min={lengths.min()} "
          f"median={lengths.median():.0f} mean={lengths.mean():.1f} max={lengths.max()}")

    out_path = corpus_cfg["frozen_path"]
    freeze(df, out_path)
    print(f"\nFroze {len(df)} documents to {out_path}")


if __name__ == "__main__":
    main()
