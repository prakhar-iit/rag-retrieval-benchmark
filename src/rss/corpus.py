"""Phase 0: load, sample and freeze the arXiv corpus.

One abstract = one document. The sample is frozen to disk so every method
in the study sees byte-identical inputs.

Two entry points on purpose:
  - load_and_sample(...)  pulls from the HuggingFace Hub (the primary path).
  - load_local_and_sample(...)  reads a file already on disk (csv/json/jsonl/
    parquet) with the same sampling + id logic. This exists because
    HuggingFace/arXiv/Kaggle are not reachable from every network this
    project runs on -- if load_and_sample can't connect, fetch the dataset
    from a machine that can reach huggingface.co and pass the file to
    load_local_and_sample instead. Both paths converge on the same
    doc_id scheme, so results are identical either way.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def _make_doc_id(text: str) -> str:
    """Stable id derived from content, not row position -- so ids survive a
    re-download, a re-order, or a switch between the HF and local-file paths."""
    return "arxiv_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# arXiv abstracts occasionally are not abstracts at all -- withdrawal/retraction
# notices where the author pulled the paper. Real example found in this corpus:
# "This preprint has been withdrawn by the author for revision". These are noise
# for both eval-question generation (nothing to ask about) and embeddings.
_DEGENERATE_ABSTRACT_RE = re.compile(
    r"withdrawn|retracted|kept secret|removed by (?:the )?author|duplicate submission",
    re.IGNORECASE,
)


def _sample_and_id(
    df: pd.DataFrame,
    text_field: str,
    sample_size: int,
    seed: int,
    min_words: int = 15,
) -> pd.DataFrame:
    # Drop pandas' own leftover index columns from CSV/parquet round-trips
    # (some HF dataset exports carry these) -- they're not part of the schema.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")

    if text_field not in df.columns:
        raise ValueError(
            f"text_field={text_field!r} not found. Available columns: {list(df.columns)}. "
            "Update corpus.text_field in configs/default.yaml to match."
        )
    df = df.dropna(subset=[text_field])
    df = df[df[text_field].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=[text_field])
    df = df[~df[text_field].str.contains(_DEGENERATE_ABSTRACT_RE)]
    df = df[df[text_field].str.split().map(len) >= min_words]

    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    df = df.sort_values(text_field).reset_index(drop=True)  # order independent of source

    df["doc_id"] = df[text_field].map(_make_doc_id)
    if df["doc_id"].duplicated().any():
        raise RuntimeError("doc_id collision after hashing -- investigate before freezing.")

    cols = ["doc_id", text_field] + [c for c in df.columns if c not in ("doc_id", text_field)]
    return df[cols].reset_index(drop=True)


def load_and_sample(
    dataset: str, sample_size: int, seed: int, text_field: str = "abstract", min_words: int = 15
) -> pd.DataFrame:
    """Load `dataset` from the HuggingFace Hub, sample `sample_size` abstracts
    deterministically, return a DataFrame with a stable `doc_id` column.

    Requires network access to huggingface.co. If that fails in this
    environment, use load_local_and_sample() on a file fetched elsewhere.
    """
    from datasets import load_dataset  # deferred import: optional/network-only dependency

    hf_ds = load_dataset(dataset, split="train")
    df = hf_ds.to_pandas()
    return _sample_and_id(df, text_field=text_field, sample_size=sample_size, seed=seed, min_words=min_words)


def load_local_and_sample(
    path: str, sample_size: int, seed: int, text_field: str = "abstract", min_words: int = 15
) -> pd.DataFrame:
    """Same sampling + id logic as load_and_sample, but reading a file already
    on disk (csv, json, jsonl, or parquet) instead of hitting the Hub."""
    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
    elif p.suffix == ".json":
        df = pd.read_json(p)
    else:
        raise ValueError(f"Unrecognized file type for {path}; expected .csv/.parquet/.jsonl/.json")

    return _sample_and_id(df, text_field=text_field, sample_size=sample_size, seed=seed, min_words=min_words)


def freeze(df: pd.DataFrame, path: str) -> None:
    """Write the frozen sample. Never re-sample after this point -- every
    downstream phase (Word2Vec training, eval-set generation, indexing)
    reads this file, not the source dataset."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def load_frozen(path: str) -> pd.DataFrame:
    """Read back a previously frozen sample."""
    return pd.read_parquet(path)
