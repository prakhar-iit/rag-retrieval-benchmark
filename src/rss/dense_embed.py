"""Phase 2: dense embeddings, and Matryoshka truncation.

MRL is a TRAINING OBJECTIVE, not a compression trick: it front-loads
information so that any prefix of the vector is itself a valid embedding.
That is why truncating an MRL-trained model barely moves quality while
truncating a non-MRL model degrades badly -- running both is the experiment.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def encode(
    model_name: str,
    texts,
    batch_size: int = 64,
    local_dir: str = "models/hf",
    cache_dir: str = "models/embed_cache",
) -> np.ndarray:
    """Encode texts with a sentence-transformers model loaded from a LOCAL
    snapshot directory, not downloaded automatically -- this dev environment
    cannot reach huggingface.co. The model must already exist at
    `<local_dir>/<model_name's final path segment>` (see README/TASKS 2.2 for
    the `huggingface_hub.snapshot_download` instructions run on a machine
    with real network access).

    Caches encodings to disk keyed by (model, text count, first-few-texts
    hash) so re-running eval scripts against the same corpus doesn't
    re-encode 20K abstracts every time -- encoding is the expensive step here,
    not retrieval.
    """
    local_path = Path(local_dir) / model_name.split("/")[-1]
    if not local_path.exists():
        raise FileNotFoundError(
            f"no local snapshot at {local_path} -- fetch it first with "
            f"huggingface_hub.snapshot_download('{model_name}', local_dir='{local_path}') "
            "on a network that can reach huggingface.co, then re-run"
        )

    texts = list(texts)
    import hashlib

    fingerprint = "".join(texts[:5]) + str(len(texts))
    cache_key = hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"{local_path.name}_{len(texts)}_{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    # trust_remote_code: nomic-embed-text-v1.5 needs it (custom architecture
    # code, vendored locally by scripts/fetch_hf_models.py -- see its
    # docstring); harmless no-op for models that don't ship custom code.
    model = SentenceTransformer(str(local_path), trust_remote_code=True)
    vectors = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Slice to `dim` dimensions and RENORMALISE.

    Cosine similarity assumes unit vectors; slicing breaks the norm.
    Forgetting this is the classic MRL bug and it fails silently -- a
    truncated-but-not-renormalised vector still "works" (cosine similarity
    is scale-invariant between two vectors that are BOTH mis-normalised the
    same way), so the bug only shows up as quietly worse nDCG, never a crash.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2D (n_docs, dim) array, got shape {vectors.shape}")
    if dim > vectors.shape[1]:
        raise ValueError(f"cannot truncate to {dim} dims, vectors only have {vectors.shape[1]}")

    sliced = vectors[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # guard: an all-zero row stays all-zero, not NaN
    return (sliced / norms).astype(np.float32)


def assert_unit_norm(vectors: np.ndarray, tol: float = 1e-5) -> None:
    """Guard against the bug `truncate` exists to avoid: raises with the
    offending row indices and their norms rather than a bare assert, so a
    failure is diagnosable without re-running under a debugger."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    bad = np.abs(norms - 1.0) > tol
    if np.any(bad):
        bad_idx = np.nonzero(bad)[0][:5]
        raise AssertionError(
            f"{int(bad.sum())} of {len(norms)} vectors are not unit-norm (tol={tol}); "
            f"first offending rows {list(bad_idx)} have norms {norms[bad_idx].tolist()}"
        )


def encode_checkpointed(
    model,
    texts,
    cache_dir,
    prefix: str,
    time_budget: float = 150.0,
    chunk_size: int = 200,
    batch_size: int = 32,
):
    """Encode `texts` in `chunk_size` pieces, saving each chunk to
    `<cache_dir>/<prefix>_NNNN.npy` as soon as it's done, and stop once
    `time_budget` seconds have elapsed.

    Exists because encoding a large corpus on CPU can take longer than a
    single process invocation is given to run (e.g. a shell tool's call
    timeout) -- `model.encode(texts)` in one call either finishes or loses
    ALL of its progress if cut off, since nothing is written to disk until
    it returns. This checkpoints per-chunk instead, so re-calling with the
    same `cache_dir`/`prefix` resumes from whatever chunks already exist
    rather than re-encoding from scratch.

    `model` is any object with an `.encode(texts, batch_size=..., ...)`
    method returning an array-like of shape (len(texts), dim) -- duck-typed
    so tests can pass a fake instead of a real sentence-transformers model.

    Returns the full stacked (len(texts), dim) array once every chunk is
    present; returns None if the time budget ran out first (some chunks
    remain unencoded) -- callers should re-invoke (e.g. re-run the script)
    to continue.
    """
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    t0 = time.time()

    for i in range(n_chunks):
        chunk_path = cache_dir / f"{prefix}_{i:04d}.npy"
        if chunk_path.exists():
            continue
        if time.time() - t0 > time_budget:
            return None
        chunk_texts = texts[i * chunk_size : (i + 1) * chunk_size]
        vecs = np.asarray(
            model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        ).astype(np.float32)
        np.save(chunk_path, vecs)

    chunks = [np.load(cache_dir / f"{prefix}_{i:04d}.npy") for i in range(n_chunks)]
    return np.concatenate(chunks, axis=0)
