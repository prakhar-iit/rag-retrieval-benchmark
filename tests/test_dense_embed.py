import numpy as np
import pytest

from rss.dense_embed import assert_unit_norm, encode, truncate


def _random_unit_vectors(n, dim, seed=13):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_truncate_slices_to_requested_dim():
    v = _random_unit_vectors(10, 768)
    out = truncate(v, 256)
    assert out.shape == (10, 256)


def test_truncate_renormalises_to_unit_norm():
    v = _random_unit_vectors(10, 768)
    out = truncate(v, 128)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_truncate_matches_manual_slice_before_renorm():
    v = _random_unit_vectors(5, 8)
    out = truncate(v, 4)
    manual = v[:, :4]
    manual_norm = manual / np.linalg.norm(manual, axis=1, keepdims=True)
    np.testing.assert_allclose(out, manual_norm, atol=1e-6)


def test_truncate_rejects_dim_larger_than_input():
    v = _random_unit_vectors(3, 64)
    with pytest.raises(ValueError):
        truncate(v, 128)


def test_truncate_handles_all_zero_row_without_nan():
    v = _random_unit_vectors(3, 8)
    v[1] = 0.0
    out = truncate(v, 4)
    assert not np.any(np.isnan(out))
    assert np.all(out[1] == 0.0)


def test_assert_unit_norm_passes_for_normalised_vectors():
    v = _random_unit_vectors(10, 32)
    assert_unit_norm(v)  # should not raise


def test_assert_unit_norm_raises_for_unnormalised_vectors():
    v = _random_unit_vectors(10, 32) * 2.0
    with pytest.raises(AssertionError):
        assert_unit_norm(v)


def test_encode_raises_when_local_snapshot_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        encode(
            "sentence-transformers/does-not-exist",
            ["a query"],
            local_dir=str(tmp_path / "no_such_dir"),
        )


from rss.dense_embed import encode_checkpointed


class _FakeModel:
    """Duck-typed stand-in for a sentence-transformers model: returns a
    deterministic vector per text (its length) so tests can check exact
    values without loading a real model."""

    def __init__(self, delay: float = 0.0, dim: int = 4):
        self.delay = delay
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        import time as _time

        if self.delay:
            _time.sleep(self.delay)
        self.calls.append(list(texts))
        return np.array([[float(len(t))] * self.dim for t in texts], dtype=np.float32)


def test_encode_checkpointed_returns_full_array_in_one_pass(tmp_path):
    model = _FakeModel()
    texts = [f"text{i}" for i in range(10)]
    out = encode_checkpointed(model, texts, tmp_path, "docs", time_budget=60, chunk_size=3)
    assert out.shape == (10, 4)
    np.testing.assert_allclose(out[:, 0], [len(t) for t in texts])


def test_encode_checkpointed_writes_one_chunk_file_per_chunk(tmp_path):
    model = _FakeModel()
    texts = [f"t{i}" for i in range(7)]
    encode_checkpointed(model, texts, tmp_path, "docs", time_budget=60, chunk_size=3)
    chunk_files = sorted(tmp_path.glob("docs_*.npy"))
    assert len(chunk_files) == 3  # ceil(7/3)


def test_encode_checkpointed_resumes_without_reencoding_done_chunks(tmp_path):
    model = _FakeModel()
    texts = [f"t{i}" for i in range(9)]
    # First pass: encode everything.
    encode_checkpointed(model, texts, tmp_path, "docs", time_budget=60, chunk_size=3)
    assert len(model.calls) == 3

    # Second pass with a FRESH model against the SAME cache dir: nothing
    # should be re-encoded, since every chunk file already exists.
    model2 = _FakeModel()
    out = encode_checkpointed(model2, texts, tmp_path, "docs", time_budget=60, chunk_size=3)
    assert len(model2.calls) == 0
    assert out.shape == (9, 4)


def test_encode_checkpointed_returns_none_when_time_budget_exhausted(tmp_path):
    model = _FakeModel(delay=0.05)
    texts = [f"t{i}" for i in range(20)]
    # time_budget=0 -> the very first chunk already exceeds it, so nothing
    # should be encoded and the function should signal "not done" via None.
    out = encode_checkpointed(model, texts, tmp_path, "docs", time_budget=0, chunk_size=3)
    assert out is None
    assert len(model.calls) == 0


def test_encode_checkpointed_partial_progress_is_resumable(tmp_path):
    # A model slow enough that only ~2 chunks fit in the budget.
    model = _FakeModel(delay=0.05)
    texts = [f"t{i}" for i in range(12)]  # 4 chunks of 3
    out = encode_checkpointed(model, texts, tmp_path, "docs", time_budget=0.12, chunk_size=3)
    assert out is None
    n_done_first_pass = len(model.calls)
    assert 0 < n_done_first_pass < 4

    # Resume with a generous budget -- should finish, and should not
    # re-encode the chunks the first pass already wrote.
    model2 = _FakeModel(delay=0.05)
    out = encode_checkpointed(model2, texts, tmp_path, "docs", time_budget=60, chunk_size=3)
    assert out.shape == (12, 4)
    assert len(model2.calls) == 4 - n_done_first_pass
