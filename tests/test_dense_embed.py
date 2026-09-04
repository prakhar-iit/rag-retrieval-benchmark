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
