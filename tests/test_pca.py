"""Tests for the PCA-based OutputCompressor."""

import numpy as np
import pytest

from surrogate_factory.preprocessing.processor import OutputCompressor


def test_fit_transform_fixed_components():
    y_raw = np.random.RandomState(0).rand(50, 100)
    compressor = OutputCompressor(n_components=3)
    z = compressor.fit_transform(y_raw)

    assert compressor.is_fitted
    assert z.shape == (50, 3)


def test_transform_preserves_component_count():
    rng = np.random.RandomState(1)
    y_train = rng.rand(50, 100)
    y_val = rng.rand(20, 100)

    compressor = OutputCompressor(n_components=0.99)
    z_train = compressor.fit_transform(y_train)
    z_val = compressor.transform(y_val)

    assert z_val.shape == (20, z_train.shape[1])


def test_inverse_transform_shape():
    y_raw = np.random.RandomState(2).rand(50, 100)
    compressor = OutputCompressor(n_components=5)
    z = compressor.fit_transform(y_raw)
    y_reconstructed = compressor.inverse_transform(z)

    assert y_reconstructed.shape == (50, 100)


def test_transform_before_fit_raises():
    compressor = OutputCompressor(n_components=3)
    with pytest.raises(RuntimeError, match="fitted"):
        compressor.transform(np.zeros((5, 10)))


def test_inverse_transform_before_fit_raises():
    compressor = OutputCompressor(n_components=3)
    with pytest.raises(RuntimeError, match="fitted"):
        compressor.inverse_transform(np.zeros((5, 3)))
