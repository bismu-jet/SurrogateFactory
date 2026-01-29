"""Tests for the Neural Network surrogate model (sklearn MLPRegressor)."""

import numpy as np
import pytest

from surrogate_factory.models.neural_network import NeuralNetworkModel

NUM_TRAIN_SAMPLES = 20
NUM_VAL_SAMPLES = 5
NUM_FEATURES = 2
NUM_COMPONENTS = 3


@pytest.fixture()
def random_data():
    """Generate reproducible random train/val/test arrays."""
    rng = np.random.RandomState(42)
    return {
        "x_train": rng.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES),
        "y_train": rng.rand(NUM_TRAIN_SAMPLES, NUM_COMPONENTS),
        "x_val": rng.rand(NUM_VAL_SAMPLES, NUM_FEATURES),
        "y_val": rng.rand(NUM_VAL_SAMPLES, NUM_COMPONENTS),
        "x_test": rng.rand(2, NUM_FEATURES),
    }


def test_train_and_predict(random_data):
    """Full workflow: train then predict."""
    model = NeuralNetworkModel()
    model.max_iter = 10  # keep CI fast

    model.train(
        random_data["x_train"],
        random_data["y_train"],
        random_data["x_val"],
        random_data["y_val"],
    )

    assert model.model is not None
    preds = model.predict_values(random_data["x_test"])
    assert preds.shape == (2, NUM_COMPONENTS)


def test_predict_before_train_raises():
    """Calling predict before train must raise RuntimeError."""
    model = NeuralNetworkModel()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict_values(np.zeros((1, NUM_FEATURES)))
