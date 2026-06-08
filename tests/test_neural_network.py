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
    model.max_epochs = 10  # keep CI fast

    model.train(
        random_data["x_train"],
        random_data["y_train"],
        random_data["x_val"],
        random_data["y_val"],
    )

    assert model.model is not None
    preds = model.predict_values(random_data["x_test"])
    assert preds.shape == (2, NUM_COMPONENTS)


def test_predict_single_component_is_2d():
    """A single latent component must still yield a 2-D prediction.

    sklearn's MLPRegressor returns a 1-D array for single-output targets; the
    wrapper must reshape it so the decode pipeline keeps working.
    """
    rng = np.random.RandomState(0)
    model = NeuralNetworkModel()
    model.max_epochs = 20
    model.train(
        rng.rand(15, NUM_FEATURES), rng.rand(15, 1), rng.rand(5, NUM_FEATURES), rng.rand(5, 1)
    )

    preds = model.predict_values(rng.rand(3, NUM_FEATURES))
    assert preds.shape == (3, 1)


def test_early_stopping_uses_external_validation():
    """Manual early stopping should select on the supplied validation set."""
    rng = np.random.RandomState(0)
    x_train = rng.rand(30, NUM_FEATURES)
    # Learnable linear target so the val RMSE genuinely improves over epochs.
    y_train = x_train @ np.array([[2.0], [3.0]])
    x_val = rng.rand(10, NUM_FEATURES)
    y_val = x_val @ np.array([[2.0], [3.0]])

    model = NeuralNetworkModel()
    model.max_epochs = 200
    model.patience = 10
    model.train(x_train, y_train, x_val, y_val)

    val_pred = model.predict_values(x_val)
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    # A model selected on the external val set should fit this simple target well.
    assert val_rmse < 0.5


def test_predict_before_train_raises():
    """Calling predict before train must raise RuntimeError."""
    model = NeuralNetworkModel()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict_values(np.zeros((1, NUM_FEATURES)))
