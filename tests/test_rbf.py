"""Tests for the RBF surrogate model."""

import numpy as np
import pytest
from smt.surrogate_models import RBF

from surrogate_factory.models.rbf_model import RBFVectorModel, _build_and_tune_single_rbf

NUM_TRAIN = 10
NUM_VAL = 4
NUM_FEATURES = 2
NUM_TIMESTEPS = 3


@pytest.fixture()
def random_data():
    rng = np.random.RandomState(42)
    return {
        "x_train": rng.rand(NUM_TRAIN, NUM_FEATURES),
        "y_train": rng.rand(NUM_TRAIN, NUM_TIMESTEPS),
        "x_val": rng.rand(NUM_VAL, NUM_FEATURES),
        "y_val": rng.rand(NUM_VAL, NUM_TIMESTEPS),
        "x_test": rng.rand(1, NUM_FEATURES),
    }


def test_vector_model_workflow(random_data):
    model = RBFVectorModel()
    model.train(
        random_data["x_train"],
        random_data["y_train"],
        random_data["x_val"],
        random_data["y_val"],
    )

    assert model.model is not None
    assert model.best_params is not None

    preds = model.predict_values(random_data["x_test"])
    assert preds.shape == (1, NUM_TIMESTEPS)


def test_build_and_tune_single():
    rng = np.random.RandomState(42)
    model = _build_and_tune_single_rbf(
        rng.rand(10, 2), rng.rand(10), rng.rand(4, 2), rng.rand(4), num_tries=10
    )
    assert isinstance(model, RBF)


def test_build_and_tune_flatline():
    """Flat-line validation data should not crash."""
    rng = np.random.RandomState(42)
    model = _build_and_tune_single_rbf(
        rng.rand(10, 2),
        rng.rand(10),
        rng.rand(4, 2),
        np.full(4, 5.0),
        num_tries=10,
    )
    assert isinstance(model, RBF)
    assert model.predict_values(rng.rand(4, 2)).shape == (4, 1)
