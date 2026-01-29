"""Tests for the Kriging (Gaussian Process) surrogate model."""

import numpy as np
import pytest
from smt.surrogate_models import KRG

from surrogate_factory.models.kriging_model import (
    KrigingVectorModel,
    _build_and_tune_single_kriging,
)

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
    model = KrigingVectorModel()
    model.train(
        random_data["x_train"],
        random_data["y_train"],
        random_data["x_val"],
        random_data["y_val"],
    )

    assert model.num_timesteps == NUM_TIMESTEPS
    assert len(model.models_per_timestep) == NUM_TIMESTEPS

    preds = model.predict_values(random_data["x_test"])
    assert preds.shape == (1, NUM_TIMESTEPS)


def test_build_and_tune_single():
    rng = np.random.RandomState(42)
    model = _build_and_tune_single_kriging(
        rng.rand(10, 2), rng.rand(10), rng.rand(4, 2), rng.rand(4)
    )
    assert isinstance(model, KRG)


def test_build_and_tune_flatline():
    """Validation data with zero variance should not crash."""
    rng = np.random.RandomState(42)
    model = _build_and_tune_single_kriging(
        rng.rand(10, 2),
        rng.rand(10),
        rng.rand(4, 2),
        np.full(4, 5.0),
    )
    assert isinstance(model, KRG)
    assert model.predict_values(rng.rand(4, 2)).shape == (4, 1)
