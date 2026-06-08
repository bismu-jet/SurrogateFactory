"""Tests for the physical-unit regression metrics."""

import numpy as np
import pytest

from surrogate_factory.evaluation.metrics import (
    calculate_mae,
    calculate_max_abs_error,
    calculate_nrmse,
    calculate_rmse,
)


def test_mae():
    assert calculate_mae([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2.0 / 3.0)


def test_max_abs_error():
    assert calculate_max_abs_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 7.0])) == 4.0


def test_nrmse_normalises_by_range():
    y_true = np.array([0.0, 10.0])
    # RMSE = 1, range = 10 => NRMSE = 0.1.
    assert calculate_nrmse(y_true, y_true + 1.0) == pytest.approx(0.1)


def test_nrmse_constant_target_returns_zero():
    y_true = np.array([5.0, 5.0, 5.0])
    assert calculate_nrmse(y_true, y_true + 3.0) == 0.0


def test_rmse_matches_manual():
    assert calculate_rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
        np.sqrt((9.0 + 16.0) / 2.0)
    )
