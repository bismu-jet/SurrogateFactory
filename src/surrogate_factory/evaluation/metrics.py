"""Regression performance metrics for surrogate model evaluation."""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (coefficient of determination) score.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        R² score (closer to 1.0 is better).
    """
    return float(r2_score(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value (lower is better).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def generate_performance_report(
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    n_samples: int,
    n_timesteps: int,
) -> Dict[str, float]:
    """Generate a summary report of global and per-run metrics.

    Args:
        y_true_flat: 1-D array of all ground-truth values (concatenated).
        y_pred_flat: 1-D array of all predicted values (concatenated).
        n_samples: Number of simulation runs.
        n_timesteps: Number of time-steps per run.

    Returns:
        Dictionary with ``rmse_global``, ``r2_global``, ``rmse_mean_run``,
        and ``rmse_std_run``.
    """
    rmse_global = calculate_rmse(y_true_flat, y_pred_flat)
    r2_global = calculate_r2(y_true_flat, y_pred_flat)

    y_true_runs = y_true_flat.reshape(n_samples, n_timesteps)
    y_pred_runs = y_pred_flat.reshape(n_samples, n_timesteps)

    run_rmses: list[float] = []
    for i in range(n_samples):
        run_rmses.append(calculate_rmse(y_true_runs[i], y_pred_runs[i]))

    return {
        "rmse_global": rmse_global,
        "r2_global": r2_global,
        "rmse_mean_run": float(np.mean(run_rmses)),
        "rmse_std_run": float(np.std(run_rmses)),
    }
