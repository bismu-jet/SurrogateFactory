"""Radial Basis Function surrogate model implementation."""

import logging
import warnings
from collections import namedtuple

import numpy as np
from smt.surrogate_models import RBF

logger = logging.getLogger(__name__)

RbfParameters = namedtuple("RbfParameters", ["d0", "degree"])

# RBF is trained on raw (unscaled) features, so the d0 basis width is sized for
# raw pairwise distances. A wide log-spaced range covers the useful band; the
# result is insensitive to grid density (verified: 100 vs 2000 candidates give
# the same fit), so a modest count keeps tuning cheap without losing quality.
_DEFAULT_NUM_TRIES = 100
_D0_MIN = 1e-1
_D0_MAX = 1e3
_DEGREES = [-1, 0, 1]


def _as_2d(array: np.ndarray) -> np.ndarray:
    """Return *array* shaped ``(n_samples, n_outputs)``."""
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _relative_l2_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean per-sample relative L2 error, robust to multi-output targets.

    Normalising each sample by its own norm keeps the error comparable across
    QoIs of different magnitudes; samples with zero norm fall back to the raw
    error to avoid division by zero.
    """
    y_pred, y_true = _as_2d(y_pred), _as_2d(y_true)
    total = 0.0
    for i in range(len(y_true)):
        norm = float(np.linalg.norm(y_true[i]))
        residual = float(np.linalg.norm(y_pred[i] - y_true[i]))
        total += residual / norm if norm > 0.0 else residual
    return total / len(y_true)


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Plain RMSE over all points (reported alongside the selection metric)."""
    return float(np.sqrt(np.mean((_as_2d(y_pred) - _as_2d(y_true)) ** 2)))


def _search_best_rbf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_tries: int = _DEFAULT_NUM_TRIES,
) -> tuple[RBF, RbfParameters, float, float]:
    """Grid-search ``(d0, degree)`` and return the model selected on validation.

    Selection uses the relative L2 error; the plain RMSE of the chosen model is
    also returned so callers can log both. Candidate combinations that fail
    (e.g. a singular system) are silenced and skipped.

    Returns:
        ``(model, params, relative_l2_error, rmse)`` for the best candidate.
    """
    if hasattr(y_train, "values"):
        y_train = y_train.values
    if hasattr(y_val, "values"):
        y_val = y_val.values
    if hasattr(x_train, "values"):
        x_train = x_train.values
    if hasattr(x_val, "values"):
        x_val = x_val.values

    best_model: RBF | None = None
    best_params: RbfParameters | None = None
    best_error = float("inf")
    best_rmse = float("inf")

    distances = np.geomspace(_D0_MIN, _D0_MAX, num=num_tries)
    for d0 in distances:
        for degree in _DEGREES:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = RBF(d0=d0, poly_degree=degree, print_global=False)
                    model.set_training_values(x_train, y_train)
                    model.train()
                    y_pred_val = model.predict_values(x_val)

                rel_error = _relative_l2_error(y_pred_val, y_val)
                if rel_error < best_error:
                    best_error = rel_error
                    best_rmse = _rmse(y_pred_val, y_val)
                    best_model = model
                    best_params = RbfParameters(d0=float(d0), degree=degree)
            except Exception:  # noqa: S112 -- failed candidates are expected; skip them
                continue

    if best_model is None or best_params is None:
        logger.warning("RBF tuning failed for all candidates; using default parameters.")
        best_params = RbfParameters(d0=1.0, degree=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = RBF(d0=best_params.d0, poly_degree=best_params.degree, print_global=False)
            best_model.set_training_values(x_train, y_train)
            best_model.train()
        best_error = _relative_l2_error(best_model.predict_values(x_val), y_val)
        best_rmse = _rmse(best_model.predict_values(x_val), y_val)

    return best_model, best_params, best_error, best_rmse


def _build_and_tune_single_rbf(
    x_train: np.ndarray,
    y_train_scalar: np.ndarray,
    x_val: np.ndarray,
    y_val_scalar: np.ndarray,
    num_tries: int = _DEFAULT_NUM_TRIES,
) -> RBF:
    """Thin wrapper returning only the tuned model (see :func:`_search_best_rbf`)."""
    model, _, _, _ = _search_best_rbf(x_train, y_train_scalar, x_val, y_val_scalar, num_tries)
    return model


class RBFVectorModel:
    """RBF model that predicts a full output vector via SMT's native multi-output support."""

    def __init__(self) -> None:
        self.model: RBF | None = None
        self.best_params: RbfParameters | None = None
        self._number_of_tryout_distances: int = _DEFAULT_NUM_TRIES

    def train(
        self,
        x_train: np.ndarray,
        y_train_vector: np.ndarray,
        x_val: np.ndarray,
        y_val_vector: np.ndarray,
    ) -> None:
        """Train an RBF model with hyperparameter search.

        Shares the exact tuning routine exercised by the unit tests, so the
        tested path and the production path cannot diverge.

        Args:
            x_train: Training features (n_train, n_features).
            y_train_vector: Training targets (n_train, n_components).
            x_val: Validation features.
            y_val_vector: Validation targets.
        """
        logger.info("Starting RBF tuning (output shape: %s).", y_train_vector.shape)
        model, params, rel_error, rmse = _search_best_rbf(
            x_train, y_train_vector, x_val, y_val_vector, self._number_of_tryout_distances
        )
        self.model = model
        self.best_params = params
        logger.info(
            "Best RBF: d0=%.4g, degree=%d (relative L2=%.6f, RMSE=%.6f).",
            params.d0,
            params.degree,
            rel_error,
            rmse,
        )

    def predict_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Predict the full output vector for new input samples.

        Args:
            x_sample: Input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_components).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("RBF model has not been trained yet.")
        predictions: np.ndarray = self.model.predict_values(x_sample)
        return predictions
