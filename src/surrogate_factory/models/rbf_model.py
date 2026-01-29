"""Radial Basis Function surrogate model implementation."""

import logging
from collections import namedtuple
from typing import Optional

import numpy as np
from smt.surrogate_models import RBF

logger = logging.getLogger(__name__)

RbfParameters = namedtuple("RbfParameters", ["d0", "degree"])


def _build_and_tune_single_rbf(
    x_train: np.ndarray,
    y_train_scalar: np.ndarray,
    x_val: np.ndarray,
    y_val_scalar: np.ndarray,
    num_tries: int = 100,
) -> RBF:
    """Train a single RBF model with hyperparameter search over (d0, degree).

    Args:
        x_train: Training features.
        y_train_scalar: Training scalar targets.
        x_val: Validation features.
        y_val_scalar: Validation scalar targets.
        num_tries: Number of d0 values to sample logarithmically.

    Returns:
        The best-performing ``RBF`` model on the validation set.
    """
    if hasattr(y_train_scalar, "values"):
        y_train_scalar = y_train_scalar.values
    if hasattr(y_val_scalar, "values"):
        y_val_scalar = y_val_scalar.values
    if hasattr(x_train, "values"):
        x_train = x_train.values
    if hasattr(x_val, "values"):
        x_val = x_val.values

    trained_models: dict[RbfParameters, RBF] = {}
    best_params: Optional[RbfParameters] = None
    smallest_error = float("inf")

    distances = np.geomspace(1, 100_000.0, num=num_tries)
    degrees = [-1, 0, 1]

    for d in distances:
        for degree in degrees:
            params = RbfParameters(d0=d, degree=degree)
            try:
                model = RBF(print_global=False, d0=d, poly_degree=degree)
                model.set_training_values(x_train, y_train_scalar)
                model.train()

                y_pred_val = model.predict_values(x_val)
                error = float(np.sqrt(np.mean((y_pred_val - y_val_scalar) ** 2)))

                trained_models[params] = model
                if error < smallest_error:
                    smallest_error = error
                    best_params = params
            except np.linalg.LinAlgError:
                continue

    if best_params is None:
        logger.warning("RBF tuning failed; using default parameters.")
        best_params = RbfParameters(d0=1.0, degree=0)
        fallback = RBF(print_global=False, d0=best_params.d0, poly_degree=best_params.degree)
        fallback.set_training_values(x_train, y_train_scalar)
        fallback.train()
        trained_models[best_params] = fallback

    return trained_models[best_params]


class RBFVectorModel:
    """RBF model that predicts a full output vector via SMT's native multi-output support."""

    def __init__(self) -> None:
        self.model: Optional[RBF] = None
        self.best_params: Optional[RbfParameters] = None
        self._number_of_tryout_distances: int = 2000

    def train(
        self,
        x_train: np.ndarray,
        y_train_vector: np.ndarray,
        x_val: np.ndarray,
        y_val_vector: np.ndarray,
    ) -> None:
        """Train an RBF model with hyperparameter search.

        Args:
            x_train: Training features (n_train, n_features).
            y_train_vector: Training targets (n_train, n_components).
            x_val: Validation features.
            y_val_vector: Validation targets.
        """
        logger.info("Starting RBF tuning (output shape: %s).", y_train_vector.shape)

        trained_models: dict[RbfParameters, RBF] = {}
        distances = np.geomspace(1, 1000, num=self._number_of_tryout_distances)
        degrees = [-1, 0, 1]

        best_error = float("inf")
        best_id: Optional[RbfParameters] = None

        for d in distances:
            for degree in degrees:
                params_id = RbfParameters(d0=d, degree=degree)
                try:
                    temp_model = RBF(d0=d, poly_degree=degree, print_global=False)
                    temp_model.set_training_values(x_train, y_train_vector)
                    temp_model.train()

                    y_pred_val = temp_model.predict_values(x_val)

                    total_error = 0.0
                    for i in range(len(y_val_vector)):
                        abs_norm = float(np.linalg.norm(y_pred_val[i] - y_val_vector[i]))
                        norm = float(np.linalg.norm(y_val_vector[i]))
                        total_error += abs_norm / norm if norm > 0.0 else abs_norm

                    relative_error = total_error / len(y_val_vector)
                    trained_models[params_id] = temp_model

                    if relative_error < best_error:
                        best_error = relative_error
                        best_id = params_id
                except Exception:
                    continue

        if best_id is None:
            logger.warning("RBF tuning failed; using default parameters.")
            best_id = RbfParameters(d0=1.0, degree=0)
            temp_model = RBF(d0=1.0, poly_degree=0, print_global=False)
            temp_model.set_training_values(x_train, y_train_vector)
            temp_model.train()
            self.model = temp_model
        else:
            self.model = trained_models[best_id]

        self.best_params = best_id
        logger.info(
            "Best RBF: d0=%.4f, degree=%d (validation error: %.6f).",
            best_id.d0,
            best_id.degree,
            best_error,
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
        return self.model.predict_values(x_sample)
