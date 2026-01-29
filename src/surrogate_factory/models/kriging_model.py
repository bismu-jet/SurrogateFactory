"""Kriging (Gaussian Process) surrogate model implementation."""

import logging
from collections import namedtuple
from typing import List, Optional

import numpy as np
from smt.surrogate_models import KRG

logger = logging.getLogger(__name__)

KrigingParameters = namedtuple("KrigingParameters", ["poly", "corr"])


def _build_and_tune_single_kriging(
    x_train: np.ndarray,
    y_train_scalar: np.ndarray,
    x_val: np.ndarray,
    y_val_scalar: np.ndarray,
) -> KRG:
    """Train a single Kriging model with hyperparameter search over (poly, corr).

    Args:
        x_train: Training feature matrix (n_train, n_features).
        y_train_scalar: Training targets – 1-D scalar per sample.
        x_val: Validation feature matrix.
        y_val_scalar: Validation targets.

    Returns:
        The best-performing ``KRG`` model on the validation set.
    """
    trained_models: dict[KrigingParameters, KRG] = {}
    best_params: Optional[KrigingParameters] = None
    smallest_error = float("inf")

    polynomials = ["constant", "linear", "quadratic"]
    correlations = ["matern32"]

    for poly in polynomials:
        for corr in correlations:
            params = KrigingParameters(poly=poly, corr=corr)
            try:
                model = KRG(poly=poly, corr=corr, print_global=False, n_start=10, nugget=1e-8)
                model.set_training_values(x_train, y_train_scalar)
                model.train()

                y_pred_val = model.predict_values(x_val)
                error = float(np.sqrt(np.mean((y_pred_val - y_val_scalar) ** 2)))

                trained_models[params] = model
                if error < smallest_error:
                    smallest_error = error
                    best_params = params
            except (np.linalg.LinAlgError, ValueError):
                logger.debug("Kriging combination poly=%s corr=%s failed; skipping.", poly, corr)
                continue

    if best_params is None:
        logger.warning("All Kriging hyperparameter combinations failed; using fallback.")
        best_params = KrigingParameters(poly="constant", corr="squar_exp")
        fallback = KRG(poly=best_params.poly, corr=best_params.corr, print_global=False)
        fallback.set_training_values(x_train, y_train_scalar)
        fallback.train()
        trained_models[best_params] = fallback

    return trained_models[best_params]


class KrigingVectorModel:
    """Ensemble of per-component Kriging models for vector-valued output."""

    def __init__(self) -> None:
        self.models_per_timestep: List[KRG] = []
        self.num_timesteps: int = 0

    def train(
        self,
        x_train: np.ndarray,
        y_train_vector: np.ndarray,
        x_val: np.ndarray,
        y_val_vector: np.ndarray,
    ) -> None:
        """Train one Kriging model per output component.

        Args:
            x_train: Training features (n_train, n_features).
            y_train_vector: Training targets (n_train, n_components).
            x_val: Validation features.
            y_val_vector: Validation targets.
        """
        self.models_per_timestep = []
        self.num_timesteps = y_train_vector.shape[1]
        logger.info("Training %d Kriging models (one per component).", self.num_timesteps)

        for i in range(self.num_timesteps):
            tuned_model = _build_and_tune_single_kriging(
                x_train,
                y_train_vector[:, i],
                x_val,
                y_val_vector[:, i],
            )
            self.models_per_timestep.append(tuned_model)

        logger.info("Kriging training complete for %d components.", self.num_timesteps)

    def predict_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Predict the full output vector for new input samples.

        Args:
            x_sample: Input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_components).
        """
        predictions = [model.predict_values(x_sample) for model in self.models_per_timestep]
        return np.hstack(predictions)
