"""Neural Network surrogate model implementation using scikit-learn MLPRegressor."""

import logging
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """MLP regressor that predicts a full output vector.

    Wraps ``sklearn.neural_network.MLPRegressor`` to provide the same
    ``.train()`` / ``.predict_values()`` interface as the Kriging and RBF
    models.

    Note:
        This model expects **scaled** inputs (e.g. via ``MinMaxScaler``).
    """

    def __init__(self) -> None:
        self.model: Optional[MLPRegressor] = None
        self.max_iter: int = 500
        self.early_stopping: bool = True
        self.validation_fraction: float = 0.15

    def train(
        self,
        x_train: np.ndarray,
        y_train_vector: np.ndarray,
        x_val: np.ndarray,
        y_val_vector: np.ndarray,
    ) -> None:
        """Train an MLP regressor on scaled feature/target data.

        Args:
            x_train: Scaled training features (n_train, n_features).
            y_train_vector: Training targets (n_train, n_components).
            x_val: Scaled validation features (used only for logging).
            y_val_vector: Validation targets (used only for logging).
        """
        output_dim = y_train_vector.shape[1]
        logger.info("Training Neural Network for %d output components.", output_dim)

        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=42,
            verbose=False,
        )

        # Combine train + val so the MLP's own early-stopping split mirrors
        # the data the caller already separated.
        x_combined = np.vstack([x_train, x_val])
        y_combined = np.vstack([y_train_vector, y_val_vector])

        self.model.fit(x_combined, y_combined)
        logger.info("Neural Network training complete (iterations=%d).", self.model.n_iter_)

    def predict_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Predict the full output vector for new input samples.

        Args:
            x_sample: Scaled input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_components).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Neural network has not been trained yet. Call .train() first.")
        return self.model.predict(x_sample)
