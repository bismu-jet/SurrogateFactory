"""Neural Network surrogate model implementation using scikit-learn MLPRegressor."""

import copy
import logging
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """MLP regressor that predicts a full output vector.

    Wraps ``sklearn.neural_network.MLPRegressor`` to provide the same
    ``.train()`` / ``.predict_values()`` interface as the Kriging and RBF
    models.

    Early stopping is driven by the **caller-supplied** validation set rather
    than sklearn's internal random split, so the NN is selected on exactly the
    same data Kriging and RBF use — making the cross-model comparison fair.

    Note:
        This model expects **scaled** inputs (e.g. via ``MinMaxScaler``).
    """

    def __init__(self) -> None:
        self.model: MLPRegressor | None = None
        self.output_dim: int = 0
        self.max_epochs: int = 500
        self.patience: int = 15
        self.hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
        self.activation: str = "relu"
        self.alpha: float = 1e-4

    def train(
        self,
        x_train: np.ndarray,
        y_train_vector: np.ndarray,
        x_val: np.ndarray,
        y_val_vector: np.ndarray,
    ) -> None:
        """Train an MLP with manual early stopping on the external validation set.

        Args:
            x_train: Scaled training features (n_train, n_features).
            y_train_vector: Training targets (n_train, n_components).
            x_val: Scaled validation features (drives early stopping).
            y_val_vector: Validation targets (drives early stopping).
        """
        self.output_dim = y_train_vector.shape[1]
        logger.info("Training Neural Network for %d output components.", self.output_dim)

        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            max_iter=1,  # one optimiser step per outer iteration (warm_start)
            warm_start=True,
            random_state=42,
            verbose=False,
        )

        best_val_rmse = float("inf")
        best_model: MLPRegressor | None = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            with warnings.catch_warnings():
                # max_iter=1 always "fails to converge"; that is expected here.
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train_vector)

            y_val_pred = model.predict(x_val)
            val_rmse = float(np.sqrt(np.mean((np.ravel(y_val_pred) - np.ravel(y_val_vector)) ** 2)))

            if val_rmse < best_val_rmse - 1e-9:
                best_val_rmse = val_rmse
                best_model = copy.deepcopy(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping at epoch %d (no val improvement).", epoch + 1)
                    break

        self.model = best_model if best_model is not None else model
        logger.info("Neural Network training complete (best val RMSE=%.6f).", best_val_rmse)

    def predict_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Predict the full output vector for new input samples.

        Args:
            x_sample: Scaled input features (n_samples, n_features).

        Returns:
            Predicted outputs (n_samples, n_components); always 2-D, even when
            there is a single latent component (sklearn would otherwise return
            a 1-D array).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Neural network has not been trained yet. Call .train() first.")
        preds: np.ndarray = self.model.predict(x_sample)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds
