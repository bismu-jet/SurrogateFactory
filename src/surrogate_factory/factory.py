"""Core SurrogateFactory: load data, train, evaluate, predict, and persist surrogate models."""

import json
import logging
import pickle
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluation.metrics import calculate_r2, calculate_rmse
from .evaluation.plotting import plot_comparison_timeseries
from .models.kriging_model import KrigingVectorModel
from .models.neural_network import NeuralNetworkModel
from .models.rbf_model import RBFVectorModel
from .parsers.general_parser import load_standard_format
from .preprocessing.processor import OutputCompressor, process_data_multi_model

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelType(Enum):
    """Supported surrogate model types."""

    KRIGING = "Kriging"
    RBF = "RBF"
    NN = "NeuralNetwork"


class SurrogateFactory:
    """High-level factory for training and serving surrogate models.

    Usage::

        factory = SurrogateFactory(ModelType.KRIGING)
        factory.load_data("features.csv", "targets.json")
        factory.train(test_size=0.2, val_size=0.1)
        predictions = factory.predict(new_features_df)
    """

    def __init__(self, model_type: ModelType) -> None:
        if not isinstance(model_type, ModelType):
            raise ValueError(
                "model_type must be a ModelType member (e.g. ModelType.KRIGING)."
            )

        self.model_type: ModelType = model_type
        self.trained_models_per_qoi: Dict[str, Any] = {}
        self.compressors_per_qoi: Dict[str, OutputCompressor] = {}

        self._df_features_raw: Optional[pd.DataFrame] = None
        self._qois_raw: Optional[Dict[str, Any]] = None
        self._all_runs_ids: Optional[np.ndarray] = None

        self.processed_data: Dict[str, Any] = {}
        self.qoi_names: List[str] = []
        self.feature_names: List[str] = []
        self.x_scaler: Optional[Any] = None
        self.y_scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, List[str]] = {}
        self.reference_qois: Dict[str, Any] = {}
        self.test_run_id_map: Optional[pd.Series] = None

        logger.info("SurrogateFactory initialised with model type: %s", self.model_type.value)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_data(self, features_df: pd.DataFrame, qois_dict: Dict[str, Any]) -> None:
        """Set feature and target data directly from in-memory objects.

        Args:
            features_df: DataFrame with a ``run_id`` (or ``run_number``) column.
            qois_dict: ``{run_id: {qoi_name: vector}}`` target dictionary.
        """
        if "run_number" in features_df.columns and "run_id" not in features_df.columns:
            features_df = features_df.rename(columns={"run_number": "run_id"})
        if "run_id" not in features_df.columns:
            raise KeyError("The supplied DataFrame must contain a 'run_id' column.")

        features_df["run_id"] = features_df["run_id"].astype(str)
        self._df_features_raw = features_df
        self._qois_raw = {str(k): v for k, v in qois_dict.items()}
        self._all_runs_ids = self._df_features_raw["run_id"].unique()

    def set_reference_data(self, reference_qois: Dict[str, Any]) -> None:
        """Attach reference (baseline) predictions for comparative evaluation."""
        self.reference_qois = {str(k): v for k, v in reference_qois.items()}
        logger.info("Reference data loaded for %d runs.", len(reference_qois))

    def load_data(
        self,
        features_path: Union[str, Path],
        targets_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Load features, targets, and optional metadata from disk.

        Args:
            features_path: Path to the ``features.csv`` file.
            targets_path: Path to the ``targets.json`` file.
            metadata_path: Optional path to a ``metadata.json`` for
                categorical feature encoding.
        """
        logger.info("Loading data.")
        features_path = Path(features_path)
        targets_path = Path(targets_path)

        self.metadata = {}
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                logger.info("Loading metadata from %s.", metadata_path)
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    self.metadata = json.load(fh)
            else:
                logger.warning("Metadata file not found at '%s'.", metadata_path)

        self._df_features_raw, self._qois_raw = load_standard_format(
            features_path=features_path,
            targets_path=targets_path,
        )
        self._all_runs_ids = self._df_features_raw["run_id"].astype(str).unique()
        logger.info("Data loaded successfully. Found %d unique runs.", len(self._all_runs_ids))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Split data, preprocess, and train surrogate model(s).

        Args:
            test_size: Fraction of the dataset reserved for testing.
            val_size: Fraction of the *total* dataset reserved for validation.
            random_state: Random seed for reproducibility.
        """
        if self._all_runs_ids is None:
            raise RuntimeError("No data loaded. Call load_data() or set_data() first.")

        logger.info(
            "Splitting %d runs (test=%.2f, val=%.2f).",
            len(self._all_runs_ids),
            test_size,
            val_size,
        )
        train_val_ids, test_ids = train_test_split(
            self._all_runs_ids, test_size=test_size, random_state=random_state
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=relative_val_size, random_state=random_state
        )
        logger.info(
            "Split: %d train, %d validation, %d test.",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

        df_train = self._df_features_raw[self._df_features_raw["run_id"].isin(train_ids)]
        df_val = self._df_features_raw[self._df_features_raw["run_id"].isin(val_ids)]
        df_test = self._df_features_raw[self._df_features_raw["run_id"].isin(test_ids)]

        qois_train = {rid: d for rid, d in self._qois_raw.items() if rid in train_ids}
        qois_val = {rid: d for rid, d in self._qois_raw.items() if rid in val_ids}
        qois_test = {rid: d for rid, d in self._qois_raw.items() if rid in test_ids}

        logger.info("Preprocessing data.")
        self.processed_data = process_data_multi_model(
            df_train_features=df_train,
            qois_train=qois_train,
            df_val_features=df_val,
            qois_val=qois_val,
            df_test_features=df_test,
            qois_test=qois_test,
            metadata=self.metadata,
        )

        self.x_scaler = self.processed_data["x_scaler"]
        self.y_scalers = self.processed_data["y_scalers"]
        self.qoi_names = self.processed_data["qoi_names"]
        self.feature_names = self.processed_data["feature_names"]
        self.test_run_id_map = pd.Series(
            df_test["run_id"].values, index=range(len(df_test))
        )

        logger.info("Found %d QoI(s): %s", len(self.qoi_names), self.qoi_names)

        self.trained_models_per_qoi = {}
        self.compressors_per_qoi = {}

        for qoi_name in self.qoi_names:
            y_train_qoi = self.processed_data["y_train_per_qoi"][qoi_name]
            y_val_qoi = self.processed_data["y_val_per_qoi"][qoi_name]

            compressor = OutputCompressor(n_components=0.999)
            z_train = compressor.fit_transform(y_train_qoi)
            z_val = compressor.transform(y_val_qoi)
            self.compressors_per_qoi[qoi_name] = compressor

            logger.info("Training %s model for QoI: %s", self.model_type.value, qoi_name)
            try:
                model = self._create_and_train_model(z_train, z_val)
                self.trained_models_per_qoi[qoi_name] = model
            except Exception:
                logger.exception("Failed to train %s for %s; skipping.", self.model_type.value, qoi_name)

        logger.info("Training complete.")

    def _create_and_train_model(
        self,
        z_train: np.ndarray,
        z_val: np.ndarray,
    ) -> Any:
        """Instantiate and train the appropriate model type."""
        if self.model_type == ModelType.KRIGING:
            model = KrigingVectorModel()
            model.train(
                self.processed_data["X_train_raw"], z_train,
                self.processed_data["X_val_raw"], z_val,
            )
        elif self.model_type == ModelType.RBF:
            model = RBFVectorModel()
            model.train(
                self.processed_data["X_train_raw"], z_train,
                self.processed_data["X_val_raw"], z_val,
            )
        elif self.model_type == ModelType.NN:
            model = NeuralNetworkModel()
            model.train(
                self.processed_data["X_train_scaled"], z_train,
                self.processed_data["X_val_scaled"], z_val,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_and_plot(self, plot_prefix: str = "surrogate_comparison") -> None:
        """Evaluate on the held-out test set and generate comparison plots."""
        x_test = (
            self.processed_data["X_test_scaled"]
            if self.model_type == ModelType.NN
            else self.processed_data["X_test_raw"]
        )

        if len(x_test) == 0:
            logger.warning("Test data is empty; skipping evaluation.")
            return

        global_metrics: Dict[str, Dict[str, list]] = {
            qoi: {"rmse": [], "r2": [], "rmse_ref": [], "r2_ref": []}
            for qoi in self.qoi_names
        }
        has_ref = bool(self.reference_qois)

        for idx in range(len(x_test)):
            run_id = self.test_run_id_map.get(idx, str(idx))
            x_sample = x_test[[idx]]

            for qoi_name in self.qoi_names:
                if qoi_name not in self.trained_models_per_qoi:
                    continue

                model = self.trained_models_per_qoi[qoi_name]
                compressor = self.compressors_per_qoi[qoi_name]

                y_true = self.processed_data["y_test_per_qoi"][qoi_name][idx]
                z_pred = model.predict_values(x_sample)
                y_pred = compressor.inverse_transform(z_pred)

                y_true_flat = y_true.flatten()
                y_pred_flat = y_pred.flatten()

                mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
                y_true_clean = y_true_flat[mask]
                y_pred_clean = y_pred_flat[mask]

                y_pred_ref: Optional[np.ndarray] = None
                if has_ref:
                    ref_data = self.reference_qois.get(str(run_id), {})
                    ref_raw = ref_data.get(qoi_name)
                    if ref_raw is not None:
                        y_pred_ref = np.array(ref_raw).flatten()
                        min_len = min(len(y_true_flat), len(y_pred_ref))
                        y_pred_ref = y_pred_ref[:min_len]

                if len(y_true_clean) == 0:
                    logger.warning("Run %s | %s: all values are NaN/Inf.", run_id, qoi_name)
                    continue

                try:
                    rmse = calculate_rmse(y_true_clean, y_pred_clean)
                    r2 = (
                        1.0 if rmse < 1e-5 else 0.0
                    ) if np.var(y_true_clean) < 1e-9 else calculate_r2(y_true_clean, y_pred_clean)
                    global_metrics[qoi_name]["rmse"].append(rmse)
                    global_metrics[qoi_name]["r2"].append(r2)
                except Exception:
                    logger.exception("Metric computation failed for run %s.", run_id)

                if y_pred_ref is not None:
                    y_true_ref = y_true_flat[: len(y_pred_ref)]
                    mask_ref = np.isfinite(y_true_ref) & np.isfinite(y_pred_ref)
                    if np.any(mask_ref):
                        global_metrics[qoi_name]["rmse_ref"].append(
                            calculate_rmse(y_true_ref[mask_ref], y_pred_ref[mask_ref])
                        )
                        global_metrics[qoi_name]["r2_ref"].append(
                            calculate_r2(y_true_ref[mask_ref], y_pred_ref[mask_ref])
                        )

                plot_comparison_timeseries(
                    y_true=y_true_flat,
                    y_pred_new=y_pred_flat,
                    y_pred_old=y_pred_ref,
                    run_number=idx,
                    qoi_name=f"{qoi_name} (Run ID: {run_id})",
                    save_path_prefix=plot_prefix,
                    new_model_label=f"{self.model_type.value} Prediction",
                )

        self._log_evaluation_summary(global_metrics)

    @staticmethod
    def _log_evaluation_summary(global_metrics: Dict[str, Dict[str, list]]) -> None:
        header = f"{'QoI':<40} | {'RMSE mean (std)':<20} | {'R² mean (std)':<20} | Ref R²"
        logger.info("\n%s\n%s", header, "-" * len(header))
        for qoi_name, m in global_metrics.items():
            rmse_str = f"{np.mean(m['rmse']):.4f} (+/-{np.std(m['rmse']):.4f})" if m["rmse"] else "N/A"
            r2_str = f"{np.mean(m['r2']):.4f} (+/-{np.std(m['r2']):.4f})" if m["r2"] else "N/A"
            ref_str = f"{np.mean(m['r2_ref']):.4f}" if m["r2_ref"] else "N/A"
            logger.info("%s | %s | %s | %s", qoi_name.ljust(40), rmse_str.ljust(20), r2_str.ljust(20), ref_str)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _preprocess_input_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Encode a raw features DataFrame using the learned metadata and ordering."""
        try:
            features_df_ordered = features_df[self.feature_names]
        except KeyError as exc:
            raise ValueError(
                f"Input DataFrame is missing required columns. Expected: {self.feature_names}"
            ) from exc

        categorical_cols = list(self.metadata.keys())
        rows: list[list[float]] = []

        for row in features_df_ordered.itertuples(index=False):
            values: list[float] = []
            for i, value in enumerate(row):
                col_name = self.feature_names[i]
                if col_name in categorical_cols:
                    try:
                        values.append(float(self.metadata[col_name].index(value)))
                    except ValueError:
                        logger.warning(
                            "Unknown categorical value '%s' for '%s'; encoding as -1.",
                            value,
                            col_name,
                        )
                        values.append(-1.0)
                else:
                    values.append(float(value))
            rows.append(values)

        return np.array(rows)

    def predict(self, x_new: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions for new input samples.

        Args:
            x_new: DataFrame whose columns match the training features.

        Returns:
            ``{qoi_name: np.ndarray}`` with shape ``(n_samples, n_timesteps)``.
        """
        if not self.trained_models_per_qoi:
            raise RuntimeError("No trained models. Call .train() or .load_model() first.")
        if self.x_scaler is None or not self.y_scalers:
            raise RuntimeError("Scalers not found; the factory is not fully trained.")

        x_raw = self._preprocess_input_features(x_new)
        x_input = self.x_scaler.transform(x_raw) if self.model_type == ModelType.NN else x_raw

        predictions: Dict[str, np.ndarray] = {}
        for qoi_name, model in self.trained_models_per_qoi.items():
            z_pred = model.predict_values(x_input)
            compressor = self.compressors_per_qoi.get(qoi_name)
            if compressor is None:
                logger.warning("No compressor found for QoI '%s'; skipping.", qoi_name)
                continue
            predictions[qoi_name] = compressor.inverse_transform(z_pred)

        logger.info("Prediction complete for %d QoI(s).", len(predictions))
        return predictions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, file_path: Union[str, Path]) -> None:
        """Serialise the trained factory to disk via pickle."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Model saved to %s.", file_path)

    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> "SurrogateFactory":
        """Deserialise a previously saved factory.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the loaded object is not a ``SurrogateFactory``.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        with open(file_path, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}.")
        logger.info("Model loaded from %s (type: %s).", file_path, obj.model_type.value)
        return obj
