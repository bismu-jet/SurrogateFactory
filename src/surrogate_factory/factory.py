"""Core SurrogateFactory: load data, train, evaluate, predict, and persist surrogate models."""

import json
import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .evaluation.metrics import (
    calculate_mae,
    calculate_max_abs_error,
    calculate_r2,
    calculate_rmse,
)
from .evaluation.plotting import plot_comparison_timeseries
from .models.kriging_model import KrigingVectorModel
from .models.neural_network import NeuralNetworkModel
from .models.rbf_model import RBFVectorModel
from .parsers.general_parser import load_standard_format
from .preprocessing.processor import OutputCompressor, base_caption, process_data_multi_model

logger = logging.getLogger(__name__)


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
            raise ValueError("model_type must be a ModelType member (e.g. ModelType.KRIGING).")

        self.model_type: ModelType = model_type
        self.trained_models_per_qoi: dict[str, Any] = {}
        self.compressors_per_qoi: dict[str, OutputCompressor] = {}
        self.failed_qois: list[str] = []
        # PCA variance to retain when compressing each QoI's time-series.
        self.n_pca_components: float | int = 0.999

        self._df_features_raw: pd.DataFrame | None = None
        self._qois_raw: dict[str, Any] | None = None
        self._all_runs_ids: np.ndarray | None = None

        self.processed_data: dict[str, Any] = {}
        self.qoi_names: list[str] = []
        self.feature_names: list[str] = []
        self.x_scaler: Any | None = None
        self.y_scalers: dict[str, Any] = {}
        self.metadata: dict[str, list[str]] = {}
        self.reference_qois: dict[str, Any] = {}
        self.test_run_id_map: pd.Series | None = None

        logger.info("SurrogateFactory initialised with model type: %s", self.model_type.value)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_data(self, features_df: pd.DataFrame, qois_dict: dict[Any, Any]) -> None:
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

    def set_reference_data(self, reference_qois: dict[str, Any]) -> None:
        """Attach reference (baseline) predictions for comparative evaluation."""
        self.reference_qois = {str(k): v for k, v in reference_qois.items()}
        logger.info("Reference data loaded for %d runs.", len(reference_qois))

    def load_data(
        self,
        features_path: str | Path,
        targets_path: str | Path,
        metadata_path: str | Path | None = None,
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
                with open(metadata_path, encoding="utf-8") as fh:
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
        self._fit_on_ids(train_ids, val_ids, test_ids)

    def _subset(self, run_ids: np.ndarray) -> "tuple[pd.DataFrame, dict[str, Any]]":
        """Return the feature rows and QoI entries for *run_ids*."""
        if self._df_features_raw is None or self._qois_raw is None:
            raise RuntimeError("No data loaded. Call load_data() or set_data() first.")
        run_id_set = set(run_ids)
        df = self._df_features_raw[self._df_features_raw["run_id"].isin(run_id_set)]
        qois = {rid: d for rid, d in self._qois_raw.items() if rid in run_id_set}
        return df, qois

    def _fit_on_ids(
        self,
        train_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray,
    ) -> None:
        """Preprocess the given split and train one model per QoI.

        Shared by :meth:`train` and :meth:`cross_validate` so both follow the
        identical preprocessing and training path.
        """
        logger.info(
            "Split: %d train, %d validation, %d test.",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        df_train, qois_train = self._subset(train_ids)
        df_val, qois_val = self._subset(val_ids)
        df_test, qois_test = self._subset(test_ids)

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
        test_run_ids = self.processed_data["test_run_ids"]
        self.test_run_id_map = pd.Series(test_run_ids, index=range(len(test_run_ids)))

        logger.info("Found %d QoI(s): %s", len(self.qoi_names), self.qoi_names)

        self.trained_models_per_qoi = {}
        self.compressors_per_qoi = {}
        self.failed_qois = []

        for qoi_name in self.qoi_names:
            logger.info("Training %s model for QoI: %s", self.model_type.value, qoi_name)
            try:
                y_train_qoi = self.processed_data["y_train_per_qoi"][qoi_name]
                y_val_qoi = self.processed_data["y_val_per_qoi"][qoi_name]

                compressor = OutputCompressor(n_components=self.n_pca_components)
                z_train = compressor.fit_transform(y_train_qoi)
                z_val = compressor.transform(y_val_qoi)

                model = self._create_and_train_model(z_train, z_val)
                self.compressors_per_qoi[qoi_name] = compressor
                self.trained_models_per_qoi[qoi_name] = model
            except Exception:
                logger.exception(
                    "Failed to train %s for QoI '%s'.", self.model_type.value, qoi_name
                )
                self.failed_qois.append(qoi_name)

        if self.failed_qois:
            logger.warning(
                "%d/%d QoI(s) failed to train and were skipped: %s",
                len(self.failed_qois),
                len(self.qoi_names),
                self.failed_qois,
            )
        if not self.trained_models_per_qoi:
            raise RuntimeError("Training produced no usable models; every QoI failed.")

        logger.info(
            "Training complete (%d/%d QoIs trained).",
            len(self.trained_models_per_qoi),
            len(self.qoi_names),
        )

    def cross_validate(
        self,
        k: int = 5,
        val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> dict[str, dict[str, float]]:
        """Estimate generalisation via K-fold cross-validation over runs.

        A single 70/10/20 split has high variance on the small datasets typical
        here.  Each fold is held out as the test set while the rest is split
        into train/validation (reusing the standard pipeline), and physical
        metrics are aggregated across folds.

        The factory is left fitted on the **last** fold; call :meth:`train`
        afterwards to obtain the deployable model fitted on the chosen split.

        Args:
            k: Number of folds.
            val_fraction: Fraction of each fold's train+val portion used for
                validation/hyperparameter selection.
            random_state: Seed for the fold shuffling.

        Returns:
            ``{qoi_name: {metric: value}}`` with cross-fold mean/std of the
            per-fold metrics.
        """
        if self._all_runs_ids is None:
            raise RuntimeError("No data loaded. Call load_data() or set_data() first.")

        run_ids = np.asarray(self._all_runs_ids)
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

        per_qoi_fold_metrics: dict[str, dict[str, list[float]]] = {}
        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(run_ids), start=1):
            logger.info("Cross-validation fold %d/%d.", fold, k)
            test_ids = run_ids[test_idx]
            train_ids, val_ids = train_test_split(
                run_ids[train_val_idx], test_size=val_fraction, random_state=random_state
            )
            try:
                self._fit_on_ids(train_ids, val_ids, test_ids)
                fold_summary = self.evaluate_and_plot(make_plots=False)
            except Exception:
                logger.exception("Fold %d failed; skipping.", fold)
                continue

            for qoi_name, metrics in fold_summary.items():
                bucket = per_qoi_fold_metrics.setdefault(qoi_name, {})
                for metric_name, value in metrics.items():
                    bucket.setdefault(metric_name, []).append(value)

        cv_summary: dict[str, dict[str, float]] = {}
        for qoi_name, fold_metrics in per_qoi_fold_metrics.items():
            entry: dict[str, float] = {"n_folds": float(len(next(iter(fold_metrics.values()))))}
            for metric_name, values in fold_metrics.items():
                entry[f"{metric_name}_mean"] = float(np.mean(values))
                entry[f"{metric_name}_std"] = float(np.std(values))
            cv_summary[qoi_name] = entry

        logger.info("Cross-validation complete over %d folds.", k)
        return cv_summary

    def _uses_scaled_features(self) -> bool:
        """Whether the model is trained/served on MinMax-scaled features.

        Only the neural network needs scaled inputs.  Kriging standardises its
        inputs internally (scaling is a no-op for it) and RBF empirically fits
        better on the raw feature geometry, so both use raw features; the d0
        search range is sized for raw pairwise distances accordingly.
        """
        return self.model_type == ModelType.NN

    def _create_and_train_model(
        self,
        z_train: np.ndarray,
        z_val: np.ndarray,
    ) -> Any:
        """Instantiate and train the appropriate model type."""
        if self.model_type == ModelType.KRIGING:
            model: Any = KrigingVectorModel()
        elif self.model_type == ModelType.RBF:
            model = RBFVectorModel()
        elif self.model_type == ModelType.NN:
            model = NeuralNetworkModel()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self._uses_scaled_features():
            x_train, x_val = (
                self.processed_data["X_train_scaled"],
                self.processed_data["X_val_scaled"],
            )
        else:
            x_train, x_val = self.processed_data["X_train_raw"], self.processed_data["X_val_raw"]
        model.train(x_train, z_train, x_val, z_val)
        return model

    def _decode_qoi(self, qoi_name: str, z_pred: np.ndarray) -> np.ndarray:
        """Map a latent prediction back to physical units.

        The full decode chain (latent unstandardisation, PCA inverse and target
        unscaling) is linear; this is exploited by :meth:`predict_with_uncertainty`.
        """
        y_scaled = self.compressors_per_qoi[qoi_name].inverse_transform(z_pred)
        scaler = self.y_scalers[qoi_name]
        physical: np.ndarray = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(
            y_scaled.shape
        )
        return physical

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_and_plot(
        self,
        plot_prefix: str = "surrogate_comparison",
        make_plots: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on the held-out test set in physical units.

        Predictions and ground truth are compared in the original physical
        units (not the internal scaled space), so the metrics — and the
        comparison against the optional physical reference model — are
        meaningful and unit-consistent.

        Args:
            plot_prefix: Path prefix for the generated comparison plots.
            make_plots: When False, metrics are computed without writing plots
                (used by cross-validation and benchmarking).

        Returns:
            ``{qoi_name: {metric: value}}`` aggregated over the test runs.
        """
        x_test = (
            self.processed_data["X_test_scaled"]
            if self._uses_scaled_features()
            else self.processed_data["X_test_raw"]
        )
        if len(x_test) == 0:
            logger.warning("Test data is empty; skipping evaluation.")
            return {}

        if self.test_run_id_map is None:
            raise RuntimeError("No test split available; train the factory first.")

        accum: dict[str, dict[str, list]] = {
            qoi: {
                "rmse": [],
                "mae": [],
                "max_err": [],
                "r2": [],
                "rmse_ref": [],
                "r2_ref": [],
                "y_true_pool": [],
                "y_pred_pool": [],
            }
            for qoi in self.trained_models_per_qoi
        }
        has_ref = bool(self.reference_qois)

        for idx in range(len(x_test)):
            run_id = self.test_run_id_map.get(idx, str(idx))
            x_sample = x_test[[idx]]

            for qoi_name, model in self.trained_models_per_qoi.items():
                physical = self.processed_data["y_test_per_qoi_physical"][qoi_name]
                y_true_flat = physical[idx].flatten()
                z_pred = model.predict_values(x_sample)
                y_pred_flat = self._decode_qoi(qoi_name, z_pred).flatten()

                mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
                y_true_clean = y_true_flat[mask]
                y_pred_clean = y_pred_flat[mask]

                y_pred_ref = (
                    self._lookup_reference(run_id, qoi_name, len(y_true_flat)) if has_ref else None
                )

                if len(y_true_clean) == 0:
                    logger.warning("Run %s | %s: all values are NaN/Inf.", run_id, qoi_name)
                    continue

                rmse = calculate_rmse(y_true_clean, y_pred_clean)
                accum[qoi_name]["rmse"].append(rmse)
                accum[qoi_name]["y_true_pool"].append(y_true_clean)
                accum[qoi_name]["y_pred_pool"].append(y_pred_clean)
                accum[qoi_name]["mae"].append(calculate_mae(y_true_clean, y_pred_clean))
                accum[qoi_name]["max_err"].append(
                    calculate_max_abs_error(y_true_clean, y_pred_clean)
                )
                # R² is undefined for a constant target; fall back to an RMSE gate.
                if np.var(y_true_clean) < 1e-9:
                    accum[qoi_name]["r2"].append(1.0 if rmse < 1e-5 else 0.0)
                else:
                    accum[qoi_name]["r2"].append(calculate_r2(y_true_clean, y_pred_clean))

                if y_pred_ref is not None:
                    y_true_ref = y_true_flat[: len(y_pred_ref)]
                    mask_ref = np.isfinite(y_true_ref) & np.isfinite(y_pred_ref)
                    if np.any(mask_ref):
                        accum[qoi_name]["rmse_ref"].append(
                            calculate_rmse(y_true_ref[mask_ref], y_pred_ref[mask_ref])
                        )
                        accum[qoi_name]["r2_ref"].append(
                            calculate_r2(y_true_ref[mask_ref], y_pred_ref[mask_ref])
                        )

                if make_plots:
                    plot_comparison_timeseries(
                        y_true=y_true_flat,
                        y_pred_new=y_pred_flat,
                        y_pred_old=y_pred_ref,
                        run_number=idx,
                        qoi_name=f"{qoi_name} (Run ID: {run_id})",
                        save_path_prefix=plot_prefix,
                        new_model_label=f"{self.model_type.value} Prediction",
                    )

        summary = self._summarise_metrics(accum)
        self._log_evaluation_summary(summary)
        return summary

    def test_predictions(self) -> dict[str, np.ndarray]:
        """Physical-unit predictions for the held-out test set, per QoI.

        Returns ``{qoi_name: (n_test, n_timesteps)}``. Useful for building an
        ensemble across model types trained on the same split.
        """
        x_test = (
            self.processed_data["X_test_scaled"]
            if self._uses_scaled_features()
            else self.processed_data["X_test_raw"]
        )
        return {
            qoi_name: self._decode_qoi(qoi_name, model.predict_values(x_test))
            for qoi_name, model in self.trained_models_per_qoi.items()
        }

    def _lookup_reference(self, run_id: Any, qoi_name: str, max_len: int) -> np.ndarray | None:
        """Fetch the physical reference prediction for a run/QoI, if available."""
        ref_data = self.reference_qois.get(str(run_id), {})
        ref_raw = ref_data.get(qoi_name)
        if ref_raw is None:
            return None
        y_pred_ref = np.array(ref_raw).flatten()
        return y_pred_ref[:max_len]

    def _summarise_metrics(self, accum: dict[str, dict[str, list]]) -> dict[str, dict[str, float]]:
        """Aggregate per-run metrics into mean/std and a range-normalised NRMSE."""
        summary: dict[str, dict[str, float]] = {}
        for qoi_name, m in accum.items():
            if not m["rmse"]:
                continue
            physical = self.processed_data["y_test_per_qoi_physical"][qoi_name]
            value_range = float(np.nanmax(physical) - np.nanmin(physical))
            rmse_mean = float(np.mean(m["rmse"]))
            # Pooled (global) metrics over every test point — the standard
            # "fraction of variance explained" headline, robust to the harsh
            # per-run R² on near-flat individual curves.
            y_true_all = np.concatenate(m["y_true_pool"])
            y_pred_all = np.concatenate(m["y_pred_pool"])
            global_r2 = calculate_r2(y_true_all, y_pred_all) if np.var(y_true_all) > 1e-12 else 0.0
            entry = {
                "global_r2": float(global_r2),
                "global_rmse": calculate_rmse(y_true_all, y_pred_all),
                "rmse_mean": rmse_mean,
                "rmse_std": float(np.std(m["rmse"])),
                "mae_mean": float(np.mean(m["mae"])),
                "max_err_mean": float(np.mean(m["max_err"])),
                "nrmse": rmse_mean / value_range if value_range > 0 else 0.0,
                "r2_mean": float(np.mean(m["r2"])),
                "r2_std": float(np.std(m["r2"])),
                "n_samples": float(len(m["rmse"])),
            }
            if m["r2_ref"]:
                entry["r2_ref_mean"] = float(np.mean(m["r2_ref"]))
                entry["rmse_ref_mean"] = float(np.mean(m["rmse_ref"]))
            summary[qoi_name] = entry
        return summary

    @staticmethod
    def _log_evaluation_summary(summary: dict[str, dict[str, float]]) -> None:
        header = (
            f"{'QoI':<40} | {'GLOBAL R²':<10} | {'RMSE mean(std)':<22} | "
            f"{'NRMSE':<8} | {'per-run R² mean(std)':<22} | Ref R²"
        )
        logger.info("\n%s\n%s", header, "-" * len(header))
        for qoi_name, m in summary.items():
            rmse_str = f"{m['rmse_mean']:.4g} (+/-{m['rmse_std']:.2g})"
            r2_str = f"{m['r2_mean']:.4f} (+/-{m['r2_std']:.3f})"
            ref_str = f"{m['r2_ref_mean']:.4f}" if "r2_ref_mean" in m else "N/A"
            logger.info(
                "%s | %s | %s | %s | %s | %s",
                qoi_name[:40].ljust(40),
                f"{m['global_r2']:.4f}".ljust(10),
                rmse_str.ljust(22),
                f"{m['nrmse']:.4f}".ljust(8),
                r2_str.ljust(22),
                ref_str,
            )

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

        categorical_cols = set(self.metadata.keys())
        rows: list[list[float]] = []

        for row in features_df_ordered.itertuples(index=False):
            values: list[float] = []
            for i, value in enumerate(row):
                base = base_caption(self.feature_names[i])
                if base in categorical_cols:
                    try:
                        values.append(float(self.metadata[base].index(value)))
                    except ValueError:
                        logger.warning(
                            "Unknown categorical value '%s' for '%s'; encoding as -1.",
                            value,
                            self.feature_names[i],
                        )
                        values.append(-1.0)
                else:
                    values.append(float(value))
            rows.append(values)

        return np.array(rows)

    def predict(self, x_new: pd.DataFrame) -> dict[str, np.ndarray]:
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

        x_encoded = self._preprocess_input_features(x_new)
        x_input = self.x_scaler.transform(x_encoded) if self._uses_scaled_features() else x_encoded

        predictions: dict[str, np.ndarray] = {}
        for qoi_name, model in self.trained_models_per_qoi.items():
            z_pred = model.predict_values(x_input)
            predictions[qoi_name] = self._decode_qoi(qoi_name, z_pred)

        logger.info("Prediction complete for %d QoI(s).", len(predictions))
        return predictions

    def predict_with_uncertainty(
        self, x_new: pd.DataFrame
    ) -> dict[str, "tuple[np.ndarray, np.ndarray]"]:
        """Predict the mean and 1-sigma std (physical units) per QoI.

        Only Kriging exposes a predictive variance.  The latent variance is
        propagated to physical units exactly: both the latent unstandardisation,
        PCA inverse, and MinMax unscaling are linear, so

            std_physical = (y_max - y_min) * sqrt(sum_k W[k,t]**2 var_zraw[k]).

        Args:
            x_new: DataFrame whose columns match the training features.

        Returns:
            ``{qoi_name: (mean, std)}``, each array shaped ``(n_samples, n_timesteps)``.

        Raises:
            NotImplementedError: If the model type is not Kriging.
        """
        if self.model_type != ModelType.KRIGING:
            raise NotImplementedError(
                "Predictive uncertainty is only available for Kriging models."
            )
        if not self.trained_models_per_qoi:
            raise RuntimeError("No trained models. Call .train() or .load_model() first.")
        if self.x_scaler is None:
            raise RuntimeError("Scalers not found; the factory is not fully trained.")

        # Kriging is trained on raw features, so predict in the same space.
        x_input = self._preprocess_input_features(x_new)

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for qoi_name, model in self.trained_models_per_qoi.items():
            z_pred = model.predict_values(x_input)
            var_z = model.predict_variances(x_input)

            mean_physical = self._decode_qoi(qoi_name, z_pred)
            std_scaled = self.compressors_per_qoi[qoi_name].propagate_latent_variance(var_z)
            std_physical = std_scaled * float(self.y_scalers[qoi_name].data_range_[0])
            results[qoi_name] = (mean_physical, std_physical)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, file_path: str | Path, include_eval_data: bool = False) -> None:
        """Serialise the trained factory to disk via pickle.

        By default only the artefacts needed for inference are stored, keeping
        ``model.pkl`` small and the served model lean.  Set *include_eval_data*
        to also persist the held-out test arrays and reference data so the model
        can be reloaded and re-evaluated/re-plotted without retraining.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._pickle_include_eval_data = include_eval_data
        try:
            with open(file_path, "wb") as fh:
                pickle.dump(self, fh)
        finally:
            self._pickle_include_eval_data = False
        logger.info("Model saved to %s (eval data included: %s).", file_path, include_eval_data)

    def __getstate__(self) -> dict[str, Any]:
        """Drop bulky training artefacts unless eval data was explicitly requested."""
        state = self.__dict__.copy()
        include_eval_data = state.pop("_pickle_include_eval_data", False)
        if not include_eval_data:
            state["_df_features_raw"] = None
            state["_qois_raw"] = None
            state["_all_runs_ids"] = None
            state["processed_data"] = {}
            state["reference_qois"] = {}
            state["test_run_id_map"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pickle_include_eval_data = False

    @classmethod
    def load_model(cls, file_path: str | Path) -> "SurrogateFactory":
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
