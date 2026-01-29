"""Data preprocessing: feature encoding, scaling, PCA compression, and QoI restructuring."""

import logging
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature array construction
# ---------------------------------------------------------------------------

def _build_feature_array(
    features_df: pd.DataFrame,
    metadata: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Convert a features DataFrame into a numeric numpy array.

    Categorical columns (listed in *metadata*) are ordinal-encoded using the
    index within their valid-values list.  All other columns are cast to float.

    Returns:
        ``(X_array, ordered_feature_names)``
    """
    categorical_cols = list(metadata.keys())
    cols_to_ignore = {"run_id", "run_number"}
    all_feature_names = [c for c in features_df.columns if c not in cols_to_ignore]

    # Consistent ordering: categorical columns first, then numeric.
    ordered = [c for c in all_feature_names if c in categorical_cols]
    ordered.extend(c for c in all_feature_names if c not in categorical_cols)

    rows: list[list[float]] = []
    for row in features_df[ordered].itertuples(index=False):
        values: list[float] = []
        for i, value in enumerate(row):
            col_name = ordered[i]
            if col_name in categorical_cols:
                try:
                    values.append(float(metadata[col_name].index(value)))
                except ValueError:
                    logger.warning(
                        "Value '%s' not found in metadata for '%s'; encoding as -1.",
                        value,
                        col_name,
                    )
                    values.append(-1.0)
            else:
                values.append(float(value))
        rows.append(values)

    return np.array(rows), ordered


# ---------------------------------------------------------------------------
# QoI restructuring
# ---------------------------------------------------------------------------

def _restructure_qois(
    qois_per_run: Dict[Any, Dict[str, List[float]]],
    run_ids: List[Any],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Pivot QoI data from ``{run: {qoi: vector}}`` to ``{qoi: (n_runs, n_timesteps)}``."""
    if not qois_per_run:
        return {}, []

    all_qoi_names_set: Set[str] = set()
    for qoi_dict in qois_per_run.values():
        all_qoi_names_set.update(qoi_dict.keys())

    all_qoi_names = sorted(all_qoi_names_set)
    y_per_qoi: Dict[str, list[np.ndarray]] = {q: [] for q in all_qoi_names}
    valid_qoi_names = set(all_qoi_names)

    for run_id in run_ids:
        run_id_key: Any = str(run_id)
        if run_id_key not in qois_per_run:
            if isinstance(run_id, int) and run_id in qois_per_run:
                run_id_key = run_id
            else:
                continue

        run_data = qois_per_run[run_id_key]
        for qoi_name in list(valid_qoi_names):
            vec = run_data.get(qoi_name)
            if vec is not None:
                y_per_qoi[qoi_name].append(np.array(vec))
            else:
                logger.warning(
                    "QoI '%s' missing from run #%s; excluding this QoI.",
                    qoi_name,
                    run_id,
                )
                valid_qoi_names.discard(qoi_name)
                y_per_qoi.pop(qoi_name, None)

    final: Dict[str, np.ndarray] = {}
    for qoi_name in valid_qoi_names:
        data_list = y_per_qoi[qoi_name]
        if data_list:
            try:
                stacked = np.array(data_list)
                if stacked.ndim == 1:
                    raise ValueError("Inconsistent vector lengths.")
                final[qoi_name] = stacked
            except ValueError:
                logger.warning(
                    "QoI '%s' has inconsistent vector lengths across runs; excluding.",
                    qoi_name,
                )

    return final, sorted(final.keys())


def _get_id_list(df: pd.DataFrame) -> List[Any]:
    """Extract the run identifier column as a list."""
    if "run_id" in df.columns:
        return df["run_id"].tolist()
    if "run_number" in df.columns:
        return df["run_number"].tolist()
    return []


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def process_data_multi_model(
    df_train_features: pd.DataFrame,
    qois_train: Dict[Any, Dict[str, List[float]]],
    df_val_features: pd.DataFrame,
    qois_val: Dict[Any, Dict[str, List[float]]],
    df_test_features: pd.DataFrame,
    qois_test: Dict[Any, Dict[str, List[float]]],
    metadata: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Run the full preprocessing pipeline for the multi-model factory.

    Steps:
        1. Encode and build feature arrays (X).
        2. Fit a ``MinMaxScaler`` on training features.
        3. Restructure QoI targets from per-run to per-QoI layout.
        4. Ensure train / val / test share the same QoI set.
        5. Fit per-QoI ``MinMaxScaler`` on training targets.

    Returns:
        Dictionary containing raw and scaled feature arrays, per-QoI target
        arrays, fitted scalers, QoI names, and feature names.
    """
    # 1. Features (X) ----------------------------------------------------------
    logger.info("Building feature arrays.")
    x_train_raw, feature_names = _build_feature_array(df_train_features, metadata)
    x_val_raw, _ = _build_feature_array(df_val_features[df_train_features.columns], metadata)
    x_test_raw, _ = _build_feature_array(df_test_features[df_train_features.columns], metadata)

    # 2. Scale features --------------------------------------------------------
    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train_raw)
    x_val_scaled = x_scaler.transform(x_val_raw)
    x_test_scaled = x_scaler.transform(x_test_raw)

    logger.info("Feature processing complete. X_train shape: %s", x_train_raw.shape)

    # 3. Targets (Y) -----------------------------------------------------------
    logger.info("Restructuring target data by QoI.")
    y_train_per_qoi, train_qoi_names = _restructure_qois(qois_train, _get_id_list(df_train_features))
    y_val_per_qoi, val_qoi_names = _restructure_qois(qois_val, _get_id_list(df_val_features))
    y_test_per_qoi, test_qoi_names = _restructure_qois(qois_test, _get_id_list(df_test_features))

    # 4. Intersect QoI names ---------------------------------------------------
    valid_qoi_names = sorted(set(train_qoi_names) & set(val_qoi_names) & set(test_qoi_names))

    final_y_train = {q: y_train_per_qoi[q] for q in valid_qoi_names if q in y_train_per_qoi}
    final_y_val = {q: y_val_per_qoi[q] for q in valid_qoi_names if q in y_val_per_qoi}
    final_y_test = {q: y_test_per_qoi[q] for q in valid_qoi_names if q in y_test_per_qoi}

    final_valid_qoi_names = sorted(final_y_train.keys())

    # 5. Scale targets per QoI -------------------------------------------------
    y_scalers: Dict[str, MinMaxScaler] = {}
    y_train_scaled: Dict[str, np.ndarray] = {}
    y_val_scaled: Dict[str, np.ndarray] = {}
    y_test_scaled: Dict[str, np.ndarray] = {}

    for qoi in final_valid_qoi_names:
        scaler = MinMaxScaler()
        flat = final_y_train[qoi].flatten().reshape(-1, 1)
        scaler.fit(flat)
        y_scalers[qoi] = scaler

        y_train_scaled[qoi] = scaler.transform(flat).reshape(final_y_train[qoi].shape)
        y_val_scaled[qoi] = scaler.transform(
            final_y_val[qoi].flatten().reshape(-1, 1)
        ).reshape(final_y_val[qoi].shape)
        y_test_scaled[qoi] = scaler.transform(
            final_y_test[qoi].flatten().reshape(-1, 1)
        ).reshape(final_y_test[qoi].shape)

    logger.info("Preprocessing complete. %d consistent QoI(s) found.", len(final_valid_qoi_names))

    return {
        # Features
        "X_train_raw": x_train_raw,
        "X_val_raw": x_val_raw,
        "X_test_raw": x_test_raw,
        "X_train_scaled": x_train_scaled,
        "X_val_scaled": x_val_scaled,
        "X_test_scaled": x_test_scaled,
        "x_scaler": x_scaler,
        # Targets
        "y_train_per_qoi": y_train_scaled,
        "y_val_per_qoi": y_val_scaled,
        "y_test_per_qoi": y_test_scaled,
        "y_scalers": y_scalers,
        # Metadata
        "qoi_names": final_valid_qoi_names,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# PCA output compressor
# ---------------------------------------------------------------------------

class OutputCompressor:
    """Reduce high-dimensional time-series output via PCA.

    Args:
        n_components: If a float in (0, 1), the fraction of variance to
            retain.  If an integer, the exact number of components.
    """

    def __init__(self, n_components: Union[float, int] = 0.99) -> None:
        self._pca = PCA(n_components=n_components)
        self.is_fitted: bool = False

    @property
    def n_components_(self) -> int:
        """Number of components retained after fitting."""
        return int(self._pca.n_components_)

    def fit_transform(self, y_raw: np.ndarray) -> np.ndarray:
        """Fit PCA on *y_raw* and return the latent representation.

        Args:
            y_raw: Matrix of shape ``(n_samples, n_timesteps)``.

        Returns:
            Latent matrix ``(n_samples, n_components)``.
        """
        logger.info("Fitting PCA on output of shape %s.", y_raw.shape)
        z = self._pca.fit_transform(y_raw)
        self.is_fitted = True

        explained = float(np.sum(self._pca.explained_variance_ratio_))
        logger.info(
            "PCA reduced to %d components (explained variance: %.4f).",
            self._pca.n_components_,
            explained,
        )
        return z

    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        """Project new data into the learned latent space (no re-fitting)."""
        if not self.is_fitted:
            raise RuntimeError("Compressor must be fitted before calling transform().")
        return self._pca.transform(y_raw)

    def inverse_transform(self, z_latent: np.ndarray) -> np.ndarray:
        """Reconstruct time-series from latent components."""
        if not self.is_fitted:
            raise RuntimeError("Compressor must be fitted before calling inverse_transform().")
        return self._pca.inverse_transform(z_latent)
