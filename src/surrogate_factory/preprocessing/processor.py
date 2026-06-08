"""Data preprocessing: feature encoding, scaling, PCA compression, and QoI restructuring."""

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# Duplicate captions are disambiguated as "<caption> [2]" by the runs-specs
# parser; the base caption is what the categorical metadata is keyed on.
_CAPTION_SUFFIX = re.compile(r" \[\d+\]$")


def base_caption(column_name: str) -> str:
    """Strip a duplicate-disambiguation suffix (``"X [2]"`` -> ``"X"``)."""
    return _CAPTION_SUFFIX.sub("", column_name)


# ---------------------------------------------------------------------------
# Feature array construction
# ---------------------------------------------------------------------------


def _build_feature_array(
    features_df: pd.DataFrame,
    metadata: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Convert a features DataFrame into a numeric numpy array.

    Categorical columns (those whose base caption is in *metadata*) are
    ordinal-encoded using the index within their valid-values list.  All other
    columns are cast to float.

    Returns:
        ``(X_array, ordered_feature_names)``
    """
    categorical_cols = set(metadata.keys())
    cols_to_ignore = {"run_id", "run_number"}
    all_feature_names = [c for c in features_df.columns if c not in cols_to_ignore]

    def is_categorical(column: str) -> bool:
        return base_caption(column) in categorical_cols

    # Consistent ordering: categorical columns first, then numeric.
    ordered = [c for c in all_feature_names if is_categorical(c)]
    ordered.extend(c for c in all_feature_names if not is_categorical(c))

    rows: list[list[float]] = []
    for row in features_df[ordered].itertuples(index=False):
        values: list[float] = []
        for i, value in enumerate(row):
            col_name = ordered[i]
            if is_categorical(col_name):
                valid_values = metadata[base_caption(col_name)]
                try:
                    values.append(float(valid_values.index(value)))
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
    qois_per_run: dict[Any, dict[str, list[float]]],
    run_ids: list[Any],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Pivot QoI data from ``{run: {qoi: vector}}`` to ``{qoi: (n_runs, n_timesteps)}``."""
    if not qois_per_run:
        return {}, []

    all_qoi_names_set: set[str] = set()
    for qoi_dict in qois_per_run.values():
        all_qoi_names_set.update(qoi_dict.keys())

    all_qoi_names = sorted(all_qoi_names_set)
    y_per_qoi: dict[str, list[np.ndarray]] = {q: [] for q in all_qoi_names}
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

    final: dict[str, np.ndarray] = {}
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


def _get_id_list(df: pd.DataFrame) -> list[Any]:
    """Extract the run identifier column as a list."""
    if "run_id" in df.columns:
        run_ids: list[Any] = df["run_id"].tolist()
        return run_ids
    if "run_number" in df.columns:
        run_numbers: list[Any] = df["run_number"].tolist()
        return run_numbers
    return []


def _resolve_qoi_key(run_id: Any, qois_per_run: dict[Any, Any]) -> Any:
    """Return the key under which *run_id* is stored in *qois_per_run*, or None.

    Run identifiers reach this module as either ``str`` (standard format) or
    ``int`` (ESSS ``run_number``); the QoI dictionary may use either, so both
    are probed.
    """
    if str(run_id) in qois_per_run:
        return str(run_id)
    if run_id in qois_per_run:
        return run_id
    return None


def _align_features_and_qois(
    df_features: pd.DataFrame,
    qois_per_run: dict[Any, Any],
    split_name: str,
    max_drop_fraction: float,
) -> tuple[pd.DataFrame, list[Any]]:
    """Keep only runs present in *both* the feature table and the QoI dict.

    Building the feature matrix from every DataFrame row while skipping
    QoI-less runs when restructuring targets silently desynchronises X and Y by
    row index.  Aligning up-front, preserving DataFrame order, guarantees the
    i-th feature row corresponds to the i-th target row.

    :raises RuntimeError:
        If the dropped fraction exceeds *max_drop_fraction*, which signals a
        systematic data problem rather than a few isolated failed runs.
    """
    id_list = _get_id_list(df_features)
    if not id_list:
        return df_features, []

    keep_flags = [_resolve_qoi_key(rid, qois_per_run) is not None for rid in id_list]
    aligned_ids = [rid for rid, keep in zip(id_list, keep_flags, strict=True) if keep]
    dropped = [rid for rid, keep in zip(id_list, keep_flags, strict=True) if not keep]

    if dropped:
        fraction = len(dropped) / len(id_list)
        message = (
            f"{split_name} split: {len(dropped)}/{len(id_list)} runs "
            f"({fraction:.0%}) have no QoI data and were dropped; "
            f"examples: {dropped[:10]}"
        )
        if fraction > max_drop_fraction:
            raise RuntimeError(
                f"Too many runs without QoI data ({fraction:.0%} > "
                f"{max_drop_fraction:.0%}). {message}"
            )
        logger.warning(message)

    df_aligned = df_features[pd.Series(keep_flags, index=df_features.index)]
    return df_aligned, aligned_ids


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def process_data_multi_model(
    df_train_features: pd.DataFrame,
    qois_train: dict[Any, dict[str, list[float]]],
    df_val_features: pd.DataFrame,
    qois_val: dict[Any, dict[str, list[float]]],
    df_test_features: pd.DataFrame,
    qois_test: dict[Any, dict[str, list[float]]],
    metadata: dict[str, list[str]],
    max_drop_fraction: float = 0.2,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline for the multi-model factory.

    Steps:
        0. Align features and QoIs per split (drop runs missing from either).
        1. Encode and build feature arrays (X).
        2. Fit a ``MinMaxScaler`` on training features.
        3. Restructure QoI targets from per-run to per-QoI layout.
        4. Ensure train / val / test share the same QoI set.
        5. Fit per-QoI ``MinMaxScaler`` on training targets.

    Args:
        max_drop_fraction: Maximum fraction of runs a split may lose to missing
            QoI data before a :class:`RuntimeError` is raised (see
            :func:`_align_features_and_qois`).

    Returns:
        Dictionary with raw/scaled feature arrays, per-QoI scaled targets, the
        unscaled (physical) targets for honest evaluation, fitted scalers, the
        aligned test run identifiers, QoI names, and feature names.
    """
    # 0. Align X and Y per split ----------------------------------------------
    df_train = df_train_features
    df_val = df_val_features[df_train_features.columns]
    df_test = df_test_features[df_train_features.columns]

    df_train, ids_train = _align_features_and_qois(df_train, qois_train, "train", max_drop_fraction)
    df_val, ids_val = _align_features_and_qois(df_val, qois_val, "validation", max_drop_fraction)
    df_test, ids_test = _align_features_and_qois(df_test, qois_test, "test", max_drop_fraction)

    # 1. Features (X) ----------------------------------------------------------
    logger.info("Building feature arrays.")
    x_train_raw, feature_names = _build_feature_array(df_train, metadata)
    x_val_raw, _ = _build_feature_array(df_val, metadata)
    x_test_raw, _ = _build_feature_array(df_test, metadata)

    # 2. Scale features --------------------------------------------------------
    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train_raw)
    x_val_scaled = x_scaler.transform(x_val_raw)
    x_test_scaled = x_scaler.transform(x_test_raw)

    logger.info("Feature processing complete. X_train shape: %s", x_train_raw.shape)

    # 3. Targets (Y) -----------------------------------------------------------
    logger.info("Restructuring target data by QoI.")
    y_train_per_qoi, train_qoi_names = _restructure_qois(qois_train, ids_train)
    y_val_per_qoi, val_qoi_names = _restructure_qois(qois_val, ids_val)
    y_test_per_qoi, test_qoi_names = _restructure_qois(qois_test, ids_test)

    # 4. Intersect QoI names ---------------------------------------------------
    valid_qoi_names = sorted(set(train_qoi_names) & set(val_qoi_names) & set(test_qoi_names))

    final_y_train = {q: y_train_per_qoi[q] for q in valid_qoi_names if q in y_train_per_qoi}
    final_y_val = {q: y_val_per_qoi[q] for q in valid_qoi_names if q in y_val_per_qoi}
    final_y_test = {q: y_test_per_qoi[q] for q in valid_qoi_names if q in y_test_per_qoi}

    final_valid_qoi_names = sorted(final_y_train.keys())

    # 5. Scale targets per QoI -------------------------------------------------
    y_scalers: dict[str, MinMaxScaler] = {}
    y_train_scaled: dict[str, np.ndarray] = {}
    y_val_scaled: dict[str, np.ndarray] = {}
    y_test_scaled: dict[str, np.ndarray] = {}

    for qoi in final_valid_qoi_names:
        scaler = MinMaxScaler()
        flat = final_y_train[qoi].flatten().reshape(-1, 1)
        scaler.fit(flat)
        y_scalers[qoi] = scaler

        y_train_scaled[qoi] = scaler.transform(flat).reshape(final_y_train[qoi].shape)
        y_val_scaled[qoi] = scaler.transform(final_y_val[qoi].flatten().reshape(-1, 1)).reshape(
            final_y_val[qoi].shape
        )
        y_test_scaled[qoi] = scaler.transform(final_y_test[qoi].flatten().reshape(-1, 1)).reshape(
            final_y_test[qoi].shape
        )

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
        # Targets (scaled, for model fitting)
        "y_train_per_qoi": y_train_scaled,
        "y_val_per_qoi": y_val_scaled,
        "y_test_per_qoi": y_test_scaled,
        # Targets (physical, for honest evaluation against the reference model)
        "y_train_per_qoi_physical": final_y_train,
        "y_test_per_qoi_physical": final_y_test,
        "y_scalers": y_scalers,
        # Metadata
        "qoi_names": final_valid_qoi_names,
        "feature_names": feature_names,
        "test_run_ids": ids_test,
    }


# ---------------------------------------------------------------------------
# PCA output compressor
# ---------------------------------------------------------------------------


class OutputCompressor:
    """Reduce high-dimensional time-series output via PCA, then standardise the
    latent components.

    PCA components have steeply decreasing variance; a model regressing the raw
    latent vector (NN, RBF) is dominated by the first component and under-fits
    the fine detail carried by the trailing modes.  Standardising the latent
    representation per component equalises their weight in the loss.  Kriging is
    unaffected (it standardises each target internally), so the extra step is
    applied uniformly and harmlessly.

    Both transforms are linear, so :meth:`propagate_latent_variance` can map a
    latent variance back to the scaled-output space in closed form.

    Args:
        n_components: If a float in (0, 1), the fraction of variance to
            retain.  If an integer, the exact number of components.
    """

    def __init__(self, n_components: float | int = 0.99) -> None:
        self._pca = PCA(n_components=n_components)
        self._z_scaler = StandardScaler()
        self.is_fitted: bool = False

    @property
    def n_components_(self) -> int:
        """Number of components retained after fitting."""
        return int(self._pca.n_components_)

    @property
    def explained_variance_ratio(self) -> float:
        """Total fraction of output variance retained by the kept components."""
        return float(np.sum(self._pca.explained_variance_ratio_))

    @property
    def explained_variance_ratio_per_component(self) -> np.ndarray:
        """Per-component fraction of output variance (ordered, descending)."""
        return np.asarray(self._pca.explained_variance_ratio_, dtype=float)

    def fit_transform(self, y_raw: np.ndarray) -> np.ndarray:
        """Fit PCA + latent scaler on *y_raw* and return the latent representation.

        Args:
            y_raw: Matrix of shape ``(n_samples, n_timesteps)``.

        Returns:
            Standardised latent matrix ``(n_samples, n_components)``.
        """
        logger.info("Fitting PCA on output of shape %s.", y_raw.shape)
        z_raw = self._pca.fit_transform(y_raw)
        z_std: np.ndarray = self._z_scaler.fit_transform(z_raw)
        self.is_fitted = True

        logger.info(
            "PCA reduced to %d components (explained variance: %.4f).",
            self._pca.n_components_,
            self.explained_variance_ratio,
        )
        return z_std

    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        """Project new data into the learned latent space (no re-fitting)."""
        if not self.is_fitted:
            raise RuntimeError("Compressor must be fitted before calling transform().")
        projected: np.ndarray = self._z_scaler.transform(self._pca.transform(y_raw))
        return projected

    def inverse_transform(self, z_latent: np.ndarray) -> np.ndarray:
        """Reconstruct (scaled) time-series from standardised latent components."""
        if not self.is_fitted:
            raise RuntimeError("Compressor must be fitted before calling inverse_transform().")
        z_raw = self._z_scaler.inverse_transform(z_latent)
        reconstructed: np.ndarray = self._pca.inverse_transform(z_raw)
        return reconstructed

    def propagate_latent_variance(self, var_z_std: np.ndarray) -> np.ndarray:
        """Map a per-component latent variance to a per-timestep std-dev.

        Both decode steps are linear, so for a diagonal latent covariance
        ``diag(var_z_std)`` the reconstructed (scaled) output variance is exact:

            z_raw = z_std * z_scale            => var_raw = var_z_std * z_scale**2
            y = z_raw @ components_ + mean      => var_y[t] = sum_k W[k,t]**2 var_raw[k]

        Independence across components is assumed, which matches how the
        per-component Kriging models are trained (no cross-covariance estimated).

        Args:
            var_z_std: Latent variances ``(n_samples, n_components)`` in the
                standardised space.

        Returns:
            Std-dev in the scaled-output space ``(n_samples, n_timesteps)``.
        """
        if not self.is_fitted:
            raise RuntimeError("Compressor must be fitted before propagating variance.")
        var_z_raw = var_z_std * (self._z_scaler.scale_**2)
        var_scaled = var_z_raw @ (self._pca.components_**2)
        std_scaled: np.ndarray = np.sqrt(var_scaled)
        return std_scaled
