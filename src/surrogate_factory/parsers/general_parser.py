"""Parser for the standard CSV + JSON data format."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_standard_format(
    features_path: Path,
    targets_path: Path,
    run_ids_to_load: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Load simulation data from a features CSV and a targets JSON file.

    Args:
        features_path: Path to the ``.csv`` file containing input features.
            Must include a ``run_id`` column.
        targets_path: Path to the ``.json`` file mapping run IDs to output
            vectors (either ``{run_id: [values]}`` or
            ``{run_id: {qoi_name: [values]}}``).
        run_ids_to_load: Optional subset of run IDs to load.  When *None*,
            all available runs are loaded.

    Returns:
        A tuple of:
        1. A ``DataFrame`` of features (filtered to matching runs).
        2. A dictionary of QoIs in the shape
           ``{run_id: {'outputs': <time-series>}}``.

    Raises:
        FileNotFoundError: If either input file does not exist.
    """
    if not features_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError(
            f"One or both data files not found: {features_path}, {targets_path}"
        )

    df_features = pd.read_csv(features_path)
    df_features["run_id"] = df_features["run_id"].astype(str)

    with open(targets_path, "r", encoding="UTF-8") as fh:
        targets_dict: Dict[str, Any] = json.load(fh)

    if run_ids_to_load:
        run_ids_to_load = [str(r) for r in run_ids_to_load]
        df_features = df_features[df_features["run_id"].isin(run_ids_to_load)].copy()
        targets_dict = {
            rid: data for rid, data in targets_dict.items() if rid in run_ids_to_load
        }
    else:
        df_features = df_features[
            df_features["run_id"].isin(targets_dict.keys())
        ].copy()

    qois_per_run: Dict[str, Dict[str, Any]] = {}
    for run_id, time_series in targets_dict.items():
        if run_id in df_features["run_id"].values:
            qois_per_run[run_id] = {"outputs": time_series}

    df_features = df_features[
        df_features["run_id"].isin(qois_per_run.keys())
    ].reset_index(drop=True)

    logger.info(
        "Loaded %d runs from %s / %s.",
        len(df_features),
        features_path.name,
        targets_path.name,
    )
    return df_features, qois_per_run
