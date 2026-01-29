"""Parser for ESSS simulation JSON file formats."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_variable_metadata(file_path: Path) -> Dict[str, List[str]]:
    """Read the ``input_metrics`` metadata JSON file.

    Returns:
        Mapping of variable caption to its list of valid values.
    """
    if not file_path.is_file():
        logger.warning("Metadata file not found at '%s'.", file_path)
        return {}

    with open(file_path, "r", encoding="UTF-8") as fh:
        data = json.load(fh)

    variable_metadata: Dict[str, List[str]] = {}
    for metric_container in data.get("input_metrics", []):
        for metric_data in metric_container.values():
            caption = metric_data.get("caption")
            valid_values = metric_data.get("valid_values")
            if caption and valid_values:
                variable_metadata[caption] = valid_values

    logger.debug("Parsed variable metadata: %s", variable_metadata)
    return variable_metadata


def _flatten_run_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested ``runs-specs`` DataFrame into a tabular format."""
    processed_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        new_row: dict[str, Any] = {"run_number": row["run_number"]}
        for metric in row["metrics"]:
            for metric_data in metric.values():
                caption = metric_data.get("caption")
                value = metric_data.get("value")
                if caption and value is not None:
                    new_row[caption] = value
        processed_rows.append(new_row)
    return pd.DataFrame(processed_rows)


def _parse_run_specs(file_path: Path) -> pd.DataFrame:
    """Read and flatten the ``runs-specs.json`` file."""
    if not file_path.is_file():
        return pd.DataFrame()
    with open(file_path, "r", encoding="UTF-8") as fh:
        data = json.load(fh)
    raw_df = pd.DataFrame(data)
    if "metrics" in raw_df.columns:
        return _flatten_run_data(raw_df)
    return raw_df


def _parse_single_sa_summary(file_path: Path) -> Dict[str, List[float]]:
    """Parse a single ``sa_summary.json`` and extract each QoI curve.

    Expected JSON structure::

        {
            "results": [
                {
                    "caption": "PRODUCER - Injector H2S Flow Rate",
                    "id": {
                        "element_id": "_g_PRODUCER",
                        "element_name": "PRODUCER",
                        "property_name": "Injector H2S Flow Rate",
                        "study_id": "project.setup_container.item00001"
                    },
                    "image": [...]
                }
            ]
        }

    Returns:
        ``{qoi_name: [float values]}`` where *qoi_name* is
        ``"{caption} - {study_id}"``.
    """
    if not file_path.is_file():
        return {}

    with open(file_path, "r", encoding="UTF-8") as fh:
        data = json.load(fh)

    curves: Dict[str, List[float]] = {}
    for idx, curve in enumerate(data.get("results", [])):
        caption = curve.get("caption")
        study_id = None
        if "id" in curve and isinstance(curve["id"], dict):
            study_id = curve["id"].get("study_id")

        qoi_name = f"{caption} - {study_id}" if caption and study_id else f"{caption} - {idx}"

        image_data = curve.get("image")
        if image_data is not None:
            curves[qoi_name] = image_data

    return curves


def load_specs_and_qois_for_runs(
    run_specs_path: Path,
    results_base_dir: Path,
    run_numbers_to_load: List[int],
    result_filename: str = "sa_summary.json",
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, List[float]]]]:
    """Load features and QoI targets for a list of run numbers.

    Args:
        run_specs_path: Path to ``runs-specs.json``.
        results_base_dir: Base directory containing ``R_000XX`` sub-folders.
        run_numbers_to_load: Run numbers to load.
        result_filename: Name of the per-run result file.

    Returns:
        A tuple of:
        1. A filtered ``DataFrame`` of features.
        2. A dictionary ``{run_number: {qoi_name: [data]}}``.
    """
    all_features_df = _parse_run_specs(run_specs_path)
    if all_features_df.empty:
        logger.warning("Could not load run specs from '%s'.", run_specs_path)
        return pd.DataFrame(), {}

    features_df = all_features_df[
        all_features_df["run_number"].isin(run_numbers_to_load)
    ].copy()

    qois_per_run: Dict[int, Dict[str, List[float]]] = {}
    project_name = run_specs_path.stem.replace(".runs-specs", "")
    valid_run_numbers: list[int] = []

    for run_number in run_numbers_to_load:
        result_path = results_base_dir / f"{project_name}_R{run_number:05}" / result_filename
        y_qoi_dict = _parse_single_sa_summary(result_path)
        if y_qoi_dict:
            qois_per_run[run_number] = y_qoi_dict
            valid_run_numbers.append(run_number)

    features_df = features_df[
        features_df["run_number"].isin(valid_run_numbers)
    ].reset_index(drop=True)

    logger.info(
        "Loaded specs and QoIs for %d / %d requested runs.",
        len(valid_run_numbers),
        len(run_numbers_to_load),
    )
    return features_df, qois_per_run
