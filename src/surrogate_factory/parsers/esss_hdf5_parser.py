"""Parser for ESSS HDF5 result files (``result.sourres``).

The JSON ``sa_summary.json`` files only expose the QoIs the sensitivity-analysis
tool was configured to export (here: H2S flow rate).  The raw ``result.sourres``
HDF5, however, stores every logged scalar time-series under
``KaseStudy/LogData/TransientLogCollection_*`` as a definition table
(``LogImageTable``) paired with a value matrix (``LogImageArray`` of shape
``(n_curves, n_timesteps)``).  This module pulls chosen curves (e.g. "Producer
H2S Water Concentration") out of that store so they can be used as surrogate
targets.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .esss_json_parser import _parse_run_specs

logger = logging.getLogger(__name__)

_LOGDATA_GROUP = "KaseStudy/LogData"


def _extract_log_series(h5_path: Path, wanted_names: set[str]) -> dict[str, np.ndarray]:
    """Return ``{qoi_label: series}`` for every transient curve whose name is wanted.

    The label embeds the owner (``"<name> (<owner_type>:<owner_name>)"``) so
    distinct wells/groups sharing a curve name stay separate.
    """
    series_by_label: dict[str, np.ndarray] = {}
    # locking=False avoids spurious Windows lock-violation errors when the file
    # is also open elsewhere (antivirus, viewers) — reads are non-destructive.
    with h5py.File(h5_path, "r", locking=False) as handle:
        if _LOGDATA_GROUP not in handle:
            return series_by_label
        for collection_name in handle[_LOGDATA_GROUP]:
            group = handle[f"{_LOGDATA_GROUP}/{collection_name}"]
            if "LogImageTable" not in group or "LogImageArray" not in group:
                continue
            table = group["LogImageTable"][()]
            values = group["LogImageArray"][()]
            for i, row in enumerate(table):
                name = row["name"].decode("utf-8", "replace")
                if name not in wanted_names:
                    continue
                owner_type = row["owner_type"].decode("utf-8", "replace")
                owner_name = row["owner_name"].decode("utf-8", "replace")
                label = f"{name} ({owner_type}:{owner_name})"
                series_by_label[label] = np.asarray(values[i], dtype=float)
    return series_by_label


def load_specs_and_log_qois_for_runs(
    run_specs_path: Path,
    results_base_dir: Path,
    run_numbers_to_load: list[int],
    quantity_names: list[str],
    result_filename: str = "result.sourres",
) -> tuple[pd.DataFrame, dict[int, dict[str, list[float]]]]:
    """Load features and transient-log QoIs (from HDF5) for a list of runs.

    Mirrors :func:`esss_json_parser.load_specs_and_qois_for_runs` but sources the
    targets from the HDF5 transient logs instead of ``sa_summary.json``.

    Args:
        quantity_names: Exact transient-curve names to extract (e.g.
            ``["Producer H2S Water Concentration"]``).

    Returns:
        ``(features_df, {run_number: {qoi_label: series}})`` for runs that both
        appear in the run specs and yield at least one requested curve.
    """
    all_features_df = _parse_run_specs(run_specs_path)
    if all_features_df.empty:
        logger.warning("Could not load run specs from '%s'.", run_specs_path)
        return pd.DataFrame(), {}

    project_name = run_specs_path.stem.replace(".runs-specs", "")
    wanted = set(quantity_names)
    qois_per_run: dict[int, dict[str, list[float]]] = {}
    valid_run_numbers: list[int] = []

    for run_number in run_numbers_to_load:
        h5_path = results_base_dir / f"{project_name}_R{run_number:05}" / result_filename
        if not h5_path.is_file():
            continue
        try:
            series = _extract_log_series(h5_path, wanted)
        except OSError as exc:
            logger.warning("Could not read '%s' (%s); skipping run %d.", h5_path, exc, run_number)
            continue
        if series:
            qois_per_run[run_number] = {label: arr.tolist() for label, arr in series.items()}
            valid_run_numbers.append(run_number)

    features_df = all_features_df[
        all_features_df["run_number"].isin(valid_run_numbers)
    ].reset_index(drop=True)

    logger.info(
        "Loaded HDF5 log QoIs %s for %d / %d requested runs.",
        quantity_names,
        len(valid_run_numbers),
        len(run_numbers_to_load),
    )
    return features_df, qois_per_run
