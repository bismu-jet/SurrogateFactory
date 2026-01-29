"""Tests for the ESSS JSON parser (integration – skipped when data is absent)."""

from pathlib import Path

import pandas as pd
import pytest

from surrogate_factory.parsers.esss_json_parser import (
    _parse_variable_metadata,
    load_specs_and_qois_for_runs,
)

# These paths point to real data that only exists in certain environments.
METADATA_FILE = Path(
    r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json"
)
RUN_DATA_FILE = Path(
    r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json"
)
RESULTS_BASE_DIR = Path(
    r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field"
)

REAL_DATA_EXISTS = RUN_DATA_FILE.is_file()


@pytest.mark.skipif(not REAL_DATA_EXISTS, reason="Real data files not available.")
def test_parser_on_real_data():
    """End-to-end parse of ESSS data files."""
    runs_to_load = [1, 5]

    metadata = _parse_variable_metadata(METADATA_FILE)
    assert isinstance(metadata, dict) and len(metadata) > 0

    features_df, qois = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=runs_to_load,
    )

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 2
    assert sorted(features_df["run_number"].tolist()) == sorted(runs_to_load)

    assert len(qois) == 2
    for run_num in runs_to_load:
        assert run_num in qois
        qoi_dict = qois[run_num]
        assert isinstance(qoi_dict, dict) and len(qoi_dict) > 0
        first_series = next(iter(qoi_dict.values()))
        assert isinstance(first_series, list) and len(first_series) > 0
