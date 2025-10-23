import json
import pandas as pd
import pytest
from pathlib import Path

from surrogate_factory.parsers.general_parser import load_standard_format
from surrogate_factory.parsers.esss_json_parser import (
    _parse_variable_metadata,
    load_specs_and_qois_for_runs
)

METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field")

REAL_DATA_EXISTS = RUN_DATA_FILE.is_file()


@pytest.mark.skipif(not REAL_DATA_EXISTS, reason="Caminhos de dados reais não encontrados. Pulando teste de integração.")
def test_new_parser_on_real_data():
    """
    Testa o novo parser (load_specs_and_qois_for_runs) contra
    os arquivos de dados reais do projeto.
    """
    runs_to_load = [1, 5]

    metadata = _parse_variable_metadata(METADATA_FILE)
    
    assert metadata is not None
    assert isinstance(metadata, dict)
    assert len(metadata.keys()) > 0

    features_df, qois_per_run = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=runs_to_load,
        result_filename="sa_summary.json"
    )

    assert features_df is not None
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 2
    assert sorted(features_df['run_number'].tolist()) == sorted(runs_to_load)
    assert 'run_number' in features_df.columns


    assert qois_per_run is not None
    assert isinstance(qois_per_run, dict)
    assert len(qois_per_run.keys()) == 2
    assert sorted(list(qois_per_run.keys())) == sorted(runs_to_load)

    assert 1 in qois_per_run
    qoi_dict_run_1 = qois_per_run[1]
    assert isinstance(qoi_dict_run_1, dict)
    assert len(qoi_dict_run_1.keys()) > 0 
    
    first_qoi_name = list(qoi_dict_run_1.keys())[0]
    first_qoi_data = qoi_dict_run_1[first_qoi_name]
    
    assert isinstance(first_qoi_data, list)
    assert len(first_qoi_data) > 0 

    assert 5 in qois_per_run
    qoi_dict_run_5 = qois_per_run[5]
    assert isinstance(qoi_dict_run_5, dict)
    assert len(qoi_dict_run_5.keys()) > 0
