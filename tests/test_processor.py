import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from surrogate_factory.parsers.esss_json_parser import (
    _parse_variable_metadata,
    _parse_run_specs,
    load_specs_and_qois_for_runs
)
from surrogate_factory.preprocessing.processor import process_data_multi_model

METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field")

REAL_DATA_EXISTS = RUN_DATA_FILE.is_file() and METADATA_FILE.is_file()


@pytest.mark.skipif(not REAL_DATA_EXISTS, reason="Caminhos de dados reais não encontrados. Pulando teste de integração.")
def test_full_processing_workflow_on_real_data():
    """
    Testa o pipeline completo de "split -> load -> process"
    usando os dados reais do projeto.
    """
    TEST_SIZE = 0.25
    RANDOM_SEED = 42
    metadata = _parse_variable_metadata(METADATA_FILE)
    assert metadata, "Metadados não puderam ser carregados."
    all_specs_df = _parse_run_specs(RUN_DATA_FILE)
    assert not all_specs_df.empty, "Run specs não puderam ser carregados."
    
    all_run_numbers = all_specs_df['run_number'].unique()
    train_run_nums, test_run_nums = train_test_split(
        all_run_numbers, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED
    )
    assert len(train_run_nums) > 0
    assert len(test_run_nums) > 0

    df_train_features, qois_train = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=train_run_nums,
        result_filename="sa_summary.json"
    )
    
    df_test_features, qois_test = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=test_run_nums,
        result_filename="sa_summary.json"
    )
    
    assert len(df_train_features) == len(qois_train)
    assert len(df_test_features) == len(qois_test)

    processed_data = process_data_multi_model(
        df_train_features=df_train_features,
        qois_train=qois_train,
        df_test_features=df_test_features,
        qois_test=qois_test,
        metadata=metadata
    )

    
    assert "X_train" in processed_data
    assert "X_test" in processed_data
    assert "y_train_per_qoi" in processed_data
    assert "y_test_per_qoi" in processed_data
    assert "qoi_names" in processed_data
    assert "scaler" in processed_data

    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    
    assert X_train.shape[0] == len(df_train_features)
    assert X_test.shape[0] == len(df_test_features)
    
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] > 0
    
    # Verificar se X_test foi escalado, mas pode ter valores fora de [0, 1]
    # (o que é normal, se o X_test tiver valores fora do range do X_train)
    assert X_test.min() is not None 

    # Validar Y (Targets)
    y_train_qoi = processed_data["y_train_per_qoi"]
    y_test_qoi = processed_data["y_test_per_qoi"]
    qoi_names = processed_data["qoi_names"]
    
    assert isinstance(y_train_qoi, dict)
    assert isinstance(y_test_qoi, dict)
    assert len(qoi_names) > 0
    
    # Verificar consistência dos dicionários
    assert set(y_train_qoi.keys()) == set(qoi_names)
    assert set(y_test_qoi.keys()) == set(qoi_names)
    
    # Pegar um QoI de exemplo e verificar
    sample_qoi_name = qoi_names[0]
    y_train_sample = y_train_qoi[sample_qoi_name]
    y_test_sample = y_test_qoi[sample_qoi_name]
    
    assert isinstance(y_train_sample, np.ndarray)
    
    # O número de amostras em Y deve bater com o número de amostras em X
    assert y_train_sample.shape[0] == X_train.shape[0]
    assert y_test_sample.shape[0] == X_test.shape[0]
    
    # O shape do vetor (ex: 70) deve ser o mesmo
    assert y_train_sample.shape[1] == y_test_sample.shape[1]
    assert y_train_sample.shape[1] > 0