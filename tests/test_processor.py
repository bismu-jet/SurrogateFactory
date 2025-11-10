import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from surrogate_factory.preprocessing.processor import process_data_multi_model

def test_process_data_multi_model_modular_workflow():
    """
    Testa o fluxo da função process_data_multi_model com dados falsos
    para garantir que os shapes e o scaling modular (X e Y) estão corretos.
    """

    metadata = {'Categorical_Feature': ['A', 'B']}

    df_train_features = pd.DataFrame({
        'run_number': [1, 2, 3, 4, 5],
        'Categorical_Feature': ['A', 'B', 'A', 'B', 'A'],
        'Numeric_Feature': [10, 20, 30, 40, 50] # Range de Treino [10, 50]
    })
    df_val_features = pd.DataFrame({
        'run_number': [6, 7],
        'Categorical_Feature': ['A', 'B'],
        'Numeric_Feature': [15, 25] # Valores DENTRO do range
    })
    df_test_features = pd.DataFrame({
        'run_number': [8, 9, 10],
        'Categorical_Feature': ['B', 'A', 'A'],
        'Numeric_Feature': [5, 60, 30] # Valores FORA do range (5 e 60)
    })
    
    # --- Targets (Y) ---
    # (Duas QoIs, com 3 timesteps cada. QOI_1 range [100, 200])
    qois_train = {
        1: {'QOI_1': [100, 110, 120], 'QOI_2': [1, 2, 3]},
        2: {'QOI_1': [110, 110, 120], 'QOI_2': [1, 2, 3]},
        3: {'QOI_1': [120, 110, 120], 'QOI_2': [1, 2, 3]},
        4: {'QOI_1': [130, 110, 120], 'QOI_2': [1, 2, 3]},
        5: {'QOI_1': [200, 210, 220], 'QOI_2': [1, 2, 3]}, # Max
    }
    qois_val = {
        6: {'QOI_1': [150, 160, 170], 'QOI_2': [1, 2, 3]}, # DENTRO do range
        7: {'QOI_1': [160, 160, 170], 'QOI_2': [1, 2, 3]},
    }
    # Teste de QoI Faltante: Run 10 não tem QOI_2.
    qois_test = {
        8: {'QOI_1': [120, 130, 140], 'QOI_2': [1, 2, 3]}, # DENTRO
        9: {'QOI_1': [300, 310, 320], 'QOI_2': [1, 2, 3]}, # FORA (Acima)
        10: {'QOI_1': [50, 60, 70]}                        # FORA (Abaixo) e QOI_2 faltante
    }

    processed_data = process_data_multi_model(
        df_train_features=df_train_features,
        qois_train=qois_train,
        df_val_features=df_val_features,
        qois_val=qois_val,
        df_test_features=df_test_features,
        qois_test=qois_test,
        metadata=metadata
    )

    assert len(processed_data['qoi_names']) == 1
    assert processed_data['qoi_names'][0] == 'QOI_1'

    # --- Verificar Shapes (X) ---
    assert processed_data['X_train_raw'].shape == (5, 2)
    assert processed_data['X_val_raw'].shape == (2, 2)
    assert processed_data['X_test_raw'].shape == (3, 2)
    assert processed_data['X_train_scaled'].shape == (5, 2)
    
    # --- Verificar Shapes (Y) ---
    y_train_qoi1 = processed_data['y_train_per_qoi']['QOI_1']
    y_val_qoi1 = processed_data['y_val_per_qoi']['QOI_1']
    y_test_qoi1 = processed_data['y_test_per_qoi']['QOI_1']
    
    assert y_train_qoi1.shape == (5, 3) # (5 amostras, 3 timesteps)
    assert y_val_qoi1.shape == (2, 3)
    assert y_test_qoi1.shape == (3, 3) # 3 amostras, 3 timesteps

    # --- Verificar Scaling (X) ---
    x_train_scaled_num_col = processed_data['X_train_scaled'][:, 1]
    x_val_scaled_num_col = processed_data['X_val_scaled'][:, 1]
    x_test_scaled_num_col = processed_data['X_test_scaled'][:, 1]
    
    assert np.isclose(np.min(x_train_scaled_num_col), 0.0) # Treino Min é 0
    assert np.isclose(np.max(x_train_scaled_num_col), 1.0) # Treino Max é 1
    
    assert np.all(x_val_scaled_num_col >= 0.0) & np.all(x_val_scaled_num_col <= 1.0) # Val está DENTRO
    
    assert np.min(x_test_scaled_num_col) < 0.0 # Teste Min (5) é < 0
    assert np.max(x_test_scaled_num_col) > 1.0 # Teste Max (60) é > 1

    # --- Verificar Scaling (Y) ---
    assert np.isclose(np.min(y_train_qoi1), 0.0)
    assert np.isclose(np.max(y_train_qoi1), 1.0)
    
    assert np.all(y_val_qoi1 >= 0.0) & np.all(y_val_qoi1 <= 1.0) # Val está DENTRO
    
    assert np.min(y_test_qoi1) < 0.0 # Teste Min (50) é < 0
    assert np.max(y_test_qoi1) > 1.0 # Teste Max (300) é > 1
    
    # --- Verificar Scalers ---
    assert 'x_scaler' in processed_data
    assert isinstance(processed_data['x_scaler'], MinMaxScaler)
    assert 'y_scalers' in processed_data
    assert 'QOI_1' in processed_data['y_scalers']
    assert isinstance(processed_data['y_scalers']['QOI_1'], MinMaxScaler)