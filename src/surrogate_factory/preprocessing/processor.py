"""
Módulo de pré-processamento de dados para a arquitetura multi-modelo.

Este módulo substitui o 'processor.py' original.
A lógica segue o "Project 2":
1. O split de dados (train/test) é feito ANTES (no pipeline).
2. O encoding das features é feito manualmente (sem LabelEncoder).
3. O scaling é feito DEPOIS do split, fitando-se apenas nos dados de treino.
4. Os dados de Y (QoIs) são reestruturados por QoI, não por run.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Set
from sklearn.preprocessing import MinMaxScaler

def _build_feature_array(
    features_df: pd.DataFrame, 
    metadata: Dict[str, List[str]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Converte o DataFrame de features em um array numérico,

    Args:
        features_df: O DataFrame de features (ex: df_train_features).
        metadata: O dicionário de metadados com as variáveis categóricas.

    Returns:
        Uma tupla contendo:
        1. O array numpy (X_raw) com os dados prontos para o scaler.
        2. A lista de nomes de features na ordem em que foram processadas.
    """
    
    categorical_cols = list(metadata.keys())
    all_feature_names = [col for col in features_df.columns if col != 'run_number']
    ordered_feature_names = [col for col in all_feature_names if col in categorical_cols]
    numeric_cols = [col for col in all_feature_names if col not in categorical_cols]
    ordered_feature_names.extend(numeric_cols)

    processed_rows = []
    
    for row in features_df[ordered_feature_names].itertuples(index=False):
        new_row_values = []
        for i, value in enumerate(row):
            col_name = ordered_feature_names[i]
            
            if col_name in categorical_cols:
                try:
                    valid_values = metadata[col_name]
                    idx = valid_values.index(value)
                    new_row_values.append(float(idx))
                except ValueError:
                    print(f"AVISO: Valor '{value}' não encontrado no metadata de '{col_name}'. Usando -1.0")
                    new_row_values.append(-1.0)
            else:
                new_row_values.append(float(value))
        
        processed_rows.append(new_row_values)
        
    return np.array(processed_rows), ordered_feature_names


def _restructure_qois(
    qois_per_run: Dict[int, Dict[str, List[float]]],
    run_numbers: List[int]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Transforma o dicionário de QoIs de [run][qoi] para [qoi][runs].
    
    Args:
        qois_per_run: O dicionário vindo do parser {run_num: {qoi_name: [dados]}}.
        run_numbers: A lista de run_numbers a serem incluídos (ex: train_runs).

    Returns:
        Uma tupla contendo:
        1. O dicionário de targets {qoi_name: np.array([[...], [...]])}.
        2. A lista de todos os nomes de QoIs encontrados.
    """
    if not qois_per_run:
        return {}, []

    all_qoi_names_set: Set[str] = set()
    for qoi_dict in qois_per_run.values():
        all_qoi_names_set.update(qoi_dict.keys())
    
    all_qoi_names = sorted(list(all_qoi_names_set))
    y_per_qoi: Dict[str, List[np.ndarray]] = {qoi_name: [] for qoi_name in all_qoi_names}
    
    for run_num in run_numbers:
        if run_num not in qois_per_run:
            continue 
            
        run_qoi_data = qois_per_run[run_num]
        
        for qoi_name in all_qoi_names:
            qoi_vector = run_qoi_data.get(qoi_name)
            
            if qoi_vector is not None:
                y_per_qoi[qoi_name].append(np.array(qoi_vector))
            else:
                del y_per_qoi[qoi_name]
                all_qoi_names.remove(qoi_name)

    final_y_per_qoi: Dict[str, np.ndarray] = {}
    for qoi_name, data_list in y_per_qoi.items():
        if data_list:
            final_y_per_qoi[qoi_name] = np.array(data_list)
            
    final_qoi_names = sorted(list(final_y_per_qoi.keys()))
            
    return final_y_per_qoi, final_qoi_names


def process_data_multi_model(
    df_train_features: pd.DataFrame,
    qois_train: Dict[int, Dict[str, List[float]]],
    df_test_features: pd.DataFrame,
    qois_test: Dict[int, Dict[str, List[float]]],
    metadata: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Função principal de pré-processamento para o fluxo multi-modelo.
    Assume que os dados já foram divididos em train e test.
    """
    
    X_train_raw, feature_names = _build_feature_array(df_train_features, metadata)
    X_test_raw, _ = _build_feature_array(df_test_features[df_train_features.columns], metadata)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    
    train_run_numbers = df_train_features['run_number'].tolist()
    test_run_numbers = df_test_features['run_number'].tolist()

    y_train_per_qoi, train_qoi_names = _restructure_qois(qois_train, train_run_numbers)
    y_test_per_qoi, test_qoi_names = _restructure_qois(qois_test, test_run_numbers)

    valid_qoi_names = sorted(list(set(train_qoi_names) & set(test_qoi_names)))
    
    final_y_train = {qoi: y_train_per_qoi[qoi] for qoi in valid_qoi_names}
    final_y_test = {qoi: y_test_per_qoi[qoi] for qoi in valid_qoi_names}
    

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train_per_qoi": final_y_train,
        "y_test_per_qoi": final_y_test,
        "scaler": scaler,
        "qoi_names": valid_qoi_names,
        "feature_names": feature_names
    }