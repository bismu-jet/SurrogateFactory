"""
Módulo parser para ler o formato de dados padrão da biblioteca (CSV + JSON).
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any

def load_standard_format(
    features_path: Path, 
    targets_path: Path,
    run_ids_to_load: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Lê os dados de entrada a partir de um arquivo de features (CSV) e um
    arquivo de targets (JSON).

    Args:
        features_path (Path): Caminho para o arquivo .csv com as features.
        targets_path (Path): Caminho para o arquivo .json com os vetores de saída.
        run_ids_to_load (List[str], opcional): Lista de 'run_id' para carregar.
                                            Se None, carrega todos.

    Returns:
        Uma tupla:
        1. DataFrame de features (filtrado).
        2. Dicionário de QoIs no formato {run_id: {'outputs': [vetor...]}}.
    """
    if not features_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError("Um ou ambos os arquivos de dados (features.csv, targets.json) não foram encontrados.")

    df_features = pd.read_csv(features_path)
    
    df_features['run_id'] = df_features['run_id'].astype(str)

    with open(targets_path, 'r', encoding='UTF-8') as f:
        targets_dict = json.load(f)

    if run_ids_to_load:
        run_ids_to_load = [str(r) for r in run_ids_to_load]
        
        df_features = df_features[df_features['run_id'].isin(run_ids_to_load)].copy()
        targets_dict = {
            run_id: data for run_id, data in targets_dict.items()
            if run_id in run_ids_to_load
        }
    else:
        df_features = df_features[df_features['run_id'].isin(targets_dict.keys())].copy()
    
    qois_per_run_dict = {}
    for run_id, time_series in targets_dict.items():
        if run_id in df_features['run_id'].values:
            qois_per_run_dict[run_id] = {
                'outputs': time_series
            }
    df_features = df_features[df_features['run_id'].isin(qois_per_run_dict.keys())].reset_index(drop=True)

    return df_features, qois_per_run_dict