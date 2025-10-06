"""
Módulo parser para ler o formato de dados padrão da biblioteca (CSV + JSON).
"""
import json
import pandas as pd
from pathlib import Path

def load_standard_format(features_path: Path, targets_path: Path) -> pd.DataFrame:
    """
    Lê os dados de entrada a partir de um arquivo de features (CSV) e um
    arquivo de targets (JSON) e os combina em um único DataFrame.

    Args:
        features_path (Path): Caminho para o arquivo .csv com as features de entrada.
        targets_path (Path): Caminho para o arquivo .json com os vetores de saída.

    Returns:
        Um DataFrame unificado com features e a coluna 'outputs'.
    """
    if not features_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError("Um ou ambos os arquivos de dados (features.csv, targets.json) não foram encontrados.")

    df_features = pd.read_csv(features_path)

    with open(targets_path, 'r', encoding='UTF-8') as f:
        targets_dict = json.load(f)

    df_features['outputs'] = df_features['run_id'].astype(str).map(targets_dict)
    
    df_features = df_features.dropna(subset=['outputs']).copy()

    return df_features