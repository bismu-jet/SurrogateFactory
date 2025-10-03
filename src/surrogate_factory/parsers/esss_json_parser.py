"""
Módulo parser específico para ler os formatos de arquivo JSON da ESSS
para a criação de modelos surrogados.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

def _parse_variable_metadata(file_path: Path) -> Dict[str, List[str]]:
    """Lê o arquivo JSON de metadados ('input_metrics')."""
    if not file_path.is_file():
        # No final, quando formos criar a biblioteca em si, aqui teremos que usar
        # um log ou um raise
        print(f"AVISO: Arquivo de metadados não encontrado em '{file_path}'.")
        return {}
    
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)

    variable_metadata = {}
    for metric_container in data.get('input_metrics', []):
        for metric_data in metric_container.values():
            caption = metric_data.get('caption')
            valid_values = metric_data.get('valid_values')
            if caption and valid_values:
                variable_metadata[caption] = valid_values
    print(variable_metadata)
    return variable_metadata

def _flatten_run_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma o DataFrame aninhado de 'runs-specs' em um formato tabular."""
    processed_rows = []
    for _, row in df.iterrows():
        new_row = {'run_number': row['run_number']}
        for metric in row['metrics']:
            for metric_data in metric.values():
                caption = metric_data.get('caption')
                value = metric_data.get('value')
                if caption and value is not None:
                    new_row[caption] = value
        processed_rows.append(new_row)
    
    return pd.DataFrame(processed_rows)

def _parse_run_data(file_path: Path) -> pd.DataFrame:
    """Lê o arquivo JSON com os dados de execução e o achata (flatten)."""
    if not file_path.is_file():
        print(f"AVISO: Arquivo de dados de execução não encontrado em '{file_path}'.")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)

    raw_df = pd.DataFrame(data)
    
    # Verifica se formato do runs-specs.json
    if 'metrics' in raw_df.columns:
        return _flatten_run_data(raw_df)
    
    print("detecção para flattening falhou")
    return raw_df

def _parse_results_data(file_path: Path) -> pd.DataFrame:
    """Lê os resultados (alvos 'y') do arquivo de summary global."""
    if not file_path.is_file():
        print(f"AVISO: Arquivo de resultados não encontrado em '{file_path}'.")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    
    results = {}
    # Itera sobre todas as possíveis saídas nos resultados
    for key, value in data.get('results', {}).items():
        if 'run_results' in value:
            results[key] = value['run_results']
            
    return pd.DataFrame(results)

def load_data(metadata_path: Path, run_data_path: Path, results_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Função principal do parser. Orquestra a leitura e junção de todos os
    arquivos de dados da ESSS.

    Args:
        metadata_path (Path): Caminho para o arquivo de metadados (global-sa-input.json).
        run_data_path (Path): Caminho para os dados de entrada dos runs (runs-specs.json).
        results_path (Path): Caminho para os resultados das simulações (global_sa_summary.json).

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[str]]]: Uma tupla contendo:
            - Um DataFrame limpo e unificado com features (X) e targets (y).
            - Um dicionário com os metadados das variáveis categóricas.
    """
    df_features = _parse_run_data(run_data_path)
    df_targets = _parse_results_data(results_path)
    variable_metadata = _parse_variable_metadata(metadata_path)
    
    if not df_features.empty and not df_targets.empty:
        # Define 'run_number' como índice para garantir o alinhamento correto, se existir
        if 'run_number' in df_features.columns:
            df_features = df_features.set_index('run_number')
        
        full_df = pd.concat([df_features, df_targets], axis=1)
        return full_df.reset_index(), variable_metadata

    return pd.DataFrame(), {}