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

def _parse_run_specs(file_path: Path) -> pd.DataFrame:
    """Lê o arquivo 'runs-specs.json' e o achata (flatten)."""
    if not file_path.is_file():
        return pd.DataFrame()
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    raw_df = pd.DataFrame(data)
    if 'metrics' in raw_df.columns:
        return _flatten_run_data(raw_df)
    return raw_df

def _parse_single_sa_summary(file_path: Path) -> Dict[str, List[float]]:
    """
    Lê um único arquivo sa_summary.json e extrai cada curva separadamente.
    
    Estrutura esperada do JSON:
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
    
    Args:
        file_path: Caminho para o arquivo sa_summary.json
        
    Returns:
        Dict com formato: {'caption - study_id': [valores...], ...}
        Exemplo: {'PRODUCER - Injector H2S Flow Rate - project.setup_container.item00001': [...]}
    """
    if not file_path.is_file():
        return {}
    
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    
    curves_dict = {}
    
    for idx, curve in enumerate(data.get('results', [])):
        caption = curve.get('caption')
        study_id = None
        if 'id' in curve and isinstance(curve['id'], dict):
            study_id = curve['id'].get('study_id')
        
        # Constrói o nome único no formato "caption - study_id"
        # f"{qoi.caption} - {qoi.id.study_id}"
        if caption and study_id:
            qoi_name = f"{caption} - {study_id}"
        else:
            qoi_name = f"{caption} - {idx}"
        
        image_data = curve.get('image')
        if image_data is not None:
            curves_dict[qoi_name] = image_data
    
    return curves_dict

def load_specs_and_qois_for_runs(
    run_specs_path: Path, 
    results_base_dir: Path, 
    run_numbers_to_load: List[int],
    result_filename: str = "sa_summary.json"
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, List[float]]]]:
    """
    Lê as especificações (features) e os QoIs (targets) para uma lista
    específica de run_numbers.

    Args:
        run_specs_path (Path): Caminho para o 'runs-specs.json'.
        results_base_dir (Path): Pasta base que contém os diretórios R_000XX.
        run_numbers_to_load (List[int]): A lista de run_numbers para os quais carregar dados.
        result_filename (str): Nome do arquivo de resultado (ex: "sa_summary.json").

    Returns:
        Uma tupla contendo:
        1. um DataFrame de features (filtrado para os run_numbers_to_load).
        2. um Dicionário de QoIs {run_number: {qoi_name: [dados...]}}.
    """
    
    all_features_df = _parse_run_specs(run_specs_path)
    if all_features_df.empty:
        print("AVISO: Não foi possível carregar 'run_specs_path'.")
        return pd.DataFrame(), {}

    features_df = all_features_df[
        all_features_df['run_number'].isin(run_numbers_to_load)
    ].copy()

    qois_per_run: Dict[int, Dict[str, List[float]]] = {}
    project_name = run_specs_path.stem.replace('.runs-specs', '')
    
    valid_run_numbers = []

    for run_number in run_numbers_to_load:
        result_file_path = results_base_dir / f"{project_name}_R{run_number:05}" / result_filename
        
        # _parse_single_sa_summary já retorna um dict {qoi_name: [dados...]}
        # e lida com arquivos não encontrados (retornando {})
        y_qoi_dict = _parse_single_sa_summary(result_file_path)
        
        if y_qoi_dict:
            qois_per_run[run_number] = y_qoi_dict
            valid_run_numbers.append(run_number)
        else:
            continue

    # Garante que as features e os QoIs são consistentes
    # Filtra o dataframe de features para conter APENAS os runs
    # para os quais encontramos um arquivo de QoI.
    features_df = features_df[
        features_df['run_number'].isin(valid_run_numbers)
    ].reset_index(drop=True)
    
    return features_df, qois_per_run