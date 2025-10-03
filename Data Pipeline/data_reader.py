import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# As funções parse_variable_metadata e parse_run_data continuam iguais...

def parse_variable_metadata(file_path: Path) -> Dict[str, List[str]]:
    """Lê o arquivo JSON de metadados."""
    print(f"--- Analisando Metadados de '{file_path.name}' ---")
    print(f"Procurando pelo arquivo em: {file_path.resolve()}")
    if not file_path.is_file():
        print(f"AVISO: Arquivo de metadados não encontrado.")
        return {}
    try:
        with open(file_path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        variable_metadata = {}
        for metric_container in data.get('input_metrics', []):
            for metric_data in metric_container.values():
                caption = metric_data.get('caption')
                valid_values = metric_data.get('valid_values')
                if caption and valid_values:
                    variable_metadata[caption] = valid_values
        return variable_metadata
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERRO: Não foi possível analisar o arquivo de metadados. Erro: {e}")
        return {}

def parse_run_data(file_path: Path) -> pd.DataFrame:
    """Lê o arquivo JSON com os dados de execução."""
    print(f"\n--- Analisando Dados de Execução de '{file_path.name}' ---")
    print(f"Procurando pelo arquivo em: {file_path.resolve()}")
    if not file_path.is_file():
        print(f"AVISO: Arquivo de dados de execução não encontrado.")
        return pd.DataFrame()
    try:
        with open(file_path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        run_data = []
        if isinstance(data, dict):
            run_data = data.get('samples', [])
        elif isinstance(data, list):
            run_data = data
        return pd.DataFrame(run_data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERRO: Não foi possível analisar o arquivo de dados de execução. Erro: {e}")
        return pd.DataFrame()

# --- NOVA FUNÇÃO DE TRANSFORMAÇÃO ---
def flatten_run_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame aninhado em um formato tabular (achatado),
    onde cada métrica vira uma coluna.
    """
    processed_rows = []
    # Itera sobre cada linha do DataFrame original
    for index, row in df.iterrows():
        # Começa um novo dicionário para a linha processada com o número do run
        new_row = {'run_number': row['run_number']}
        # Itera sobre a lista de métricas daquela linha
        for metric in row['metrics']:
            # O conteúdo está dentro de uma chave como 'DiscreteMetric' ou 'ContinuousMetric'
            for metric_data in metric.values():
                caption = metric_data.get('caption')
                value = metric_data.get('value')
                if caption and value is not None:
                    new_row[caption] = value
        processed_rows.append(new_row)
    
    return pd.DataFrame(processed_rows)


if __name__ == "__main__":
    # Mantém os nomes dos arquivos como estão
    METADATA_FILE = Path("esss 3phase lgr field.global-sa-input.json") 
    RUN_DATA_FILE = Path("esss 3phase lgr field.runs-specs.json")

    # 1. Parse dos Metadados
    metadata = parse_variable_metadata(METADATA_FILE)
    if metadata:
        print("Metadados das Variáveis Discretas Encontradas:")
        for name, values in metadata.items():
            print(f"  - {name}: {values}")
    
    # 2. Parse dos Dados de Execução
    raw_run_data_df = parse_run_data(RUN_DATA_FILE)
    
    # 3. Transformação (Flatten) dos Dados
    if not raw_run_data_df.empty:
        print("\n--- Transformando Dados de Execução ---")
        clean_run_data_df = flatten_run_data(raw_run_data_df)
        
        print("\nDados das 5 Primeiras Execuções (Formato Final):")
        print(clean_run_data_df.head())
    else:
        print("Nenhum dado de execução foi extraído.")