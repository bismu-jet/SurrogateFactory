"""
Módulo de pré-processamento de dados para a arquitetura multi-modelo.
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
    imitando a lógica de '_GetPointsFromRun' do Project 2.
    """
    
    categorical_cols = list(metadata.keys())
    
    # Pega todas as colunas que não são 'run_number'
    cols_to_ignore = ['run_id', 'run_number']
    all_feature_names = [col for col in features_df.columns if col not in cols_to_ignore]
    
    # Garante uma ordem consistente: categóricas primeiro, depois numéricas
    ordered_feature_names = [col for col in all_feature_names if col in categorical_cols]
    numeric_cols = [col for col in all_feature_names if col not in categorical_cols]
    ordered_feature_names.extend(numeric_cols)

    processed_rows = []
    
    # itertuples é mais rápido que iterrows
    for row in features_df[ordered_feature_names].itertuples(index=False):
        new_row_values = []
        for i, value in enumerate(row):
            col_name = ordered_feature_names[i]
            
            if col_name in categorical_cols:
                # Lógica do Project 2: converter valor categórico em seu índice
                try:
                    valid_values = metadata[col_name]
                    idx = valid_values.index(value)
                    new_row_values.append(float(idx))
                except ValueError:
                    print(f"AVISO: Valor '{value}' não encontrado no metadata de '{col_name}'. Usando -1.0")
                    new_row_values.append(-1.0)
            else:
                # Lógica do Project 2: apenas converte para float
                new_row_values.append(float(value))
        
        processed_rows.append(new_row_values)
        
    # Retorna os dados crus (raw)
    return np.array(processed_rows), ordered_feature_names


def _restructure_qois(
    qois_per_run: Dict[int, Dict[str, List[float]]],
    run_ids: List[int]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Transforma o dicionário de QoIs de [run][qoi] para [qoi][runs].
    """
    if not qois_per_run:
        return {}, []

    # 1. Encontrar o super-conjunto de todos os nomes de QoIs
    all_qoi_names_set: Set[str] = set()
    for qoi_dict in qois_per_run.values():
        all_qoi_names_set.update(qoi_dict.keys())
    
    all_qoi_names = sorted(list(all_qoi_names_set))
    
    # 2. Inicializar o dicionário de saída
    y_per_qoi: Dict[str, List[np.ndarray]] = {qoi_name: [] for qoi_name in all_qoi_names}
    
    # 3. Rastrear QoIs que se mostrarem inválidos
    valid_qoi_names_final = set(all_qoi_names)

    # 4. Coletar os dados
    for run_id in run_ids:
        run_id_str = str(run_id)
        if run_id_str not in qois_per_run:
            if isinstance(run_id,int) and run_id in qois_per_run:
                run_id_str = run_id
            else:
                continue
            
        run_qoi_data = qois_per_run[run_id_str]
        
        for qoi_name in all_qoi_names:
            if qoi_name not in valid_qoi_names_final:
                continue # Pula QoI já invalidado
                
            qoi_vector = run_qoi_data.get(qoi_name)
            
            if qoi_vector is not None:
                # Adiciona o vetor de dados
                y_per_qoi[qoi_name].append(np.array(qoi_vector))
            elif qoi_name in valid_qoi_names_final:
                # Se um run não tem um QoI, esse QoI é inválido para todos
                print(f"AVISO: QoI '{qoi_name}' não encontrado no Run #{run_id}. Excluindo este QoI do dataset.")
                valid_qoi_names_final.remove(qoi_name)
                del y_per_qoi[qoi_name] # Remove dos dados de saída

    # 5. Converter listas para arrays numpy 2D e filtrar inconsistências
    final_y_per_qoi: Dict[str, np.ndarray] = {}
    for qoi_name in valid_qoi_names_final:
        data_list = y_per_qoi[qoi_name]
        if data_list:
             try:
                # Tenta empilhar os arrays. Falhará se tiverem tamanhos diferentes.
                stacked_array = np.array(data_list)
                if stacked_array.ndim == 1: # numpy cria array 1D se os tamanhos forem inconsistentes
                     raise ValueError("Tamanhos de vetor inconsistentes.")
                final_y_per_qoi[qoi_name] = stacked_array
             except ValueError:
                 print(f"AVISO: QoI '{qoi_name}' tem vetores de tamanhos inconsistentes entre runs. Excluindo.")
            
    final_qoi_names = sorted(list(final_y_per_qoi.keys()))
            
    return final_y_per_qoi, final_qoi_names

def _get_id_list(df: pd.DataFrame) -> List[Any]:
    if 'run_id' in df.columns:
        return df['run_id'].tolist()
    elif 'run_number' in df.columns:
        return df['run_number'].tolist()
    else:
        return[]


def process_data_multi_model(
    df_train_features: pd.DataFrame,
    qois_train: Dict[int, Dict[str, List[float]]],
    df_val_features: pd.DataFrame,
    qois_val:Dict[int, Dict[str, List[float]]],
    df_test_features: pd.DataFrame,
    qois_test: Dict[int, Dict[str, List[float]]],
    metadata: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Função principal de pré-processamento para o fluxo multi-modelo.
    O SCALER FOI REMOVIDO para ser compatível com o RBF.
    """
    
    # --- 1. Processar Features (X) ---
    print("Processando features de treino (X_train)...")
    X_train_raw, feature_names = _build_feature_array(df_train_features, metadata)

    print("Processando features de validação (X_val)...")
    X_val_raw, _ = _build_feature_array(df_val_features[df_train_features.columns], metadata)
    
    print("Processando features de teste (X_test)...")
    # Garante que o df_test tenha as colunas na mesma ordem
    X_test_raw, _ = _build_feature_array(df_test_features[df_train_features.columns], metadata)

    # --- 2. Escalonar Features (X) ---
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_val_scaled = x_scaler.transform(X_val_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)
    
    print(f"Processamento de X concluído. Shape X_train: {X_train_raw.shape}")
    if X_train_raw.shape[0] > 0:
        print(f"Exemplo de dados X_train (primeira linha, crus): {X_train_raw[0]}")
        print(f"Exemplo de dados X_train (primeira linha, escalada): {X_train_scaled[0]}")


    # --- 3. Processar Targets (Y) ---
    print("Reestruturando dados de target (Y) por QoI...")
    
    train_run_ids = _get_id_list(df_train_features)
    val_run_ids = _get_id_list(df_val_features)
    test_run_ids = _get_id_list(df_test_features)

    y_train_per_qoi, train_qoi_names = _restructure_qois(qois_train, train_run_ids)
    y_val_per_qoi, val_qoi_names = _restructure_qois(qois_val, val_run_ids)
    y_test_per_qoi, test_qoi_names = _restructure_qois(qois_test, test_run_ids)

    # --- 4. Garantir Consistência de QoIs ---
    # Garante que ambos os conjuntos (treino e teste) tenham os mesmos QoIs
    valid_qoi_names = sorted(list(set(train_qoi_names) & set(test_qoi_names) & set(val_qoi_names)))
    
    final_y_train = {qoi: y_train_per_qoi[qoi] for qoi in valid_qoi_names if qoi in y_train_per_qoi}
    final_y_val = {qoi: y_val_per_qoi[qoi] for qoi in valid_qoi_names if qoi in y_val_per_qoi}
    final_y_test = {qoi: y_test_per_qoi[qoi] for qoi in valid_qoi_names if qoi in y_test_per_qoi}
    
    # Filtra novamente caso o restructure tenha removido algo
    final_valid_qoi_names = sorted(list(final_y_train.keys()))

    y_scalers = {}
    final_y_train_scaled = {}
    final_y_val_scaled = {}
    final_y_test_scaled = {}

    for qoi in final_valid_qoi_names:
        y_scaler = MinMaxScaler()
        y_train_flat = final_y_train[qoi].flatten().reshape(-1,1)
        y_scaler.fit(y_train_flat)
        y_scalers[qoi] = y_scaler

        final_y_train_scaled[qoi] = y_scaler.transform(y_train_flat).reshape(final_y_train[qoi].shape)

        y_val_flat = final_y_val[qoi].flatten().reshape(-1, 1)
        final_y_val_scaled[qoi] = y_scaler.transform(y_val_flat).reshape(final_y_val[qoi].shape)

        y_test_flat = final_y_test[qoi].flatten().reshape(-1, 1)
        final_y_test_scaled[qoi] = y_scaler.transform(y_test_flat).reshape(final_y_test[qoi].shape)
    
    print(f"Processamento concluído. Encontrados {len(final_valid_qoi_names)} QoIs consistentes.")

    return {
        # --- Features (X) ---
        "X_train_raw": X_train_raw,         # Dados crus
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "X_train_scaled": X_train_scaled,   # Dados escalados [0, 1]
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "x_scaler": x_scaler,               # O scaler de features (fitado)
        
        # --- Targets (Y) ---
        "y_train_per_qoi": final_y_train_scaled,
        "y_val_per_qoi": final_y_val_scaled,
        "y_test_per_qoi": final_y_test_scaled,
        "y_scalers": y_scalers,             # Dicionário de scalers de targets
        
        # --- Metadados ---
        "qoi_names": final_valid_qoi_names,
        "feature_names": feature_names
    }