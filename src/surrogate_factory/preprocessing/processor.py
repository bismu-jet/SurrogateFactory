"""
Módulo de pré-processamento de dados. Transforma o DataFrame limpo em
conjuntos de treino e teste prontos para o modelo.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def process_data(
    df: pd.DataFrame, 
    metadata: Dict[str, List[str]], 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Função principal de pré-processamento. Executa o encoding, escalonamento
    e divisão dos dados.

    Args:
        df (pd.DataFrame): O DataFrame unificado vindo do parser, com a coluna 'outputs'.
        metadata (Dict[str, List[str]]): O dicionário de metadados com as variáveis categóricas.
        test_size (float): A proporção do dataset a ser usada para teste.
        random_state (int): Seed para reprodutibilidade da divisão.

    Returns:
        Um dicionário contendo os dados processados e os artefatos.
    """
    if df.empty:
        raise ValueError("O DataFrame de entrada está vazio.")

    # 1. Separar features (X) e targets (y) --- LÓGICA ALTERADA ---
    # 'y' agora é a coluna 'outputs', convertida para um array NumPy.
    y = np.array(df['outputs'].tolist())
    
    # 'X' são todas as outras colunas, exceto 'run_number' e 'outputs'.
    feature_cols = [col for col in df.columns if col not in ['run_number', 'outputs']]
    X = df[feature_cols].copy()

    # 2. Encoding de variáveis categóricas (Lógica inalterada)
    categorical_cols = list(metadata.keys())
    encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            encoders[col] = encoder

    # 3. Escalonamento das features (Lógica inalterada)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # 4. Divisão em treino e teste (Lógica inalterada, agora com o 'y' correto)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state
    )

    # 5. Retorna os dados e os "artefatos"
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler
    }