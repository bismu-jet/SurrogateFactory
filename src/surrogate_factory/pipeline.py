import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Novos imports (estilo Project 2) ---
from parsers.esss_json_parser import (
    _parse_variable_metadata, 
    _parse_run_specs, 
    load_specs_and_qois_for_runs,
    _parse_single_sa_summary # Já estava no pipeline original
)
from preprocessing.processor import process_data_multi_model

# --- Imports mantidos ---
from models.neural_network import build_model
from evaluation.plotting import plot_feature_distribution, plot_target_timeseries
from evaluation.plotting import plot_comparison_timeseries

print("--- INICIANDO PIPELINE DE TREINAMENTO MULTI-MODELO (Estilo Project 2) ---")

# --- 1. Definições de Caminhos e Parâmetros ---
METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field") 

# Parâmetros para o split de dados
TEST_SIZE = 0.2
RANDOM_STATE = 42

try:
    print("\n--- Etapa 1: Carregando Specs e Metadados ---")
    metadata = _parse_variable_metadata(METADATA_FILE)
    all_specs_df = _parse_run_specs(RUN_DATA_FILE)
    
    all_run_numbers = all_specs_df['run_number'].unique()
    print(f"Total de runs encontrados: {len(all_run_numbers)}")

    print(f"\n--- Etapa 2: Dividindo os Run Numbers (Test size={TEST_SIZE}) ---")
    train_run_nums, test_run_nums = train_test_split(
        all_run_numbers, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    print(f"{len(train_run_nums)} runs para treino, {len(test_run_nums)} runs para teste.")

    print("\n--- Etapa 3: Carregando Dados de Treino e Teste ---")
    
    print("Carregando dados de TREINO...")
    df_train_features, qois_train = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=train_run_nums,
        result_filename="sa_summary.json"
    )
    
    print("Carregando dados de TESTE...")
    df_test_features, qois_test = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=test_run_nums,
        result_filename="sa_summary.json"
    )

    print("\n--- Etapa 4: Pré-processando os Dados (Estilo Project 2) ---")
    processed_data = process_data_multi_model(
        df_train_features=df_train_features,
        qois_train=qois_train,
        df_test_features=df_test_features,
        qois_test=qois_test,
        metadata=metadata
    )

    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    y_train_per_qoi = processed_data["y_train_per_qoi"]
    y_test_per_qoi = processed_data["y_test_per_qoi"]
    qoi_names = processed_data["qoi_names"]
    
    test_indices_map = pd.Series(
        df_test_features['run_number'].values, 
        index=range(len(df_test_features))
    )

    print(f"Dados prontos. {len(qoi_names)} QoIs encontrados.")
    print(f"Formato de X_train: {X_train.shape}, Formato de X_test: {X_test.shape}")

    
    print("\n--- Etapa 5: Treinamento Multi-Modelo (Loop por QoI) ---")
    
    trained_models = {}

    for qoi_name in qoi_names:
        print(f"\n--- Treinando Modelo para o QoI: {qoi_name} ---")
        
        y_train_qoi = y_train_per_qoi[qoi_name]
        y_test_qoi = y_test_per_qoi[qoi_name]
        
        n_features = X_train.shape[1]
        n_outputs = y_train_qoi.shape[1] 
        
        print(f"Arquitetura: {n_features} features -> {n_outputs} saídas")
        
        surrogate_model = build_model(input_shape=n_features, output_shape=n_outputs)
        if qoi_name == qoi_names[0]:
            surrogate_model.summary() 

        print("Iniciando o Treinamento ▶️")
        history = surrogate_model.fit(
            X_train,
            y_train_qoi,
            epochs=10,
            batch_size=2,
            validation_data=(X_test, y_test_qoi),
            verbose=1
        )
        
        loss = surrogate_model.evaluate(X_test, y_test_qoi, verbose=0)
        print(f"Perda (Loss) final no teste para '{qoi_name}': {loss:.4f}")
        
        trained_models[qoi_name] = surrogate_model

    print("\n--- Treinamento de todos os modelos concluído ---")


    print("\n--- Etapa 6: Avaliação e Comparação Visual (Loop por QoI) ---")
        
    if len(X_test) > 0:
        # Pega o primeiro run do nosso conjunto de teste para a comparação
        sample_to_compare_idx = 0
        run_number_to_compare = test_indices_map[sample_to_compare_idx]
        
        # Prepara o X_sample (é o mesmo para todos os modelos)
        x_sample = X_test[[sample_to_compare_idx]]

        # Carrega o arquivo do modelo antigo UMA VEZ
        project_name = RUN_DATA_FILE.stem.replace('.runs-specs', '')
        old_surrogate_path = RESULTS_BASE_DIR / f"{project_name}_R{run_number_to_compare:05}" / "surrogate_summary.json"
        
        print(f"Carregando dados do modelo antigo do Run #{run_number_to_compare} de '{old_surrogate_path}'...")
        y_pred_old_dict = _parse_single_sa_summary(old_surrogate_path)
        
        # Faz a predição e plota para CADA modelo treinado
        for qoi_name in qoi_names:
            print(f"Gerando gráfico de comparação para o QoI: {qoi_name}...")
            
            # Pega o modelo e os dados Y corretos
            model = trained_models[qoi_name]
            y_true = y_test_per_qoi[qoi_name][sample_to_compare_idx]
            
            # Previsão do nosso novo modelo (MLP)
            y_pred_new = model.predict(x_sample, verbose=0)[0] # Adicionado verbose=0
            
            # Previsão do modelo antigo (RBF)
            y_pred_old = y_pred_old_dict.get(qoi_name)

            # Gera o gráfico
            if y_pred_old:
                # --- CHAMADA CORRIGIDA ---
                plot_comparison_timeseries(
                    y_true=y_true, 
                    y_pred_new=y_pred_new, 
                    y_pred_old=y_pred_old, 
                    run_number=run_number_to_compare,  # Passa o número
                    qoi_name=qoi_name                   # Passa o nome
                )
            else:
                print(f"AVISO: Não foi possível encontrar o QoI '{qoi_name}' no 'surrogate_summary.json' (modelo antigo).")

    print("\n--- PIPELINE CONCLUÍDA COM SUCESSO ---")

except Exception as e:
    print(f"\n--- ERRO NA PIPELINE: {e} ---")
    import traceback
    traceback.print_exc()