import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*multiple x input features have the same value.*")
# Ignora o aviso de divisão por zero do Kriging (lidamos com isso no try/except)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered in log10.*")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from parsers.esss_json_parser import (
    _parse_variable_metadata, 
    _parse_run_specs, 
    load_specs_and_qois_for_runs,
    _parse_single_sa_summary
)
from preprocessing.processor import process_data_multi_model

from models.rbf_model import RBFVectorModel
from models.kriging_model import KrigingVectorModel
from models.neural_network import NeuralNetworkModel

from evaluation.plotting import plot_feature_distribution, plot_target_timeseries
from evaluation.plotting import plot_comparison_timeseries

print("--- INICIANDO PIPELINE DE TREINAMENTO MULTI-MODELO (Estilo Project 2) ---")

METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field") 

TEST_SIZE = 0.2
RANDOM_STATE = 41

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
    val_split_size = 0.125 
    train_run_nums, val_run_nums = train_test_split(
        train_run_nums, # Divide o que sobrou do treino
        test_size=val_split_size, 
        random_state=RANDOM_STATE
    )
    print(f"{len(train_run_nums)} runs para treino, {len(val_run_nums)} runs para validação, {len(test_run_nums)} runs para teste.")

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

    print("Carregando dados de VALIDAÇÃO...")
    df_val_features, qois_val = load_specs_and_qois_for_runs(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        run_numbers_to_load=val_run_nums,
        result_filename="sa_summary.json"
    )
    print("\n" + "="*30)
    print("  INVESTIGAÇÃO: Verificando Dados CRUS (Etapa 3)")
    
    # Bloco dinâmico para encontrar um Run e QoI de teste
    if qois_test:
        RUN_PARA_TESTAR = list(qois_test.keys())[0] # Pega o primeiro run_num do dict de teste
        QOI_PARA_TESTAR = list(qois_test[RUN_PARA_TESTAR].keys())[0] # Pega o primeiro QoI desse run
    
    if RUN_PARA_TESTAR is not None:
        try:
            # Pega o dado CRU do dicionário carregado pelo parser
            raw_vector = qois_test[RUN_PARA_TESTAR][QOI_PARA_TESTAR]
            print(f"  QoI: {QOI_PARA_TESTAR}")
            print(f"  Run: {RUN_PARA_TESTAR} (Primeiro run encontrado no TEST set)")
            print(f"  Média do Vetor CRU: {np.mean(raw_vector)}")
            print(f"  Max do Vetor CRU:   {np.max(raw_vector)}")
            print(f"  Min do Vetor CRU:   {np.min(raw_vector)}")
        except KeyError:
            print(f"  AVISO: Falha ao inspecionar Run {RUN_PARA_TESTAR} ou QoI '{QOI_PARA_TESTAR}'.")
    else:
        print("  AVISO: `qois_test` está vazio. Nenhum run de teste foi carregado.")
    print("="*30 + "\n")

    print("\n--- Etapa 4: Pré-processando os Dados (Estilo Project 2) ---")
    processed_data = process_data_multi_model(
        df_train_features=df_train_features,
        qois_train=qois_train,
        df_test_features=df_test_features,
        df_val_features = df_val_features,
        qois_val = qois_val,
        qois_test=qois_test,
        metadata=metadata
    )

    X_train_raw = processed_data["X_train_raw"]
    X_val_raw = processed_data["X_val_raw"]
    X_test_raw = processed_data["X_test_raw"]

    X_train_scaled = processed_data["X_train_scaled"]
    X_val_scaled = processed_data["X_val_scaled"]
    X_test_scaled = processed_data["X_test_scaled"]

    x_scaler = processed_data["x_scaler"]
    y_scalers = processed_data["y_scalers"]

    y_train_per_qoi = processed_data["y_train_per_qoi"]
    y_val_per_qoi = processed_data["y_val_per_qoi"]
    y_test_per_qoi = processed_data["y_test_per_qoi"]

    qoi_names = processed_data["qoi_names"]
    
    test_indices_map = pd.Series(
        df_test_features['run_number'].values, 
        index=range(len(df_test_features))
    )

    print(f"Dados prontos. {len(qoi_names)} QoIs encontrados.")
    print(f"Formato de X_train: {X_train_raw.shape}, Formato de X_test: {X_test_raw.shape}, Formato de X_val: {X_val_raw.shape}")

    
    print("\n--- Etapa 5: Treinamento Multi-Modelo (Kriging c/ Tuning) ---")
    
    trained_models = {}

    for qoi_name in qoi_names:
        print(f"\n--- Treinando Modelo Kriging para o QoI: {qoi_name} ---")
        
        y_train_qoi = y_train_per_qoi[qoi_name]
        y_test_qoi = y_test_per_qoi[qoi_name]

        print(f"  [DIAG] Y_TRAIN (Scaled) Min: {np.min(y_train_qoi):.4f}, Max: {np.max(y_train_qoi):.4f}")
        
        try:
            # Nota: X_train e X_test já estão escalados (MinMax)
            surrogate_model = KrigingVectorModel()
            surrogate_model.train(
                X_train_raw, y_train_qoi,
                X_val_raw,y_val_per_qoi[qoi_name]
            )

            trained_models[qoi_name] = surrogate_model
        except Exception as e:
            print(f"ERRO ao treinar kriging para {qoi_name}: {e}. Pulando este QoI.")
            import traceback
            traceback.print_exc()
            continue # Pula para o próximo QoI

    print("\n--- Treinamento de todos os modelos concluído ---")


    # --- ETAPA 6 ATUALIZADA: Avaliação e Comparação Visual ---
    print("\n--- Etapa 6: Avaliação e Comparação Visual (Loop por QoI) ---")
        
    if len(X_test_raw) > 0 and trained_models:
        # Pega o primeiro run do nosso conjunto de teste para a comparação
        sample_to_compare_idx = 0
        run_number_to_compare = test_indices_map[sample_to_compare_idx]
        
        # Prepara o X_sample (é o mesmo para todos os modelos)
        x_sample = X_test_raw[[sample_to_compare_idx]]

        # Carrega o arquivo do modelo antigo UMA VEZ
        project_name = RUN_DATA_FILE.stem.replace('.runs-specs', '')
        old_surrogate_path = RESULTS_BASE_DIR / f"{project_name}_R{run_number_to_compare:05}" / "surrogate_summary.json"
        
        print(f"Carregando dados do modelo antigo do Run #{run_number_to_compare} de '{old_surrogate_path}'...")
        y_pred_old_dict = _parse_single_sa_summary(old_surrogate_path)
        
        # Faz a predição e plota para CADA modelo treinado
        for qoi_name in trained_models.keys(): # Itera apenas nos modelos que treinaram com sucesso
            print(f"Gerando gráfico de comparação para o QoI: {qoi_name}...")
            
            # Pega o modelo e os dados Y corretos
            model = trained_models[qoi_name]
            y_true_scaled = y_test_per_qoi[qoi_name][sample_to_compare_idx]
            y_scaler = y_scalers[qoi_name]
            print(f"  [DIAG] Y_TRUE (Scaled) Min: {np.min(y_true_scaled):.4f}, Max: {np.max(y_true_scaled):.4f}")

            y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
            print(f"  [DIAG] Y_TRUE (Unscaled) Min: {np.min(y_true):.4f}, Max: {np.max(y_true):.4f}")
            y_pred_new_scaled = model.predict_values(x_sample)[0] 
            print(f"  [DIAG] Y_PRED (Scaled) Min: {np.min(y_pred_new_scaled):.4f}, Max: {np.max(y_pred_new_scaled):.4f}")
            y_pred_new = y_scaler.inverse_transform(y_pred_new_scaled.reshape(-1, 1)).flatten()
            print(f"  [DIAG] Y_PRED (Unscaled) Min: {np.min(y_pred_new):.4f}, Max: {np.max(y_pred_new):.4f}")
            # Previsão do modelo antigo (RBF)
            y_pred_old = y_pred_old_dict.get(qoi_name)

            # Gera o gráfico
            if y_pred_old:
                plot_comparison_timeseries(
                    y_true=y_true, 
                    y_pred_new=y_pred_new, 
                    y_pred_old=y_pred_old, 
                    run_number=run_number_to_compare,
                    qoi_name=qoi_name,
                    new_model_label="Previsão do Novo Modelo (Kriging Tuned)"
                )
            else:
                print(f"AVISO: Não foi possível encontrar o QoI '{qoi_name}' no 'surrogate_summary.json' (modelo antigo).")
    
    elif not trained_models:
        print("Nenhum modelo foi treinado com sucesso. Pulando etapa de avaliação.")

    print("\n--- PIPELINE CONCLUÍDA COM SUCESSO ---")

except Exception as e:
    print(f"\n--- ERRO NA PIPELINE: {e} ---")
    import traceback
    traceback.print_exc()