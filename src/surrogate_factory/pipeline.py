import numpy as np
from pathlib import Path

from parsers.esss_json_parser import load_data
from preprocessing.processor import process_data
from models.neural_network import build_model
from evaluation.plotting import plot_feature_distribution, plot_target_timeseries

print("--- INICIANDO PIPELINE DE TREINAMENTO DO MODELO SURROGADO ---")

METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field") 

try:
    print("\n--- Etapa 1: Carregando e Analisando os Dados ---")
    full_df, metadata = load_data(
        run_specs_path=RUN_DATA_FILE,
        results_base_dir=RESULTS_BASE_DIR,
        metadata_path=METADATA_FILE,
        result_filename="sa_summary.json"
    )

    if full_df.empty:
        raise RuntimeError("Nenhum dado foi carregado. Verifique os caminhos e os arquivos.")

    print("\n--- Etapa 2: Pré-processando os Dados ---")
    processed_data = process_data(df=full_df, metadata=metadata)
    
    X_train = processed_data["X_train"]
    X_test = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_test = processed_data["y_test"]
    
    print(f"Dados prontos. Formato de X_train: {X_train.shape}, Formato de y_train: {y_train.shape}")

    print("\n--- Etapa 3: Gerando Gráficos de Análise ---")
    plot_feature_distribution(X_train)
    plot_target_timeseries(y_train)

    print("\n--- Etapa 4: Construindo a Arquitetura do Modelo ---")
    n_features = X_train.shape[1]
    n_outputs = y_train.shape[1]
    
    surrogate_model = build_model(input_shape=n_features, output_shape=n_outputs)
    surrogate_model.summary()

    print("\n--- Etapa 5: Iniciando o Treinamento ▶️ ---")
    history = surrogate_model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=2,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("\n--- Etapa 6: Avaliação Final do Modelo 🏁 ---")
    loss = surrogate_model.evaluate(X_test, y_test, verbose=0)
    print(f"Perda (Loss) final no conjunto de teste: {loss:.4f}")

    print("\n--- PIPELINE CONCLUÍDA COM SUCESSO ---")

except Exception as e:
    print(f"\n--- ERRO NA PIPELINE: {e} ---")