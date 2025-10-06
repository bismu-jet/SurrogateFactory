from pathlib import Path
from parsers.esss_json_parser import load_data
from preprocessing.processor import process_data
from evaluation.plotting import plot_feature_distribution, plot_target_timeseries


# Define os caminhos dos arquivos
METADATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.global-sa-input.json")
RUN_DATA_FILE = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field.runs-specs.json")
RESULTS_BASE_DIR = Path(r"C:\PFC\SurrogateFactory\Data Pipeline\esss 3phase lgr field")

# --- ETAPA 1: PARSING ---
full_df, metadata = load_data(
    run_specs_path=RUN_DATA_FILE,
    results_base_dir=RESULTS_BASE_DIR, # Supondo que você criou essa variável
    metadata_path=METADATA_FILE
)

if not full_df.empty:
    print("--- Dados Carregados ---")
    
    # --- ETAPA 2: PRÉ-PROCESSAMENTO ---
    print("\n--- Pré-processando os Dados ---")
    processed_data = process_data(
        df=full_df,
        metadata=metadata
    )

    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]

    print("\nDados prontos para o treinamento:")
    print(f"Formato de X_train: {X_train.shape}")
    print(f"Formato de y_train: {y_train.shape}")

    # --- NOVA ETAPA: VISUALIZAÇÃO ---
    print("\n--- Gerando Gráficos de Análise ---")
    plot_feature_distribution(X_train)
    plot_target_timeseries(y_train)
