import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum
from sklearn.model_selection import train_test_split
import json
import warnings
import pickle

from .parsers.general_parser import load_standard_format
from .preprocessing.processor import process_data_multi_model, OutputCompressor

from .models.kriging_model import KrigingVectorModel
from .models.neural_network import NeuralNetworkModel
from .models.rbf_model import RBFVectorModel

from .evaluation.plotting import plot_comparison_timeseries
from .evaluation.metrics import calculate_r2, calculate_rmse

class ModelType(Enum):
    KRIGING = "Kriging"
    RBF = "RBF"
    NN = "NeuralNetwork"

class SurrogateFactory:

    def __init__(self, model_type: ModelType):

        if not isinstance(model_type, ModelType):
            raise ValueError("model_type deve ser um membro de ModelType (ex: ModelType.KRIGING)")

        self.model_type = model_type
        self.trained_models_per_qoi = {}

        self._df_features_raw = None
        self._qois_raw = None
        self._all_runs_ids = None

        self.processed_data = {}
        self.qoi_names = []
        self.feature_names = []
        self.x_scaler = None
        self.y_scalers = {}
        self.metadata = {}

        print(f"SurrogateFactory inicializada com o tipo de modelo: {self.model_type.value}")

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def set_data(self, features_df: pd.DataFrame, qois_dict: dict):
        """
        Define os dados manualmente.
        
        Args:
            features_df (pd.DataFrame): DataFrame com as features. Deve ter 'run_id' ou 'run_number'.
            qois_dict (dict): Dicionário de targets {run_id: {qoi_name: vetor}}.
        """
        if 'run_number' in features_df.columns and 'run_id' not in features_df.columns:
            features_df = features_df.rename(columns={'run_number': 'run_id'})
        if 'run_id' not in features_df.columns:
            raise KeyError("O dataframe fornecido precisa de uma coluna 'run_id'")
        
        features_df['run_id'] = features_df['run_id'].astype(str)
        self._df_features_raw = features_df

        self._qois_raw = {str(k): v for k, v in qois_dict.items()}

        self._all_runs_ids = self._df_features_raw['run_id'].unique()

    def load_data(self, features_path:str | Path, targets_path: str | Path, metadata_path: str | Path = None):
        """
        Carrega os dados de entrada usando o parser genérico.
        
        Args:
            features_path (str | Path): Caminho para o arquivo features.csv.
            targets_path (str | Path): Caminho para o arquivo targets.json.
            metadata_path (str | Path, opcional): Caminho para o metadata.json (para features categóricas).
        """

        print("--- Etapa 1: Carregando Dados ---")
        features_path = Path(features_path)
        targets_path = Path(targets_path)

        self.metadata = {}
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                print(f"Carregando dados de {metadata_path}")
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    print(f"Erro ao tentar ler:{e}")
            else:
                print(f"O path '{metadata_path}' não contem o arquivo de metadados.")

        try:
            self._df_features_raw, self._qois_raw = load_standard_format(
                features_path=features_path,
                targets_path=targets_path
            )
            self._all_runs_ids = self._df_features_raw['run_id'].astype(str).unique()
            print(f"Dados carregados com sucesso. Encontrados {len(self._all_runs_ids)} runs únicos.")
        
        except FileNotFoundError:
            print(f"ERRO: Arquivos não encontrados em '{features_path}' ou '{targets_path}'")
            raise
        except Exception as e:
            print(f"ERRO ao carregar dados: {e}")
            raise

    def train(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Divide os dados, pré-processa e treina o(s) modelo(s).
        
        Args:
            test_size (float): Proporção do dataset a ser usada para teste (ex: 0.2 para 20%).
            val_size (float): Proporção do dataset *total* a ser usada para validação (ex: 0.1 para 10%).
            random_state (int): Semente aleatória para reprodutibilidade.
        """
        print("\n--- Etapa 2: Dividindo, Processando e Treinando ---")

        if self._all_runs_ids is None:
            raise RuntimeError("Dados não carregados, chame o load_data()")
        
        print(f"Dividindo {len(self._all_runs_ids)} runs (teste={test_size}, val={val_size})...")
        train_val_ids, test_ids = train_test_split(
            self._all_runs_ids,
            test_size=test_size,
            random_state=random_state
        )

        relative_val_size = val_size / (1.0-test_size)

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val_size,
            random_state=random_state
        )

        print(f"{len(train_ids)} treino, {len(val_ids)} validação, {len(test_ids)} teste.")

        df_train_features = self._df_features_raw[self._df_features_raw['run_id'].isin(train_ids)]
        df_val_features = self._df_features_raw[self._df_features_raw['run_id'].isin(val_ids)]
        df_test_features = self._df_features_raw[self._df_features_raw['run_id'].isin(test_ids)]

        qois_train = {run_id: data for run_id, data in self._qois_raw.items() if run_id in train_ids}
        qois_val = {run_id: data for run_id, data in self._qois_raw.items() if run_id in val_ids}
        qois_test = {run_id: data for run_id, data in self._qois_raw.items() if run_id in test_ids}

        print("--- Pré processamento dos dados ---")

        self.processed_data = process_data_multi_model(
            df_train_features=df_train_features,
            qois_train=qois_train,
            df_val_features=df_val_features,
            qois_val=qois_val,
            df_test_features=df_test_features,
            qois_test=qois_test,
            metadata=self.metadata
        )

        self.x_scaler = self.processed_data["x_scaler"]
        self.y_scalers = self.processed_data["y_scalers"]
        self.qoi_names = self.processed_data["qoi_names"]
        self.feature_names = self.processed_data["feature_names"]

        self.test_run_id_map = pd.Series(
            df_test_features['run_id'].values,
            index=range(len(df_test_features))
        )

        print(f"Processamento concluído. {len(self.qoi_names)} QoI(s) encontrados: {self.qoi_names}")

        

        self.trained_models_per_qoi = {}
        self.compressors_per_qoi = {}

        for qoi_name in self.qoi_names:
            y_train_qoi = self.processed_data["y_train_per_qoi"][qoi_name]
            y_val_qoi = self.processed_data["y_val_per_qoi"][qoi_name]

            #PCA compression for qoi
            compressor = OutputCompressor(n_components=0.99) # you can set the variance that you want to be captured by the PCA here
            
            Z_train_qoi = compressor.fit_transform(y_train_qoi)
            Z_val_qoi = compressor.transform(y_val_qoi)

            self.compressors_per_qoi[qoi_name] = compressor

            print(f"\n--- Treinando Modelo {self.model_type.value} para o QoI: {qoi_name} ---")

            try:
                if self.model_type == ModelType.KRIGING:
                    model = KrigingVectorModel()
                    model.train(
                        self.processed_data["X_train_raw"], Z_train_qoi,
                        self.processed_data["X_val_raw"], Z_val_qoi
                    )
                
                elif self.model_type == ModelType.RBF:
                    model = RBFVectorModel()
                    model.train(
                        self.processed_data["X_train_raw"], Z_train_qoi,
                        self.processed_data["X_val_raw"], Z_val_qoi
                    )

                elif self.model_type == ModelType.NN:
                    model = NeuralNetworkModel()
                    model.train(
                        self.processed_data["X_train_scaled"], Z_train_qoi,
                        self.processed_data["X_val_scaled"], Z_val_qoi
                    )

                self.trained_models_per_qoi[qoi_name] = model
            except Exception as e:
                print(f"Erro ao treinar {self.model_type.value} para {qoi_name}: {e}. Pulando este QoI")
                import traceback
                traceback.print_exc()
                continue
        print("\n --- Treinamento concluido ---")

    def evaluate_and_plot(self, plot_prefix="surrogate_comparison"):
        if self.model_type==ModelType.NN:
            X_test_data = self.processed_data["X_test_scaled"]
        else:
            X_test_data = self.processed_data["X_test_raw"]

        global_metrics = {qoi: {'rmse': [], 'r2': []} for qoi in self.qoi_names}

        if len(X_test_data) == 0:
            print("WARNING: Test data is empty. skipping evaluation")
            return
        for sample_idx in range(len(X_test_data)):
            run_id = self.test_run_id_map.get(sample_idx, str(sample_idx))

            x_sample = X_test_data[[sample_idx]]
            for qoi_name in self.qoi_names:
                if qoi_name not in self.trained_models_per_qoi:
                    continue
                model = self.trained_models_per_qoi[qoi_name]
                compressor = self.compressors_per_qoi[qoi_name]

                y_true = self.processed_data["y_test_per_qoi"][qoi_name][sample_idx]
                z_pred = model.predict_values(x_sample)
                
                y_pred_reconstructed = compressor.inverse_transform(z_pred)
                y_true_flat = y_true.flatten()
                y_pred_flat = y_pred_reconstructed.flatten()

                mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)

                y_true_clean = y_true_flat[mask]
                y_pred_clean = y_pred_flat[mask]

                if len(y_true_clean) == 0:
                    print(f"[ALERTA] Run {run_id} | {qoi_name}: Todos os dados são NaN/Inf.")
                    continue

                try:
                    rmse = calculate_rmse(y_true_clean,y_pred_clean)
                    if np.var(y_true_clean) < 1e-9:
                        r2 = 1.0 if rmse < 1e-5 else 0.0
                    else:
                        r2 = calculate_r2(y_true_clean, y_pred_clean)
                    global_metrics[qoi_name]['rmse'].append(rmse)
                    global_metrics[qoi_name]['r2'].append(r2)

                except Exception as e:
                    print(f"Erro ao calcular métrica para {run_id}: {e}")

                plot_comparison_timeseries(
                    y_true=y_true_flat,
                    y_pred_new=y_pred_flat,
                    y_pred_old=None,
                    run_number=sample_idx,
                    qoi_name=f"{qoi_name} (Run ID: {run_id})",
                    save_path_prefix=plot_prefix,
                    new_model_label=f"Previsão {self.model_type.value}"
                )
        print("\n" + "="*60)
        print(f"{'QoI NAME':<30} | {'RMSE Médio (std)':<20} | {'R² Médio (std)':<20}")
        print("-" * 60)
        for qoi_name, metrics in global_metrics.items():
            rmse_avg = np.mean(metrics['rmse'])
            rmse_std = np.std(metrics['rmse'])

            r2_avg = np.mean(metrics['r2'])
            r2_std = np.std(metrics['r2'])
            print(f"{qoi_name:<30} | {rmse_avg:.4f} (+-{rmse_std:.4f})   | {r2_avg:.4f} (+-{r2_std:.4f})")
        
        print("="*60 + "\n")

    def _preprocess_input_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Converte um DataFrame de features brutas em um array numpy,
        usando os metadados e a ordem das features aprendidos durante o treino.
        (Lógica adaptada de _build_feature_array em processor.py)
        """
        try:
            features_df_ordered = features_df[self.feature_names]
        except KeyError:
            raise ValueError(f"O DataFrame de entrada não possui todas as colunas necessárias. Esperado: {self.feature_names}")
    
        categorical_col = list(self.metadata.keys())
        processed_rows = []

        for row in features_df_ordered.itertuples(index=False):
            new_row_values = []
            for i, value in enumerate(row):
                col_name = self.feature_names[i]

                if col_name in categorical_col:
                    try:
                        valid_values = self.metadata[col_name]
                        idx = valid_values.index(value)
                        new_row_values.append(float(idx))
                    except ValueError:
                        print(f"AVISO: Valor '{value}' não encontrado no metadata de '{col_name}'. Usando -1.0")
                        new_row_values.append(-1.0)
                else:
                    new_row_values.append(float(value))

            processed_rows.append(new_row_values)

        return np.array(processed_rows)
    
    def predict(self, X_new: pd.DataFrame):
        """
        Faz uma nova predição baseada nos dados de entrada.

        Args:
            X_new (pd.Dataframe): Um df com novas amostras de entrada.
                                  As colunas devem corresponder aos dados de treino.
        Returns:
            Dict[str, np.ndarray]: Um dict onde cada chave é um qoi_name e cada valor
                                   é um np.ndarray (n_samples, n_timesteps) com as
                                   predições des-escaladas (valores reais).
        """
        print("\n --- fazendo a predição ---")

        if not self.trained_models_per_qoi:
            raise RuntimeError("Nenhum modelo treinado. Chame .train() ou .load_model()")
        if self.x_scaler is None or not self.y_scalers:
            raise RuntimeError("Scalers não foram ecnontrados.")
        
        X_raw_numpy = self._preprocess_input_features(X_new)

        if self.model_type == ModelType.NN:
            X_input = self.x_scaler.transform(X_raw_numpy)
        else:
            X_input = X_raw_numpy

        predictions_physical = {}

        for qoi_name, model in self.trained_models_per_qoi.items():

            Z_pred = model.predict_values(X_input)
            if qoi_name not in self.compressors_per_qoi:
                print(f"AVISO: Compressor para {qoi_name} não encontrado.")
                continue
            compressor = self.compressors_per_qoi[qoi_name]
            Y_pred_reconstructed = compressor.inverse_transform(Z_pred)

            predictions_physical[qoi_name] = Y_pred_reconstructed
        print("--- predição completa ---")
        return predictions_physical
    
    def save_model(self, file_path: str | Path):
        """
        Salva a fábrica treinada (com scalers e tals).
        """
        file_path = Path(file_path)
        print(f"\nSalvando modelo em {file_path}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path ,'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, file_path: str | Path):
        """
        Carrega uma fábrica salva pelo save_model
        """
        file_path = Path(file_path)
        print(f"Carregando modelo de {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {file_path}")
        with open(file_path,'rb') as f:
            loaded_object = pickle.load(f)
        if not isinstance(loaded_object, cls):
            raise TypeError(f"O arquivo carregado não é uma instância de {cls.__name__}")
        
        print(f"Modelo carregado com sucesso. Tupo: {loaded_object.model_type.value}")
        return loaded_object