import numpy as np
from smt.surrogate_models import RBF
from collections import namedtuple

def _build_and_tune_single_rbf(X_train, y_train_scalar, X_val, y_val_scalar, num_tries=100):
    """
    Cria e treina um modelo RBF simples (sem tuning) para regressão.
    """
    if hasattr(y_train_scalar, 'values'): y_train_scalar = y_train_scalar.values
    if hasattr(y_val_scalar, 'values'): y_val_scalar = y_val_scalar.values
    if hasattr(X_train, 'values'): X_train = X_train.values
    if hasattr(X_val, 'values'): X_val = X_val.values

    RbfParameters = namedtuple("RbfParameters", ["d0", "degree"])
    trained_models = {}
    errors = {}
    best_params = None
    smallest_error = float('inf')

    distances = np.geomspace(1, 100000.0, num=num_tries)
    degrees= [-1,0,1]

    for d in distances:
        for degree in degrees:
            params = RbfParameters(d0=d, degree=degree)
            try:
                temp_model = RBF(print_global=False, d0=d, poly_degree=degree)
                temp_model.set_training_values(X_train, y_train_scalar)
                temp_model.train()

                y_pred_val = temp_model.predict_values(X_val)
                error = np.sqrt(np.mean((y_pred_val - y_val_scalar)**2))

                trained_models[params] = temp_model
                errors[params] = error
            except np.linalg.LinAlgError:
                continue

    if best_params is None:
        best_params = RbfParameters(d0=1, degree=0)
        temp_model = RBF(print_global=False, d0=best_params.d0, poly_degree=best_params.degree)
        temp_model.set_training_values(X_train, y_train_scalar)
        temp_model.train()
        trained_models[best_params] = temp_model

    return trained_models[best_params]

class RBFVectorModel:
    """
    Esta classe agrupa múltiplos modelos RBF (um para cada timestep)
    para simular um único modelo que prevê um vetor.
    """
    def __init__(self):
        self.model = None
        self.best_params = None
        self._number_of_tryout_distances = 2000

    def train(self, X_train, y_train_vector, X_val, y_val_vector):
        """
        Treina um modelo RBF separado para cada timestep.
        
        Args:
            X_train: Inputs de treino (normalizados).
            y_train_vector: Vetor de saída de treino (n_samples, n_timesteps).
            X_val: Inputs de validação (normalizados).
            y_val_vector: Vetor de saída de validação.
        """
        print(f"--- Iniciando Tuning do RBF Nativo (Output Shape: {y_train_vector.shape}) ---")
        
        RbfParameters = namedtuple("RbfParameters", ["d0", "degree"])

        trained_models = {}
        errors = {}

        distances = np.geomspace(1, 1000, num=self._number_of_tryout_distances)
        degrees = [-1, 0, 1]

        best_error = float('inf')
        best_id = None

        for d in distances:
            for degree in degrees:
                params_id = RbfParameters(d0=d, degree=degree)
                try:
                    temp_model = RBF(d0=d, poly_degree=degree, print_global=False)

                    temp_model.set_training_values(X_train, y_train_vector)
                    temp_model.train()

                    y_pred_val = temp_model.predict_values(X_val)

                    current_total_error = 0.0

                    for i in range(len(y_val_vector)):
                        prediction = y_pred_val[i]
                        validation = y_val_vector[i]

                        abs_norm = np.linalg.norm(prediction - validation)
                        norm = np.linalg.norm(validation)

                        if norm > 0.0:
                            current_total_error += float(abs_norm/norm)
                        else:
                            current_total_error += float(abs_norm)
                        
                        relative_error_metric = current_total_error / len(y_val_vector)
                        trained_models[params_id]= temp_model

                        if relative_error_metric < best_error:
                            best_error = relative_error_metric
                            best_id = params_id

                except Exception as e:
                    continue

        if best_id is None:
            print("AVISO: Tuning falhou. Usando parâmetros padrão.")
            best_id = RbfParameters(d0=1.0, degree=0)
            temp_model = RBF(d0=1.0, poly_degree=0, print_global=False)
            temp_model.set_training_values(X_train, y_train_vector)
            temp_model.train()
            self.model = temp_model
        else:
            self.model = trained_models[best_id]
        
        self.best_params = best_id
        
        print(f"--- Melhor RBF encontrado: d0={best_id.d0:.4f}, degree={best_id.degree} (RMSE Val: {best_error:.6f}) ---")

    def predict_values(self, X_sample):
        """
        Prevê o vetor de time-series completo para uma nova amostra de entrada X.
        
        X_sample tem shape (n_samples, n_features)
        """

        if self.model is None:
            raise RuntimeError("Modelo RBF ainda não foi treinado.")
        
        return self.model.predict_values(X_sample)