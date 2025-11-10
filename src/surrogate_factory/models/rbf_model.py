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
        self.models_per_timestep = []
        self.num_timesteps = 0

    def train(self, X_train, y_train_vector, X_val, y_val_vector):
        """
        Treina um modelo RBF separado para cada timestep.
        
        y_train_vector e y_val_vector têm shape (num_samples, num_timesteps)
        """
        self.models_per_timestep = []
        self.num_timesteps = y_train_vector.shape[1]

        print(f"--- Treinando {self.num_timesteps} modelos RBF (um por timestep) ---")

        for i in range(self.num_timesteps):
            y_train_scalar = y_train_vector[:, i]
            y_val_scalar = y_val_vector[:, i]

            tuned_model = _build_and_tune_single_rbf(
                X_train, y_train_scalar,
                X_val, y_val_scalar
            )

            self.models_per_timestep.append(tuned_model)

        print(f"--- Treinamento dos {self.num_timesteps} modelos concluído ---")

    def predict_values(self, X_sample):
        """
        Prevê o vetor de time-series completo para uma nova amostra de entrada X.
        
        X_sample tem shape (n_samples, n_features)
        """

        all_timestep_prediction_arrays = []

        for model in self.models_per_timestep:
            y_pred_timestep = model.predict_values(X_sample)
            all_timestep_prediction_arrays.append(y_pred_timestep)

        return np.hstack(all_timestep_prediction_arrays)