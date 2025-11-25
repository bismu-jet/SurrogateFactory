import numpy as np
from smt.surrogate_models import KRG
from collections import namedtuple

def _build_and_tune_single_kriging(X_train, y_train_scalar, X_val, y_val_scalar):
    """
    Cria e treina um ÚNICO modelo Kriging (KRG), procurando os melhores
    hiperparâmetros (poly, corr) usando os dados de validação.
    
    Nota: y_train_scalar e y_val_scalar são vetores 1D (escalares).
    """
    KrigingParameters = namedtuple("KrigingParameters",["poly","corr"])
    trained_models = {}
    errors = {}
    best_params = None
    smallest_error = float('inf')
    polynomials = ['constant', 'linear', 'quadratic']
    correlations = ['matern32']

    for poly in polynomials:
        for corr in correlations:
            params = KrigingParameters(poly=poly, corr=corr)
            try:
                temp_model = KRG(poly=poly, corr=corr, print_global=False, n_start=10, nugget=1e-8)
                temp_model.set_training_values(X_train,y_train_scalar)
                temp_model.train()

                y_pred_val= temp_model.predict_values(X_val)

                error = np.sqrt(np.mean((y_pred_val - y_val_scalar)**2))

                trained_models[params] = temp_model
                errors[params] = error
                if error < smallest_error:
                    smallest_error = error
                    best_params = params
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"    [!] Combinação poly={poly}, corr={corr} falhou. Pulando.")
                continue
        if best_params is None:
            best_params = KrigingParameters(poly='constant', corr='squar_exp')
            temp_model = KRG(poly=best_params.poly, corr=best_params.corr, print_global=False)
            temp_model.set_training_values(X_train, y_train_scalar)
            temp_model.train()
            trained_models[best_params] = temp_model

        return trained_models[best_params]

class KrigingVectorModel:

    def __init__(self):
        self.models_per_timestep = []
        self.best_params_per_timestep = [] # variávei para diagnósico
        self.num_timesteps = 0

    def train(self, X_train, y_train_vector, X_val, y_val_vector):
        self.models_per_timestep = []
        self.num_timesteps = y_train_vector.shape[1]

        print(f"--- Treinando {self.num_timesteps} modelos Kriging (um por timestep) ---")

        for i in range(self.num_timesteps):
            y_train_scalar = y_train_vector[:, i]
            y_val_scalar = y_train_vector[:, i]
            tuned_model = _build_and_tune_single_kriging(
                X_train, y_train_scalar,
                X_val, y_val_scalar
            )
            self.models_per_timestep.append(tuned_model)
        print(f"--- Treinamento dos {self.num_timesteps} modelos concluído ---")

    
    def predict_values(self, X_sample):

        all_timestep_prediction_arrays = []

        for model in self.models_per_timestep:
            y_pred_timestep = model.predict_values(X_sample)
            all_timestep_prediction_arrays.append(y_pred_timestep)

        return np.hstack(all_timestep_prediction_arrays)