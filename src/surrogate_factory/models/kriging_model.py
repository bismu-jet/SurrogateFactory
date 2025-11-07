import numpy as np
from smt.surrogate_models import KRG
from collections import namedtuple
import sys

def build_and_train_tuned_kriging(X_train, y_train, X_test, y_test):
    """
    Cria e treina um modelo Kriging (KRG), procurando os melhores
    hiperparâmetros (poly, corr) usando os dados de validação.
    """
    if hasattr(y_train, 'values'): y_train = y_train.values
    if hasattr(y_test, 'values'): y_test = y_test.values
    if hasattr(X_train, 'values'): X_train = X_train.values
    if hasattr(X_test, 'values'): X_test = X_test.values

    KrigingParameters = namedtuple("KrigingParameters", ["poly", "corr"])
    trained_models = {}
    errors = {}
    best_params = None
    smallest_error = float('inf')
    polynomials = ['constant', 'linear', 'quadratic']
    
    correlations = ['squar_exp', 'abs_exp', 'matern52', 'matern32']

    print(f"--- Iniciando Tuning do Kriging (Testando {len(polynomials) * len(correlations)} combinações) ---")

    for poly in polynomials:
        for corr in correlations:
            params = KrigingParameters(poly=poly, corr=corr)
            try:
                temp_model = KRG(poly=poly, corr=corr, print_global=False, n_start=10, nugget=1e-8)
                temp_model.set_training_values(X_train, y_train)
                temp_model.train()

                y_pred_test = temp_model.predict_values(X_test)
                
                error = np.sqrt(np.mean((y_pred_test - y_test)**2))

                trained_models[params] = temp_model
                errors[params] = error

                if error < smallest_error:
                    smallest_error = error
                    best_params = params
                    
            except np.linalg.LinAlgError:
                 print(f"    [!] Combinação poly={poly}, corr={corr} falhou (LinAlgError). Pulando.")
                 continue
            except ValueError:
                 print(f"    [!] Combinação poly={poly}, corr={corr} falhou (ValueError). Pulando.")
                 continue

    if best_params is None:
        raise RuntimeError("Tuning do Kriging falhou. Nenhuma combinação válida encontrada.")

    print(f"--- Tuning do Kriging concluído ---")
    print(f"Melhores parâmetros encontrados: Poly={best_params.poly}, Corr={best_params.corr}")
    print(f"Menor erro RMSE na validação: {smallest_error:.4f}")

    return trained_models[best_params]