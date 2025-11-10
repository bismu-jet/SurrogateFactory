import numpy as np
from smt.surrogate_models import RBF
from collections import namedtuple

def build_and_train_rbf(X_train, y_train):
    """
    Cria e treina um modelo RBF simples (sem tuning) para regressão.
    """
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    rbf_model = RBF(print_global=False, d0=1, poly_degree=0) # Usando defaults simples
    rbf_model.set_training_values(X_train, y_train)
    print("--- Treinando o Modelo RBF Simples ---")
    rbf_model.train()
    print("Treinamento do RBF Simples concluído.")
    return rbf_model

def build_and_train_tuned_rbf(X_train, y_train, X_val, y_val, num_tries=1000):
    """
    Cria e treina um modelo RBF, procurando os melhores hiperparâmetros (d0, degree)
    usando os dados de validação (X_val, y_val).
    """
    if hasattr(y_train, 'values'): y_train = y_train.values
    if hasattr(y_val, 'values'): y_val = y_val.values
    if hasattr(X_train, 'values'): X_train = X_train.values
    if hasattr(X_val, 'values'): X_val = X_val.values

    RbfParameters = namedtuple("RbfParameters", ["d0", "degree"])
    trained_models = {}
    errors = {}
    best_params = None
    smallest_error = float('inf')

    # Define as distâncias e graus a serem testados
    distances = np.geomspace(1, 100000.0, num=num_tries)
    degrees = [0]

    print(f"--- Iniciando Tuning do RBF (Testando {len(distances) * len(degrees)} combinações) ---")

    for i, d in enumerate(distances):
        for degree in degrees:
            params = RbfParameters(d0=d, degree=degree)
            try:
                # Cria e treina um modelo RBF com os parâmetros atuais
                temp_model = RBF(print_global=False, d0=d, poly_degree=degree)
                temp_model.set_training_values(X_train, y_train)
                temp_model.train()

                # Avalia o erro nos dados de TESTE
                y_pred_val = temp_model.predict_values(X_val)
                
                # Usa RMSE como métrica de erro (menor é melhor)
                error = np.sqrt(np.mean((y_pred_val - y_val)**2))

                trained_models[params] = temp_model
                errors[params] = error

                # Atualiza o melhor modelo encontrado até agora
                if error < smallest_error:
                    smallest_error = error
                    best_params = params
                    
            except np.linalg.LinAlgError:
                 # Ignora combinações que causam erros de álgebra linear (matriz singular)
                 print(f"    [!] Combinação d0={d:.2f}, degree={degree} falhou. Pulando.")
                 continue # Pula para a próxima combinação

        # Print de progresso opcional
        if (i + 1) % (num_tries // 10) == 0:
             print(f"    Progresso do Tuning: {(i+1)/num_tries*100:.0f}% concluído...")

    if best_params is None:
        raise RuntimeError("Tuning do RBF falhou. Nenhuma combinação válida encontrada.")

    print(f"--- Tuning do RBF concluído ---")
    print(f"Melhores parâmetros encontrados: d0={best_params.d0:.4f}, degree={best_params.degree}")
    print(f"Menor erro RMSE na validação: {smallest_error:.4f}")

    # Retorna o melhor modelo encontrado
    return trained_models[best_params]