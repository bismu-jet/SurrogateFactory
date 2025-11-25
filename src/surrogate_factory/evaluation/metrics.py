"""
Módulo para calcular métricas de performance de modelos de regressão.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o score R-quadrado (R²). Próximo de 1.0 é melhor.
    """
    return r2_score(y_true, y_pred)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE). Menor é melhor.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def generate_performace_report(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, n_samples: int, n_timesteps: int) -> Dict[str, float]:
    """
    Gera um relatório completo de métricas (Globais e por Run).
    
    Args:
        y_true_flat: Array 1D com todos os valores reais concatenados.
        y_pred_flat: Array 1D com todos os valores previstos concatenados.
        n_samples: Número de runs (amostras).
        n_timesteps: Número de passos de tempo por run.
        
    Returns:
        Dict com as métricas calculadas.
    """
    rmse_global = calculate_rmse(y_true_flat,y_pred_flat)
    r2_global = calculate_r2(y_true_flat,y_pred_flat)

    y_true_runs = y_true_flat.reshape(n_samples, n_timesteps)
    y_pred_runs = y_pred_flat.reshape(n_samples, n_timesteps)

    run_rmses = []
    for i in range(n_samples):
        err = calculate_rmse(y_true_runs[i], y_pred_runs[i])
        run_rmses(err)
    return{
        'rmse_global': rmse_global,
        'r2_global': r2_global,
        'rmse_mean_run': np.mean(run_rmses),
        'rmse_std_run': np.std(run_rmses)
    }