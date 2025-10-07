"""
Módulo para calcular métricas de performance de modelos de regressão.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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