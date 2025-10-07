# src/fabrica_de_surrogados/evaluation/plotting.py

"""
Módulo para gerar visualizações úteis para a avaliação
do pipeline e dos modelos.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_distribution(X_data: pd.DataFrame, save_path: str = "feature_distribution.png"):
    """
    Gera um gráfico de dispersão para visualizar a distribuição das features.
    Funciona melhor para 2 features.
    """
    if X_data.shape[1] != 2:
        print("Aviso: O plot de distribuição de features funciona melhor com 2 features.")
        return
        
    plt.figure(figsize=(8, 6))
    plt.scatter(X_data.iloc[:, 0], X_data.iloc[:, 1], alpha=0.7)
    plt.title('Distribuição das Amostras de Treino no Espaço de Features')
    plt.xlabel(f'Feature 1 (escalonada)')
    plt.ylabel(f'Feature 2 (escalonada)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de distribuição de features salvo em: {save_path}")

def plot_target_timeseries(y_data: np.ndarray, num_to_plot: int = 5, save_path: str = "target_timeseries.png"):
    """
    Plota as séries temporais de saída (targets) para algumas amostras.
    """
    plt.figure(figsize=(12, 6))
    
    # Garante que não tentaremos plotar mais amostras do que temos
    num_to_plot = min(num_to_plot, len(y_data))
    
    for i in range(num_to_plot):
        plt.plot(y_data[i], label=f'Amostra de Treino {i+1}', alpha=0.8)
        
    plt.title(f'Visualização das {num_to_plot} Primeiras Séries Temporais de Saída (y_train)')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Valor da Saída')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico das séries temporais de saída salvo em: {save_path}")

def plot_comparison_timeseries(y_true, y_pred_new, y_pred_old, run_number, save_path="comparison_run_"):
    """
    Plota a comparação entre o resultado real, a previsão do modelo novo e a do antigo.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(y_true, label='Resultado Real (Ground Truth)', color='black', linewidth=2.5, alpha=0.8)
    plt.plot(y_pred_new, label='Previsão do Novo Modelo (Rede Neural)', color='blue', linestyle='--', linewidth=2)
    plt.plot(y_pred_old, label='Previsão do Modelo Antigo (RBF)', color='red', linestyle=':', linewidth=2)
    
    plt.title(f'Comparação de Modelos para o Run #{run_number}')
    plt.xlabel('Passos de Tempo Concatenados')
    plt.ylabel('Valor da Saída')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    final_save_path = f"{save_path}{run_number}.png"
    plt.savefig(final_save_path)
    plt.close()
    print(f"Gráfico de comparação para o Run #{run_number} salvo em: {final_save_path}")