"""
Módulo para construção de arquiteturas de modelos de redes neurais.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_shape: int, output_shape: int) -> tf.keras.Model:
    """
    Constrói e compila um modelo de rede neural (MLP) para regressão.

    A arquitetura usa camadas densas com ativação 'relu', uma escolha robusta
    para aprender relações não-lineares. A camada de saída não tem ativação
    (linear), o que é padrão para problemas de regressão.

    Args:
        input_shape (int): O número de features de entrada (X_train.shape[1]).
        output_shape (int): O número de valores de saída (y_train.shape[1]).

    Returns:
        Um modelo Keras sequencial, compilado e pronto para o treinamento.
    """
    model = Sequential([
        # Camada de entrada com o formato dos nossos dados X
        Input(shape=(input_shape,), name='input_layer'),
        
        # Camadas ocultas (hidden layers) para aprender os padrões.
        # A escolha de 128 e 64 neurônios é um bom ponto de partida.
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        
        # Camada de saída com um neurônio para cada valor do vetor de saída
        Dense(output_shape, name='output_layer')
    ], name='surrogate_model')
    
    # Compila o modelo. 'adam' é um otimizador eficiente e 'mean_squared_error'
    # é a função de perda padrão para problemas de regressão.
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model