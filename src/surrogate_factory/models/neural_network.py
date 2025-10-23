"""
Módulo para construção de arquiteturas de modelos de redes neurais.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout

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
        Input(shape=(input_shape,), name='input_layer'),
        
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        
        Dense(output_shape, name='output_layer')
    ], name='surrogate_model')
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    return model