"""
Módulo para construção de arquiteturas de modelos de redes neurais.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def _build_model(input_shape: int, output_shape: int) -> tf.keras.Model:
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

class NeuralNetworkModel:
    """
    Esta classe agrupa um único modelo Keras para prever um vetor de saída,
    mantendo uma interface consistente de .train() e .predict_values().
    """

    def __init__(self):
        self.model: tf.keras.Model = None
        self.epochs = 200
        self.batch_size = 32
        self.validation_patience = 20

    def train(self, X_train, y_train_vector, X_val, y_val_vector):
        """
        Treina um único modelo Keras para prever o vetor de saída completo.
        
        Nota: Esta classe espera que X_train e X_val já estejam ESCALADOS.
        """

        input_shape = X_train.shape[1]
        output_shape = y_train_vector.shape[1]

        self.model = _build_model(input_shape, output_shape)

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.validation_patience,
            restore_best_weights=True
        )
        print(f"--- Treinando Modelo de Rede Neural (para {output_shape} timesteps) ---")

        self.model.fit(
            X_train,
            y_train_vector,
            validation_data=(X_val, y_val_vector),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop_callback],
            verbose=0
        )
        
        print(f"--- Treinamento da Rede Neural concluído ---")

    def predict_values(self, X_sample):
        """
        Prevê o vetor de time-series completo para uma nova amostra de entrada X.
        """

        if self.model is None:
            raise RuntimeError("modelo ainda não foi treinado, Chame .train() primeiro")
        return self.model.predict(X_sample)