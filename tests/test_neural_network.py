import pytest
import numpy as np
import tensorflow as tf

import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from surrogate_factory.models.neural_network import NeuralNetworkModel

def test_neural_network_model_workflow():
    """
    Testa o fluxo completo (train, predict) da NeuralNetworkModel
    com dados pequenos.
    """

    NUM_TRAIN_SAMPLES = 20 
    NUM_VAL_SAMPLES = 5
    NUM_FEATURES = 2
    NUM_TIMESTEPS = 3

    np.random.seed(42)

    X_train = np.random.rand(NUM_TRAIN_SAMPLES,NUM_FEATURES)
    y_train_vector = np.random.rand(NUM_TRAIN_SAMPLES,NUM_TIMESTEPS)

    X_val = np.random.rand(NUM_VAL_SAMPLES,NUM_FEATURES)
    y_val_vector = np.random.rand(NUM_VAL_SAMPLES,NUM_TIMESTEPS)

    X_test_sample = np.random.rand(2, NUM_FEATURES)

    model = NeuralNetworkModel()

    model.epochs=3
    model.patience=1

    model.train(X_train, y_train_vector, X_val, y_val_vector)

    y_pred_vector = model.predict_values(X_test_sample)

    assert model.model is not None, "o modelo keras não foi criado"
    assert isinstance(model.model, tf.keras.Model), "o modelo não é um objeto keras"

    expected_shape = (2, NUM_TIMESTEPS)
    assert y_pred_vector.shape == expected_shape, \
        f"Shape da predição está incorreto. Esperado {expected_shape}, mas foi {y_pred_vector.shape}"