import numpy as np
import pytest
from smt.surrogate_models import RBF
from collections import namedtuple

import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from surrogate_factory.models.rbf_model import RBFVectorModel, _build_and_tune_single_rbf

def test_rbf_vector_model_workflow():
    NUM_TRAIN_SAMPLES = 10
    NUM_VAL_SAMPLES = 4
    NUM_FEATURES = 2
    NUM_TIMESTEPS = 3
    np.random.seed(42)

    X_train = np.random.rand(NUM_TRAIN_SAMPLES, NUM_FEATURES)
    y_train_vector = np.random.rand(NUM_TRAIN_SAMPLES,NUM_TIMESTEPS)

    X_val = np.random.rand(NUM_VAL_SAMPLES,NUM_FEATURES)
    y_val_vector = np.random.rand(NUM_VAL_SAMPLES,NUM_TIMESTEPS)

    X_test_sample = np.random.rand(1, NUM_FEATURES)

    model = RBFVectorModel()

    model.train(X_train, y_train_vector, X_val, y_val_vector)

    y_pred_vector = model.predict_values(X_test_sample)

    assert model.num_timesteps == NUM_TIMESTEPS, \
        f"Esperado {NUM_TIMESTEPS} timesteps, mas o modelo registrou {model.num_timesteps}"
    
    assert len(model.models_per_timestep) == NUM_TIMESTEPS, \
        f"Esperado {NUM_TIMESTEPS} modelos na lista, mas foram encontrados {len(model.models_per_timestep)}"
    
    expected_shape = (1, NUM_TIMESTEPS)
    assert y_pred_vector.shape == expected_shape, \
        f"Shape da predição está incorreto. Esperado {expected_shape}, mas foi {y_pred_vector.shape}"
    
def test_build_and_tune_single_rbf():
    """
    Testa a função "helper" _build_and_tune_single_rbf
    para garantir que ela retorna um modelo RBF válido.
    """

    X_train = np.random.rand(10, 2)
    y_train_scalar = np.random.rand(10)
    X_val = np.random.rand(4, 2)
    y_val_scalar = np.random.rand(4)

    model = _build_and_tune_single_rbf(X_train,y_train_scalar,X_val,y_val_scalar, num_tries=10)

    assert isinstance(model, RBF), "A função não retornou um objeto RBF"

def test_build_and_tune_single_rbf_flatline():
    """
    Testa a função "helper" com dados de validação "flat line" (variância zero)
    para garantir que ela não falha.
    """

    X_train = np.random.rand(10, 2)
    y_train_scalar = np.random.rand(10)
    X_val = np.random.rand(4, 2)
    y_val_scalar_flat = np.full((4,), 5.0)

    try:
        model = _build_and_tune_single_rbf(X_train, y_train_scalar, X_val, y_val_scalar_flat, num_tries=10)
        assert isinstance(model, RBF), "a função não retornou um objeto RBF"

        pred = model.predict_values(X_val)
        assert pred.shape == (4,1), "A predição do modelo falhou"

    except Exception as e:
        pytest.fail(f"A função falhou com dados 'flatline': {e}")