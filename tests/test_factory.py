import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_path))

from surrogate_factory.factory import SurrogateFactory, ModelType

@pytest.fixture
def fake_data_files(tmp_path):
    df = pd.DataFrame({
        'run_id': [str(i) for i in range(10)],
        'Feature_A': np.linspace(0, 10, 10),
        'Feature_B': np.linspace(100, 110, 10),
        'Categoria': ['A', 'B'] * 5
    })
    feat_path = tmp_path / "features.csv"
    df.to_csv(feat_path, index=False)

    targets= {}
    for i in range(10):
        targets[str(i)] = [float(i), float(i)*2, float(i)*3]

    target_path = tmp_path / "targets.json"
    with open(target_path, 'w') as f:
        json.dump(targets, f)

    metadata = {'Categoria':['A', 'B']}
    meta_path = tmp_path / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    
    return feat_path, target_path, meta_path

def test_factory_initialization():
    factory = SurrogateFactory(ModelType.KRIGING)
    assert factory.model_type == ModelType.KRIGING

    with pytest.raises(ValueError):
        SurrogateFactory("StringInvalida")

def test_factory_full_workflow_kriging(fake_data_files, tmp_path):
    """
    Testa o ciclo completo (Load -> Train -> Predict -> Save -> Load)
    usando o modelo Kriging.
    """
    feat_path, target_path, meta_path = fake_data_files

    factory = SurrogateFactory(ModelType.KRIGING)
    factory.load_data(feat_path, target_path, meta_path)
    assert factory._all_runs_ids is not None
    assert len(factory._all_runs_ids)==10
    assert factory.metadata == {'Categoria': ['A', 'B']}

    factory.train(test_size=0.2, val_size=0.2)

    assert 'outputs' in factory.trained_models_per_qoi
    assert len(factory.feature_names) == 3

    X_new = pd.DataFrame({
        'Feature_A': [5.0],
        'Feature_B': [105.0],
        'Categoria': ['A']
    })

    preds= factory.predict(X_new)
    assert 'outputs' in preds
    assert preds['outputs'].shape == (1,3)

    try:
        factory.evaluate_and_plot(plot_prefix=str(tmp_path / "test_plot"))
    except Exception as e:
        pytest.fail(f"evaluate_and_plot falhou: {e}")

    save_path = tmp_path / "model_test.pkl"
    factory.save_model(save_path)
    assert save_path.exists()

    loaded_factory = SurrogateFactory.load_model(save_path)
    assert loaded_factory.model_type == ModelType.KRIGING

    preds_loaded = loaded_factory.predict(X_new)
    assert np.allclose(preds['outputs'], preds_loaded['outputs'])

def test_factory_workflow_neural_network(fake_data_files):
    """
    Testa o ciclo rápido com Neural Network para garantir que o scaling
    (Modularidade) está funcionando na Factory.
    """
    feat_path, target_path, meta_path = fake_data_files

    factory = SurrogateFactory(ModelType.NN)
    factory.load_data(feat_path, target_path, meta_path)

    factory.train(test_size=0.2, val_size=0.2)

    X_new = pd.DataFrame({
        'Feature_A': [5.0], 
        'Feature_B': [105.0],
        'Categoria': ['A']
    })

    preds = factory.predict(X_new)
    assert preds['outputs'].shape == (1,3)