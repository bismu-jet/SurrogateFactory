"""Integration tests for the SurrogateFactory end-to-end workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surrogate_factory.factory import ModelType, SurrogateFactory


@pytest.fixture()
def fake_data_files(tmp_path: Path):
    """Create minimal feature/target/metadata files on disk."""
    df = pd.DataFrame(
        {
            "run_id": [str(i) for i in range(10)],
            "Feature_A": np.linspace(0, 10, 10),
            "Feature_B": np.linspace(100, 110, 10),
            "Categoria": ["A", "B"] * 5,
        }
    )
    feat_path = tmp_path / "features.csv"
    df.to_csv(feat_path, index=False)

    targets = {str(i): [float(i), float(i) * 2, float(i) * 3] for i in range(10)}
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets))

    metadata = {"Categoria": ["A", "B"]}
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))

    return feat_path, target_path, meta_path


def test_factory_initialisation():
    factory = SurrogateFactory(ModelType.KRIGING)
    assert factory.model_type == ModelType.KRIGING

    with pytest.raises(ValueError):
        SurrogateFactory("InvalidString")


def test_full_workflow_kriging(fake_data_files, tmp_path: Path):
    """Load → Train → Predict → Save → Load round-trip with Kriging."""
    feat_path, target_path, meta_path = fake_data_files

    factory = SurrogateFactory(ModelType.KRIGING)
    factory.load_data(feat_path, target_path, meta_path)

    assert factory._all_runs_ids is not None
    assert len(factory._all_runs_ids) == 10
    assert factory.metadata == {"Categoria": ["A", "B"]}

    factory.train(test_size=0.2, val_size=0.2)
    assert "outputs" in factory.trained_models_per_qoi
    assert len(factory.feature_names) == 3

    x_new = pd.DataFrame(
        {"Feature_A": [5.0], "Feature_B": [105.0], "Categoria": ["A"]}
    )
    preds = factory.predict(x_new)
    assert "outputs" in preds
    assert preds["outputs"].shape == (1, 3)

    factory.evaluate_and_plot(plot_prefix=str(tmp_path / "test_plot"))

    save_path = tmp_path / "model_test.pkl"
    factory.save_model(save_path)
    assert save_path.exists()

    loaded = SurrogateFactory.load_model(save_path)
    assert loaded.model_type == ModelType.KRIGING

    preds_loaded = loaded.predict(x_new)
    np.testing.assert_allclose(preds["outputs"], preds_loaded["outputs"])


def test_workflow_neural_network(fake_data_files):
    """Quick training cycle with Neural Network to verify scaling path."""
    feat_path, target_path, meta_path = fake_data_files

    factory = SurrogateFactory(ModelType.NN)
    factory.load_data(feat_path, target_path, meta_path)
    factory.train(test_size=0.2, val_size=0.2)

    x_new = pd.DataFrame(
        {"Feature_A": [5.0], "Feature_B": [105.0], "Categoria": ["A"]}
    )
    preds = factory.predict(x_new)
    assert preds["outputs"].shape == (1, 3)
