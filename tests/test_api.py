"""Tests for the FastAPI prediction API."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from surrogate_factory.factory import ModelType, SurrogateFactory


@pytest.fixture()
def trained_factory(tmp_path: Path):
    """Train a minimal Kriging factory and persist it."""
    df = pd.DataFrame(
        {
            "run_id": [str(i) for i in range(10)],
            "Feature_A": np.linspace(0, 10, 10),
            "Feature_B": np.linspace(100, 110, 10),
        }
    )
    feat_path = tmp_path / "features.csv"
    df.to_csv(feat_path, index=False)

    targets = {str(i): [float(i), float(i) * 2, float(i) * 3] for i in range(10)}
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(targets))

    factory = SurrogateFactory(ModelType.KRIGING)
    factory.load_data(feat_path, target_path)
    factory.train(test_size=0.2, val_size=0.2)

    model_path = tmp_path / "model.pkl"
    factory.save_model(model_path)
    return model_path


@pytest.fixture()
def client(trained_factory):
    """Create a test client with the model loaded."""
    with patch.dict("os.environ", {"SURROGATE_MODEL_PATH": str(trained_factory)}):
        # Re-import to pick up env var
        from surrogate_factory.api.app import app

        with TestClient(app) as c:
            yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "Kriging"


def test_predict(client):
    resp = client.post(
        "/predict",
        json={"features": [{"Feature_A": 5.0, "Feature_B": 105.0}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert "outputs" in body["predictions"]
    assert len(body["predictions"]["outputs"]) == 1
    assert len(body["predictions"]["outputs"][0]) == 3


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "Kriging"
    assert "feature_names" in body
