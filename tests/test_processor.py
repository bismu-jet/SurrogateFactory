"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from surrogate_factory.preprocessing.processor import process_data_multi_model


def test_process_data_multi_model():
    """Verify shapes, scaling bounds, and QoI consistency filtering."""
    metadata = {"Categorical_Feature": ["A", "B"]}

    df_train = pd.DataFrame(
        {
            "run_number": [1, 2, 3, 4, 5],
            "Categorical_Feature": ["A", "B", "A", "B", "A"],
            "Numeric_Feature": [10, 20, 30, 40, 50],
        }
    )
    df_val = pd.DataFrame(
        {
            "run_number": [6, 7],
            "Categorical_Feature": ["A", "B"],
            "Numeric_Feature": [15, 25],
        }
    )
    df_test = pd.DataFrame(
        {
            "run_number": [8, 9, 10],
            "Categorical_Feature": ["B", "A", "A"],
            "Numeric_Feature": [5, 60, 30],
        }
    )

    qois_train = {
        1: {"QOI_1": [100, 110, 120], "QOI_2": [1, 2, 3]},
        2: {"QOI_1": [110, 110, 120], "QOI_2": [1, 2, 3]},
        3: {"QOI_1": [120, 110, 120], "QOI_2": [1, 2, 3]},
        4: {"QOI_1": [130, 110, 120], "QOI_2": [1, 2, 3]},
        5: {"QOI_1": [200, 210, 220], "QOI_2": [1, 2, 3]},
    }
    qois_val = {
        6: {"QOI_1": [150, 160, 170], "QOI_2": [1, 2, 3]},
        7: {"QOI_1": [160, 160, 170], "QOI_2": [1, 2, 3]},
    }
    # Run 10 is missing QOI_2, so only QOI_1 should survive consistency check.
    qois_test = {
        8: {"QOI_1": [120, 130, 140], "QOI_2": [1, 2, 3]},
        9: {"QOI_1": [300, 310, 320], "QOI_2": [1, 2, 3]},
        10: {"QOI_1": [50, 60, 70]},
    }

    result = process_data_multi_model(
        df_train_features=df_train,
        qois_train=qois_train,
        df_val_features=df_val,
        qois_val=qois_val,
        df_test_features=df_test,
        qois_test=qois_test,
        metadata=metadata,
    )

    # Only QOI_1 should survive because run 10 lacks QOI_2.
    assert result["qoi_names"] == ["QOI_1"]

    # Feature shapes
    assert result["X_train_raw"].shape == (5, 2)
    assert result["X_val_raw"].shape == (2, 2)
    assert result["X_test_raw"].shape == (3, 2)
    assert result["X_train_scaled"].shape == (5, 2)

    # Target shapes
    assert result["y_train_per_qoi"]["QOI_1"].shape == (5, 3)
    assert result["y_val_per_qoi"]["QOI_1"].shape == (2, 3)
    assert result["y_test_per_qoi"]["QOI_1"].shape == (3, 3)

    # Feature scaling bounds
    x_train_num = result["X_train_scaled"][:, 1]
    assert np.isclose(x_train_num.min(), 0.0)
    assert np.isclose(x_train_num.max(), 1.0)

    # Validation stays within [0, 1]; test exceeds bounds.
    x_val_num = result["X_val_scaled"][:, 1]
    assert x_val_num.min() >= 0.0
    assert x_val_num.max() <= 1.0

    x_test_num = result["X_test_scaled"][:, 1]
    assert x_test_num.min() < 0.0  # value 5 < training min 10
    assert x_test_num.max() > 1.0  # value 60 > training max 50

    # Target scaling bounds
    y_train = result["y_train_per_qoi"]["QOI_1"]
    assert np.isclose(y_train.min(), 0.0)
    assert np.isclose(y_train.max(), 1.0)

    y_test = result["y_test_per_qoi"]["QOI_1"]
    assert y_test.min() < 0.0
    assert y_test.max() > 1.0

    # Scalers present
    assert isinstance(result["x_scaler"], MinMaxScaler)
    assert isinstance(result["y_scalers"]["QOI_1"], MinMaxScaler)
