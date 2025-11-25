import pytest
import numpy as np
from sklearn.decomposition import PCA
from surrogate_factory.preprocessing.processor import OutputCompressor

def test_fit_transform_shape_fixa():
    Y_raw = np.random.rand(50, 100)

    compressor = OutputCompressor(n_components=3)
    Z = compressor.fit_transform(Y_raw)

    assert compressor.is_fitted is True
    assert Z.shape == (50,3)

def test_transform_consistency_train_validation():
    Y_train = np.random.rand(50,100)
    Y_val = np.random.rand(20,100)

    compressor = OutputCompressor(n_components=0.99)

    Z_train = compressor.fit_transform(Y_train)
    n_components_learned = Z_train.shape[1]

    Z_val = compressor.transform(Y_val)

    assert Z_val.shape[0] == 20
    assert Z_val.shape[1] == n_components_learned

def test_inverse_transform_decoder():
    Y_raw = np.random.rand(50,100)

    compressor = OutputCompressor(n_components=5)
    Z = compressor.fit_transform(Y_raw)
        
    Y_reconstruido = compressor.inverse_transform(Z)

    assert Y_reconstruido.shape == (50, 100)

