"""SurrogateFactory - A factory for building surrogate models of expensive simulations.

Supports Kriging (Gaussian Process), RBF (Radial Basis Function), and
Neural Network surrogate models with PCA-compressed time-series outputs.
"""

from .factory import ModelType, SurrogateFactory

__version__ = "0.1.0"
__all__ = ["ModelType", "SurrogateFactory"]
