"""
SurrogateFactory - A factory for creating surrogate models for expensive simulations.

Supports Kriging, RBF, and Neural Network models with PCA-compressed outputs.
"""

from .factory import SurrogateFactory, ModelType

__version__ = "0.1.0"
__all__ = ["SurrogateFactory", "ModelType"]
