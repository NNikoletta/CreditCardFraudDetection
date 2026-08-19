import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class DevelopmentData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray

@dataclass(frozen=True)
class TestData:
    x_test: np.ndarray
    y_test: np.ndarray
