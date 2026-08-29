import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


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


@dataclass(frozen=True)
class ExperimentData:
    unscaled_development: DevelopmentData
    scaled_development: DevelopmentData
    test_data: TestData
    scaler: StandardScaler


@dataclass(frozen=True)
class FinalEvaluation:
    unscaled_train_data: TestData
    scaled_train_data: TestData
    test_data: TestData
    scaler: StandardScaler
