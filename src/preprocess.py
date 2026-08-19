import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_types import DevelopmentData, TestData


def prepare_data(x: np.ndarray, y: np.ndarray,
                 train_indices: np.ndarray,
                 validation_indices: np.ndarray,
                 test_indices: np.ndarray):  # Preparation of data for processing

    development_data = DevelopmentData(x_train=x[train_indices],
                                       y_train=y[train_indices],
                                       x_valid=x[validation_indices],
                                       y_valid=y[validation_indices])

    test_data = TestData(x_test=x[test_indices], y_test=y[test_indices])

    return development_data, test_data


def standardize(development_data: DevelopmentData):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(development_data.x_train)
    x_valid_scaled = scaler.transform(development_data.x_valid)
    scaled_development_data = DevelopmentData(x_train=x_train_scaled,
                                              y_train=development_data.y_train,
                                              x_valid=x_valid_scaled,
                                              y_valid=development_data.y_valid)
    return scaler, scaled_development_data
