import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)  # creates directory if it doesn't already exist


def calculate_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes,
                                   y=y_train)  # custom weights are calculated based on the distribution of the training labels
    custom_weights = {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }
    return custom_weights
