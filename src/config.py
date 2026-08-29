import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class SplitConfig:
    split_id: str = "fixed_split_v2"
    split_seed: tuple[int, ...] = (27082026, 27082027)
    test_fraction: float = 0.10
    validation_fraction: float = 0.10


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    threshold: float = 0.5
    keras_random_seed: int = 400


@dataclass(frozen=True)
class CNNConfig(TrainingConfig):
    candidate_id: int = 0
    filters: tuple[int, ...] = (8, 16)
    kernel_size: tuple[int, ...] = (29, 29)
    strides: tuple[int, ...] = (1, 1)
    padding: tuple[str, ...] = ('same', 'same')
    fc_units: int = 4
    # dropout: float = 0


@dataclass(frozen=True)
class LogisticRegressionConfig:
    random_state: int = 42
    class_weight: str = None  # default = None, other = 'balanced'
    solver: str = 'lbfgs'  # default = 'lbfgs', other = 'newton-cholesky'
    penalty: str = 'l2'  # default = 'l2'
    max_iter: int = 100  # default = 100


@dataclass(frozen=True)
class XGBoostConfig:
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = 3
    random_state: int = 42
    gamma: float = 0
    min_child_weight: float = 1
    ratio: float = 1  # default = 1, other sqrt(expected_legitimate_count/expected_fraudulent_count)


@dataclass(frozen=True)
class DataValidationConfig:
    raw_data_file: str = "creditcard.csv"
    expected_row_count: int = 284807
    expected_column_count: int = 31
    expected_legitimate_count: int = 284315
    expected_fraudulent_count: int = 492
    expected_columns: tuple[str, ...] = (
        "Time",
        *tuple(f"V{i}" for i in range(1, 29)),
        "Amount",
        "Class"
    )


load_dotenv()

project_root = Path(__file__).resolve().parents[1]
raw_data_dir = project_root / "data" / "raw"
split_dir = project_root / "data" / "splits"
results_dir = project_root / "results"

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")
kaggle_dataset = os.getenv("KAGGLE_DATASET")
