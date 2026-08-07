import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 10


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 100
    max_depth: int = 3
    random_state: int = 42
    learning_rate: float = 0.1
    gamma: float = 0
    min_child_weight: float = 1


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

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")
kaggle_dataset = os.getenv("KAGGLE_DATASET")
