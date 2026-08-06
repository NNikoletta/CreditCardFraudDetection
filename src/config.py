import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5


@dataclass(frozen=True)
class DataValidationConfig:
    raw_data_file: str = "creditcard.csv"
    expected_row_count: int = 284807
    expected_column_count: int = 31
    expected_true_transaction: int = 284315
    expected_fraudulent_transaction: int = 492
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
