from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5

load_dotenv()

project_root = Path(__file__).resolve().parents[1]
raw_data_dir = project_root / "data" / "raw"

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")
kaggle_dataset = os.getenv("KAGGLE_DATASET")

