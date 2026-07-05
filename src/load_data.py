from src.config import raw_data_dir, kaggle_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils import ensure_dir

def load_dataset():
    ensure_dir(raw_data_dir)
    if raw_data_dir.is_dir() and any(raw_data_dir.iterdir()):
        print("Dataset already exists.")
    else:
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(dataset=kaggle_dataset, path=raw_data_dir, unzip=True)
        print("Dataset successfully downloaded.")
