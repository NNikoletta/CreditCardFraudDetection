from src.config import raw_data_dir, kaggle_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils import ensure_dir
import pandas as pd
import numpy as np

raw_data_file = 'creditcard.csv'


def download_dataset():
    ensure_dir(raw_data_dir)  # this ensures that the directory for the data exists
    if raw_data_dir.is_dir() and any(raw_data_dir.iterdir()):  # if directory exists AND it's not empty
        print("Dataset already exists.")  # assume that raw data exists
    else:  # otherwise
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(dataset=kaggle_dataset, path=raw_data_dir, unzip=True)  # download data from kaggle
        print("Dataset successfully downloaded.")


def load_dataset():
    df_tmp = pd.read_csv(raw_data_dir/raw_data_file, header=0, usecols=lambda column: column != 0)
    df = df_tmp.to_numpy()
    x = df[:, range(1, 30)]
    y = df[:, 30].astype(int)
    return x, y
