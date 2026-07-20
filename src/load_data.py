from src.config import raw_data_dir, kaggle_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils import ensure_dir
from pathlib import Path
import pandas as pd
import numpy as np

raw_data_file = 'creditcard.csv'


def download_dataset():
    ensure_dir(raw_data_dir)  # this ensures that the directory for the data exists
    if Path(raw_data_dir/raw_data_file).is_file():  # if directory exists AND has the correct file
        print("Dataset already exists.")  # assume that raw data exists
    else:  # otherwise
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(dataset=kaggle_dataset, path=raw_data_dir, unzip=True)  # download data from kaggle
        print("Dataset successfully downloaded.")


def load_dataset():
    df_tmp = pd.read_csv(raw_data_dir/raw_data_file)
    x = df_tmp.drop(columns=["Time", "Class"]).to_numpy(dtype=np.float64)
    y = df_tmp["Class"].to_numpy(dtype=np.int32)

    return x, y
