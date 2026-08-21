from src.load_data import download_dataset, load_dataset
from src.data_splits import load_split
from src.preprocess import prepare_data, standardize
from src.config import SplitConfig

from src.model_selection import select_and_run_model


def main() -> None:
    split_config = SplitConfig()
    download_dataset()
    x, y = load_dataset()
    train_indices, validation_indices, test_indices = load_split(split_config.split_id, y)
    development_data, test_data = prepare_data(x, y, train_indices, validation_indices, test_indices)
    scaler, development_data = standardize(development_data)
    select_and_run_model("mlp", development_data)


if __name__ == "__main__":
    main()
