from sklearn.preprocessing import StandardScaler

from src.load_data import download_dataset, load_dataset
from src.data_splits import create_split_indices, save_split, load_split
from src.data_splits import create_eval_split_indices, save_eval_split, load_eval_split
from src.preprocess import prepare_data, standardize, prepare_eval_data, standardize_eval_data
from src.config import SplitConfig
from src.data_types import ExperimentData, FinalEvaluation


def create_experiment_split(config: SplitConfig) -> None:
    download_dataset()
    _, y = load_dataset()
    train_indices, validation_indices, test_indices = create_split_indices(y, config)
    save_split(y, train_indices, validation_indices, test_indices, config)


def create_eval_split(config: SplitConfig) -> None:
    download_dataset()
    _, y = load_dataset()
    train_indices, test_indices = create_eval_split_indices(y, config)
    save_eval_split(y, train_indices, test_indices, config)


def load_experiment_data(config: SplitConfig) -> tuple[ExperimentData, StandardScaler]:
    download_dataset()
    x, y = load_dataset()
    train_indices, validation_indices, test_indices = load_split(config.split_id, y)
    unscaled_development, test_data = prepare_data(x, y, train_indices, validation_indices, test_indices)
    scaled_development, scaler = standardize(unscaled_development)

    experiment_data = ExperimentData(unscaled_development=unscaled_development,
                                     scaled_development=scaled_development,
                                     test_data=test_data,
                                     scaler=scaler)

    return experiment_data, scaler


def load_eval_data(config: SplitConfig) -> tuple[FinalEvaluation, StandardScaler]:
    download_dataset()
    x, y = load_dataset()
    train_indices, test_indices = load_eval_split(config.split_id, y)

    unscaled_train_data, test_data = prepare_eval_data(x, y, train_indices, test_indices)
    scaled_train_data, scaler = standardize_eval_data(unscaled_train_data)

    final_evaluation_data = FinalEvaluation(unscaled_train_data=unscaled_train_data,
                                            scaled_train_data=scaled_train_data,
                                            test_data=test_data,
                                            scaler=scaler)

    return final_evaluation_data, scaler


