from src.config import CNNConfig, SplitConfig

from src.model_selection import final_evaluation, Model
from src.data_pipeline import create_experiment_split, load_experiment_data, create_eval_split
from src.load_data import download_dataset, load_dataset
from src.metrics import calculate_avg_metrics, save_avg_results


def main() -> None:
    test_seed = 27082026
    validation_seed = [37082027,
                       47082027,
                       57082027,
                       67082027,
                       77082027]

    final_eval_split = SplitConfig(split_id="final_eval_split_v1",
                                   split_seed=(test_seed, 0),
                                   test_fraction=0.10,
                                   validation_fraction=0.0)

    cnn17_config = CNNConfig(batch_size=16, candidate_id=17, filters=(2, 4, 8), kernel_size=(29, 29, 29),
                             strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8)
    cnn19_config = CNNConfig(batch_size=16, candidate_id=19, filters=(2, 4, 8), kernel_size=(29, 29, 29),
                             strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8, threshold=0.4)

    final_evaluation("cnn", final_eval_split=final_eval_split, config=cnn17_config)



if __name__ == "__main__":
    main()
