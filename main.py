from src.config import CNNConfig, SplitConfig

from src.model_selection import final_evaluation, Model
from src.data_pipeline import create_experiment_split, load_experiment_data
from src.load_data import download_dataset, load_dataset
from src.metrics import calculate_avg_metrics, save_avg_results


def main() -> None:
    test_seed = 27082026
    validation_seed = [37082027,
                       47082027,
                       57082027,
                       67082027,
                       77082027]


    # cnn_candidates = [
    #     CNNConfig(batch_size=16, candidate_id=1, filters=(2, 4), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=4),
    #     CNNConfig(batch_size=16, candidate_id=2, filters=(4, 8), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=4),
    #     CNNConfig(batch_size=16, candidate_id=3, filters=(8, 16), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=4),
    #     CNNConfig(batch_size=16, candidate_id=4, filters=(2, 4), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8),
    #     CNNConfig(batch_size=16, candidate_id=5, filters=(4, 8), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8),
    #     CNNConfig(batch_size=16, candidate_id=6, filters=(8, 16), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8),
    #     CNNConfig(batch_size=16, candidate_id=7, filters=(2, 4), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=8, filters=(4, 8), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=9, filters=(8, 16), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=10, filters=(8, 16), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=16, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=11, filters=(2, 4), kernel_size=(29, 15),
    #               strides=(1, 14), padding=('same', 'valid'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=12, filters=(8, 16), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'valid'), fc_units=16, threshold=0.2),
    #     CNNConfig(batch_size=16, candidate_id=13, filters=(2, 4), kernel_size=(29, 15),
    #               strides=(1, 14), padding=('same', 'same'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=14, filters=(2, 4), kernel_size=(29, 15),
    #               strides=(1, 14), padding=('same', 'valid'), fc_units=16, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=15, filters=(2, 4), kernel_size=(29, 29),
    #               strides=(1, 1), padding=('same', 'same'), fc_units=8, threshold=0.3),
    #     CNNConfig(batch_size=16, candidate_id=16, filters=(4, 8), kernel_size=(29, 15),
    #               strides=(1, 14), padding=('same', 'same'), fc_units=8, threshold=0.3)
    # ]


    # for candidate in cnn_candidates:
    #     candidate_results = []
    #     for run in range(0, len(validation_seed)):
    #         experiment_split = SplitConfig(f"experiment_split_v{run + 1}",
    #                                        split_seed=(test_seed, validation_seed[run]),
    #                                        test_fraction=0.10,
    #                                        validation_fraction=0.10)
    #         model = Model("cnn", experiment_split, config=candidate)
    #         model.create_model()
    #         model.run_model()
    #         run_results = model.save_results()
    #         candidate_results.append(run_results)
    #     avg_candidate_results = calculate_avg_metrics(candidate_results)
    #     save_avg_results(avg_candidate_results)
    #
    all_results = []
    for run in range(0, len(validation_seed)):
        experiment_split = SplitConfig(f"experiment_split_v{run + 6}",
                                       split_seed=(test_seed, validation_seed[run]),
                                       test_fraction=0.10,
                                       validation_fraction=0.10)
        model = Model("logistic_regression", experiment_split)
        model.create_model()
        model.run_model()
        run_results = model.save_results()
        all_results.append(run_results)
    avg_results = calculate_avg_metrics(all_results)
    save_avg_results(avg_results)


if __name__ == "__main__":
    main()
