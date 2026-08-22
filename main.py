from src.config import SplitConfig, CNNConfig
from src.metrics import calculate_avg_metrics, save_avg_results

from src.model_selection import Model


def main() -> None:
    test_seed = 21082026
    validation_seed = [31082027,
                       41082027,
                       51082027,
                       61082027,
                       71082027]

    cnn_candidates = [
        CNNConfig(batch_size=16,
                  candidate_id=1,
                  filters=(2, 4),
                  kernel_size=(29, 29),
                  strides=(1, 1),
                  padding=('same', 'valid'),
                  fc_units=(4,)),
        CNNConfig(candidate_id=2,
                  filters=(4, 8),
                  kernel_size=(29, 29),
                  strides=(1, 1),
                  padding=('same', 'valid'),
                  fc_units=(4,))
    ]

    for candidate in cnn_candidates:
        candidate_results = []
        for run in range(0, len(validation_seed)):
            experiment_split = SplitConfig(f"experiment_split_v{run+1}",
                                           split_seed=(test_seed, validation_seed[run]),
                                           test_fraction=0.10,
                                           validation_fraction=0.10)
            model = Model("cnn", experiment_split, config=candidate)
            model.create_model()
            model.run_model()
            run_results = model.save_results()
            candidate_results.append(run_results)
        avg_candidate_results = calculate_avg_metrics(candidate_results)
        save_avg_results(avg_candidate_results)

    all_results = []
    for run in range(0, len(validation_seed)):
        experiment_split = SplitConfig(f"experiment_split_v{run+1}",
                                       split_seed=(test_seed, validation_seed[run]),
                                       test_fraction=0.10,
                                       validation_fraction=0.10)
        model = Model("xgboost", experiment_split)
        model.create_model()
        model.run_model()
        run_results = model.save_results()
        all_results.append(run_results)
    avg_results = calculate_avg_metrics(all_results)
    save_avg_results(avg_results)


if __name__ == "__main__":
    main()
