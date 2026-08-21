from src.data_pipeline import load_experiment_data, create_experiment_split
from src.config import SplitConfig, TrainingConfig

from src.model_selection import Model


def main() -> None:
    # test_seed = 21082026
    # validation_seed = [31082027,
    #                    41082027,
    #                    51082027,
    #                    61082027,
    #                    71082027]
    #
    # for run in range(0, len(validation_seed)):
    #     experiment_split = SplitConfig(f"experiment_split_v{run+1}",
    #                                    split_seed=(test_seed, validation_seed[run]),
    #                                    test_fraction=0.10,
    #                                    validation_fraction=0.10)
    #     create_experiment_split(experiment_split)

    split_config = SplitConfig()
    model = Model("mlp", split_config)
    model.create_model()
    model.run_model()
    model.save_results()


if __name__ == "__main__":
    main()
