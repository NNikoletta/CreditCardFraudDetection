from src.config import CNNConfig, SplitConfig
from src.model_selection import final_evaluation
from src.data_pipeline import create_eval_split


def main() -> None:
    test_seed = 27082026

    final_eval_split = SplitConfig(split_id="final_eval_split_v1",
                                   split_seed=(test_seed, 0),
                                   test_fraction=0.10,
                                   validation_fraction=0.0)
    create_eval_split(final_eval_split)

    cnn17_config = CNNConfig(batch_size=16, candidate_id=17, filters=(2, 4, 8), kernel_size=(29, 29, 29),
                             strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8)

    cnn19_config = CNNConfig(batch_size=16, candidate_id=19, filters=(2, 4, 8), kernel_size=(29, 29, 29),
                             strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8, threshold=0.4)

    final_evaluation("logistic_regression", final_eval_split=final_eval_split)
    final_evaluation("mlp", final_eval_split=final_eval_split)
    final_evaluation("cnn", final_eval_split=final_eval_split, config=cnn17_config)
    final_evaluation("cnn", final_eval_split=final_eval_split, config=cnn19_config)
    final_evaluation("xgboost", final_eval_split=final_eval_split)


if __name__ == "__main__":
    main()
