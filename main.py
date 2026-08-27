from src.config import CNNConfig, SplitConfig

from src.model_selection import final_evaluation, Model


def main() -> None:
    cnn_config = CNNConfig(batch_size=16, candidate_id=13, filters=(2, 4), kernel_size=(29, 15),
                           strides=(1, 14), padding=('same', 'same'), fc_units=8, threshold=0.3)

    final_evaluation("logistic_regression")
    final_evaluation("mlp")
    final_evaluation("cnn", config=cnn_config)
    final_evaluation("xgboost")


if __name__ == "__main__":
    main()
