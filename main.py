from src.load_data import download_dataset, load_dataset
from src.data_splits import load_split
from src.preprocess import prepare_data, standardize
from src.config import SplitConfig, LogisticRegressionConfig
from src.visualization import visualize
from src.metrics import calculate_metrics
from models.linear import LogisticRegressionModel


def main() -> None:
    split_config = SplitConfig()
    download_dataset()
    x, y = load_dataset()
    train_indices, validation_indices, test_indices = load_split(split_config.split_id, y)
    development_data, test_data = prepare_data(x, y, train_indices, validation_indices, test_indices)
    scaler, development_data = standardize(development_data)

    model_config = LogisticRegressionConfig()
    model = LogisticRegressionModel(model_config)
    model.train(development_data.x_train, development_data.y_train)
    predicted_classes, predicted_probabilities = model.predict(development_data.x_valid)
    model.evaluate(development_data.y_valid, predicted_classes, predicted_probabilities)
    calculate_metrics(development_data.y_valid, predicted_classes, predicted_probabilities)
    visualize(development_data.y_valid, predicted_classes)

if __name__ == "__main__":
    main()
