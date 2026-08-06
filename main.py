from models import neural_networks as nn
from src import load_data, preprocess
from src import visualization as vs
from src.metrics import calculate_metrics


def main() -> None:
    load_data.download_dataset()
    x, y = load_data.load_dataset()
    x_train, x_valid, x_test, train_label, valid_label, test_label, scaler = preprocess.prepare_data(x, y)

    network = nn.PipelineTestModel()
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    calculate_metrics(test_label, predicted_classes)
    vs.visualize(test_label, predicted_classes)


if __name__ == "__main__":
    main()
