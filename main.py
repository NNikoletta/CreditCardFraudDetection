from models import neural_networks as nn
from models import decision_trees as dt
from src import load_data, preprocess
from src import visualization as vs
from src.metrics import calculate_metrics


def run_neural_network_CNN(x_train, x_valid, x_test, y_train, y_valid, y_test):
    network = nn.CNN()
    network.train(x_train, y_train, x_valid, y_valid)
    network.evaluate(x_test, y_test)
    predicted_classes = network.predict(x_test)
    calculate_metrics(y_test, predicted_classes)
    vs.visualize(y_test, predicted_classes)


def run_xgboost(x_train, x_valid, x_test, y_train, y_valid, y_test):
    model = dt.XGBoostModel()
    history = model.train(x_train, x_valid, y_train, y_valid)
    predicted_classes = model.predict(x_test, y_test)
    calculate_metrics(y_test, predicted_classes)
    vs.visualize(y_test, predicted_classes)


def main() -> None:
    load_data.download_dataset()
    x, y = load_data.load_dataset()
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = preprocess.prepare_data(x, y)
    run_neural_network_CNN(x_train, x_valid, x_test, y_train, y_valid, y_test)
    # run_xgboost(x_train, x_valid, x_test, y_train, y_valid, y_test)


if __name__ == "__main__":
    main()
