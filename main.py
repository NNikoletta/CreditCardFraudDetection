from models import neural_networks as nn
from models import decision_trees as dt
from src import load_data, preprocess
from src import visualization as vs
from src.metrics import calculate_metrics

import numpy as np


def call_nn(x_train, x_valid, x_test, train_label, valid_label, test_label):
    network = nn.CNN()
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    calculate_metrics(test_label, predicted_classes)
    vs.visualize(test_label, predicted_classes)


def main() -> None:
    load_data.download_dataset()
    x, y = load_data.load_dataset()
    x_train, x_valid, x_test, train_label, valid_label, test_label, scaler = preprocess.prepare_data(x, y)
    # call_nn(x_train, x_valid, x_test, train_label, valid_label, test_label)
    y_test = np.argmax(test_label, axis=1)
    y_train = np.argmax(train_label, axis=1)
    y_val = np.argmax(valid_label, axis=1)
    predicted_classes = dt.xg(x_train, y_train, x_test, y_test, x_valid, y_val)
    calculate_metrics(test_label, predicted_classes)
    vs.visualize(test_label, predicted_classes)

if __name__ == "__main__":
    main()
