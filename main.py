from pathlib import Path

from models import neural_networks as nn
from models import decision_trees as dt
from src import load_data, preprocess
from src import visualization as vs
from src.metrics import calculate_metrics
from src.config import project_root


def run_logistic_regression(x_train, x_test, y_train, y_test, valid):
    print('Running Logistic Regression...')
    model = nn.LogisticRegressionModel()
    model.train(x_train, y_train)
    predicted_classes, predicted_probabilities = model.predict(x_test, y_test, valid=valid)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics


def run_cnn(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running the regular CNN...')
    network = nn.CNN()
    network.train(x_train, y_train, x_valid, y_valid)
    network.evaluate(x_test, y_test, valid=valid)
    predicted_classes, predicted_probabilities = network.predict(x_test)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics


def run_attention_cnn(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running the Attention CNN...')
    network = nn.AttentionCNN()
    network.train(x_train, y_train, x_valid, y_valid)
    network.evaluate(x_test, y_test, valid=valid)
    predicted_classes, predicted_probabilities = network.predict(x_test)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics


def run_xgboost(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running XGBoost...')
    model = dt.XGBoostModel()
    history = model.train(x_train, x_valid, y_train, y_valid)
    predicted_classes, predicted_probabilities = model.predict(x_test, y_test, valid=valid)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics


def main() -> None:
    load_data.download_dataset()
    x, y = load_data.load_dataset()
    scaled_data, regular_data, labels, scaler = preprocess.prepare_data(x, y)

    validation = True
    results_path = project_root/"results"/"experiments"/"cnn_experiments"

    if validation:
        confusion_matrices = []
        all_metrics = []
        for i in range(0, 5):
            # cm, metrics = run_logistic_regression(x_train=scaled_data['x_train_scaled'],
            #                                       x_test=scaled_data['x_valid_scaled'],
            #                                       y_train=labels['y_train'],
            #                                       y_test=labels['y_valid'], valid=True)

            cm, metrics = run_cnn(x_train=scaled_data['x_train_scaled'],
                                  x_valid=scaled_data['x_valid_scaled'],
                                  x_test=scaled_data['x_valid_scaled'],
                                  y_train=labels['y_train'],
                                  y_valid=labels['y_valid'],
                                  y_test=labels['y_valid'], valid=True)

            # cm, metrics = run_attention_cnn(x_train=scaled_data['x_train_scaled'],
            #                                 x_valid=scaled_data['x_valid_scaled'],
            #                                 x_test=scaled_data['x_valid_scaled'],
            #                                 y_train=labels['y_train'],
            #                                 y_valid=labels['y_valid'],
            #                                 y_test=labels['y_valid'], valid=True)

            # cm, metrics = run_xgboost(x_train=regular_data['x_train'],
            #                           x_valid=regular_data['x_valid'],
            #                           x_test=regular_data['x_valid'],
            #                           y_train=labels['y_train'],
            #                           y_valid=labels['y_valid'],
            #                           y_test=labels['y_valid'], valid=True)

            confusion_matrices.append(cm)
            all_metrics.append(metrics)

        all_f1 = []
        all_recall = []
        all_precision = []
        all_avg_precision = []
        all_roc_auc = []
        for i in range(0, 5):
            all_f1.append(all_metrics[i]['f1'])
            all_recall.append(all_metrics[i]['recall'])
            all_precision.append(all_metrics[i]['precision'])
            all_avg_precision.append(all_metrics[i]['avg_precision'])
            all_roc_auc.append(all_metrics[i]['roc_auc'])

        file_dir = results_path/"CNN_exp1.txt"
        with open(file_dir, "w") as f:
            for i in range(0, 5):
                f.write(f"CNN Results of run {i}")
                f.write("Metrics:")
                f.write(str(all_metrics[i]))
                f.write("Confusion matrix")
                f.write(str(confusion_matrices[i]))

    else:
        run_logistic_regression(x_train=scaled_data['x_train_scaled'], x_test=scaled_data['x_test_scaled'],
                                y_train=labels['y_train'], y_test=labels['y_test'], valid=False)

        run_cnn(x_train=scaled_data['x_train_scaled'], x_valid=scaled_data['x_valid_scaled'],
                x_test=scaled_data['x_test_scaled'],
                y_train=labels['y_train'], y_valid=labels['y_valid'], y_test=labels['y_test'], valid=True)

        run_attention_cnn(x_train=scaled_data['x_train_scaled'], x_valid=scaled_data['x_valid_scaled'],
                          x_test=scaled_data['x_test_scaled'],
                          y_train=labels['y_train'], y_valid=labels['y_valid'], y_test=labels['y_test'], valid=True)

        run_xgboost(x_train=regular_data['x_train'], x_valid=regular_data['x_valid'], x_test=regular_data['x_test'],
                    y_train=labels['y_train'], y_valid=labels['y_valid'], y_test=labels['y_test'], valid=True)


if __name__ == "__main__":
    main()
