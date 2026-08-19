import numpy as np

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
    predicted_classes, predicted_probabilities, test_metrics = model.predict(x_test, y_test, valid=valid)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics, test_metrics


def run_mlp(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running MLP...')
    network = nn.MLP()
    network.train(x_train, y_train, x_valid, y_valid)
    test_metrics = network.evaluate(x_test, y_test, valid=valid)
    predicted_classes, predicted_probabilities = network.predict(x_test)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics, test_metrics


def run_cnn(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running the regular CNN...')
    network = nn.CNN()
    network.train(x_train, y_train, x_valid, y_valid)
    test_metrics = network.evaluate(x_test, y_test, valid=valid)
    predicted_classes, predicted_probabilities = network.predict(x_test)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics, test_metrics


def run_attention_cnn(x_train, x_valid, x_test, y_train, y_valid, y_test, valid):
    print('Running the Attention CNN...')
    network = nn.AttentionCNN()
    network.train(x_train, y_train, x_valid, y_valid)
    test_metrics = network.evaluate(x_test, y_test, valid=valid)
    predicted_classes, predicted_probabilities = network.predict(x_test)
    metrics = calculate_metrics(y_test, predicted_classes, predicted_probabilities[:, 1])
    cm = vs.visualize(y_test, predicted_classes)
    return cm, metrics, test_metrics


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

    validation = True
    runs = 5
    results_path = project_root/"results"/"outdated_experiments"

    if validation:
        confusion_matrices = []
        all_metrics = []
        validation_metrics = []
        for i in range(0, runs):
            scaled_data, regular_data, labels, scaler = preprocess.prepare_data(x, y)
            cm, metrics, valid_metric = run_cnn(x_train=scaled_data['x_train_scaled'],
                                                x_valid=scaled_data['x_valid_scaled'],
                                                x_test=scaled_data['x_valid_scaled'],
                                                y_train=labels['y_train'],
                                                y_valid=labels['y_valid'],
                                                y_test=labels['y_valid'], valid=True)

            # cm, metrics, valid_metric = run_logistic_regression(x_train=scaled_data['x_train_scaled'],
            #                                                     x_test=scaled_data['x_valid_scaled'],
            #                                                     y_train=labels['y_train'],
            #                                                     y_test=labels['y_valid'], valid=True)

            confusion_matrices.append(cm)
            all_metrics.append(metrics)
            validation_metrics.append(valid_metric)

        all_f1 = []
        all_recall = []
        all_precision = []
        all_avg_precision = []
        all_roc_auc = []
        all_loss = []
        all_acc = []
        for i in range(0, runs):
            all_f1.append(all_metrics[i]['f1'])
            all_recall.append(all_metrics[i]['recall'])
            all_precision.append(all_metrics[i]['precision'])
            all_avg_precision.append(all_metrics[i]['avg_precision'])
            all_roc_auc.append(all_metrics[i]['roc_auc'])
            all_loss.append(validation_metrics[i]['loss'])
            all_acc.append(validation_metrics[i]['accuracy']*100)

        file_dir = results_path/"CNN_exp5_2.txt"
        with open(file_dir, "w") as f:
            for i in range(0, runs):
                f.write(f"CNN Results of run {i+1}\n")
                f.write(f"Validation loss: {np.round(all_loss[i], decimals=4)} \n")
                f.write(f"Validation accuracy: {np.round(all_acc[i], decimals=4)}% \n")
                f.write("Metrics:\n")
                f.write(str(all_metrics[i])+"\n")
                f.write("Confusion matrix\n")
                f.write(str(confusion_matrices[i])+"\n")
                f.write("\n")
            f.write(f"Validation loss> mean: {np.round(np.mean(np.array(all_loss)), decimals=4)} std: {np.round(np.std(np.array(all_loss)), decimals=4)}\n")
            f.write(f"Validation accuracy> mean: {np.round(np.mean(np.array(all_acc)), decimals=4)} std: {np.round(np.std(np.array(all_acc)), decimals=4)}\n")
            f.write(f"F1 score> mean: {np.round(np.mean(np.array(all_f1)), decimals=4)} std: {np.round(np.std(np.array(all_f1)), decimals=4)}\n")
            f.write(f"Recall> mean: {np.round(np.mean(np.array(all_recall)), decimals=4)} std: {np.round(np.std(np.array(all_recall)), decimals=4)}\n")
            f.write(f"Precision> mean: {np.round(np.mean(np.array(all_precision)), decimals=4)} std: {np.round(np.std(np.array(all_precision)), decimals=4)}\n")
            f.write(f"Average precision> mean: {np.round(np.mean(np.array(all_avg_precision)), decimals=4)} std: {np.round(np.std(np.array(all_avg_precision)), decimals=4)}\n")
            f.write(f"ROC-AUC score> mean: {np.round(np.mean(np.array(all_roc_auc)), decimals=4)} std: {np.round(np.std(np.array(all_roc_auc)), decimals=4)}\n")


if __name__ == "__main__":
    main()
