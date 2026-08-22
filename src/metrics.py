import json
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, roc_auc_score

from src.config import SplitConfig, results_dir
from src.utils import ensure_dir


def calculate_metrics(y_test, predicted_classes, predicted_probabilities) -> dict[str, float]:
    f1 = f1_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, zero_division=0)
    avg_precision = average_precision_score(y_test, predicted_probabilities)
    roc_auc = roc_auc_score(y_test, predicted_probabilities)
    metrics = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'avg_precision': avg_precision,
               'roc_auc': roc_auc}
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: dict) -> None:
    print('F1 score: ', round(metrics['f1'], ndigits=3))
    print('Recall score: ', round(metrics['recall'], ndigits=3))  # out of all yes cases, how many did the model catch
    print('Precision score: ', round(metrics['precision'], ndigits=3))  # when the model says yes, how often is it right
    print('Average precision: ', round(metrics['avg_precision'], ndigits=3))
    print('ROC-AUC: ', round(metrics['roc_auc'], ndigits=3))


def calculate_avg_metrics(all_results: list) -> dict:
    loss = []
    acc = []
    f1 = []
    recall = []
    precision = []
    avg_precision = []
    roc_auc = []
    training_time = []
    prediction_time = []
    for i in range(len(all_results)):
        loss.append(all_results[i]['results']['loss'])
        acc.append(all_results[i]['results']['accuracy'])
        f1.append(all_results[i]['results']['f1'])
        recall.append(all_results[i]['results']['recall'])
        precision.append(all_results[i]['results']['precision'])
        avg_precision.append(all_results[i]['results']['avg_precision'])
        roc_auc.append(all_results[i]['results']['roc_auc'])
        training_time.append(all_results[i]['runtime_seconds']['training_time'])
        prediction_time.append(all_results[i]['runtime_seconds']['prediction_time'])

    avg_results = {'model_name': all_results[0]['model_name'],
                   'model_configuration': all_results[0]['model_configuration'],
                   'avg_results': {'loss_avg_std': (np.mean(loss), np.std(loss)),
                                   'accuracy_avg_std': (np.mean(acc), np.std(acc)),
                                   'f1_avg_std': (np.mean(f1), np.std(f1)),
                                   'recall_avg_std': (np.mean(recall), np.std(recall)),
                                   'precision_avg_std': (np.mean(precision), np.std(precision)),
                                   'avg_precision_avg_std': (np.mean(avg_precision), np.std(avg_precision)),
                                   'roc_auc_avg_std': (np.mean(roc_auc), np.std(roc_auc))},
                   'runtime_seconds_avg': {'training_time_avg_std': (np.mean(training_time), np.std(training_time)),
                                           'prediction_time_avg_std': (np.mean(prediction_time), np.std(prediction_time))}
                   }

    return avg_results


def save_avg_results(avg_results: dict):
    model_name = avg_results['model_name']
    results_folder_path = results_dir / model_name
    file_id = f"{model_name}_avg_results.json"
    if model_name == "cnn" or model_name == "attention_cnn":
        candidate_id = avg_results['model_configuration']['candidate_id']
        results_folder_path = results_folder_path / f"{model_name}_candidate_{candidate_id}"
        file_id = f"{model_name}_candidate_{candidate_id}_avg_results.json"
    ensure_dir(results_folder_path)
    file_path = results_folder_path / file_id

    if file_path.is_file():
        print(f"Experimental results '{file_id}' already exist and will not be overwritten.")
        return

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(avg_results, json_file, indent=2)

