from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, roc_auc_score


def calculate_metrics(y_test, predicted_classes, predicted_probabilities) -> dict[str, float]:
    f1 = f1_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, zero_division=0)
    avg_precision = average_precision_score(y_test, predicted_probabilities)
    roc_auc = roc_auc_score(y_test, predicted_probabilities)
    metrics = {'f1': round(f1, ndigits=3),
               'recall': round(recall, ndigits=3),
               'precision': round(precision, ndigits=3),
               'avg_precision': round(avg_precision, ndigits=3),
               'roc_auc': round(roc_auc, ndigits=3)}
    print_metrics(metrics)
    return metrics


def print_metrics(metrics: dict) -> None:
    print('F1 score: ', metrics['f1'])
    print('Recall score: ', metrics['recall'])  # out of all yes cases, how many did the model catch
    print('Precision score: ', metrics['precision'])  # when the model says yes, how often is it right
    print('Average precision: ', metrics['avg_precision'])
    print('ROC-AUC: ', metrics['roc_auc'])
