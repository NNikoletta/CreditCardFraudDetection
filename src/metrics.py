from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, roc_auc_score


def calculate_metrics(y_test, predicted_classes, predicted_probabilities):
    f1 = f1_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, zero_division=0)
    avg_precision = average_precision_score(y_test, predicted_probabilities)
    roc_auc = roc_auc_score(y_test, predicted_probabilities)
    print('F1 score: ', round(f1, ndigits=3))
    print('Recall score: ', round(recall, ndigits=3))  # out of all yes cases, how many did the model catch
    print('Precision score: ', round(precision, ndigits=3))  # when the model says yes, how often is it right
    print('Average precision: ', round(avg_precision, ndigits=3))
    print('ROC-AUC: ', round(roc_auc, ndigits=3))
    metrics = {'f1': f1,
               'recall': recall,
               'precision': precision,
               'avg_precision': avg_precision,
               'roc_auc': roc_auc}
    return metrics
