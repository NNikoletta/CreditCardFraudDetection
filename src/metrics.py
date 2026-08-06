import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def calculate_metrics(test_label, predicted_classes):
    true_classes = np.argmax(test_label, axis=1)
    f1 = f1_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes)
    print('F1 score: ', round(f1, ndigits=2))
    print('Recall score: ', round(recall, ndigits=2))  # out of all yes casses, how many did the model catch
    print('Precision score: ', round(precision, ndigits=2))  # when the model says yes, how often is it right
    return f1, recall, precision
