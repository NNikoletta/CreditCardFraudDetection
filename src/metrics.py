import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def calculate_metrics(y_test, predicted_classes):
    f1 = f1_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes)
    print('F1 score: ', round(f1, ndigits=3))
    print('Recall score: ', round(recall, ndigits=3))  # out of all yes cases, how many did the model catch
    print('Precision score: ', round(precision, ndigits=3))  # when the model says yes, how often is it right
    return f1, recall, precision
