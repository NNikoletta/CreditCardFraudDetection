from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize(test_label, predicted_classes):
    true_classes = np.argmax(test_label, axis=1)
    cm = pd.DataFrame(confusion_matrix(true_classes, predicted_classes),
                      columns=['(0) \nTrue', '(1) \nFraud'],
                      index=['True\n (0)     ', 'Fraud\n (1)     '])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.2)
    sn.heatmap(cm, annot=True, cmap='crest', fmt="d")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

