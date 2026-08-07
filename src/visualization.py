from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def visualize(y_test, predicted_classes):
    cm = pd.DataFrame(confusion_matrix(y_test, predicted_classes),
                      columns=['(0) \n Legitimate', '(1) \n Fraud'],
                      index=['Legitimate \n (0)     ', 'Fraud\n (1)     '])
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, cmap='PuRd', fmt="d")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()

