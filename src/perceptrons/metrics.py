"""Evaluation metrics computed from predictions and targets, independent of any model."""

# TODO: metrics for the Exercise 1 generalization study (the report must justify the choice)
# TODO: whatever the fraud-threshold recommendation to CompanyX is based on (Exercise 1)
# TODO: digit classification metrics (Exercises 2 and 3; client target is accuracy >= 98%)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy(cm: np.array) -> np.float32:
    return np.trace(cm) / np.sum(cm)

def recall(cm: np.array) -> np.array:
    row_sums = np.maximum(np.sum(cm, axis=1), 1)
    return np.diag(cm) / row_sums
    # return [cm[i][i] / sum(cm[i]) for i in range(len(cm))]
    
def confusion_matrix(y_true: np.array, y_pred: np.array) -> np.array:
    size = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((size, size), dtype=int)

    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    return cm

def plot_digits_confusion_matrix(cm: np.array) -> None:
    # A class with no samples (8 in digits.csv) would divide 0 by 0; leave its row at 0
    row_sums = np.maximum(np.sum(cm, axis=1).reshape(-1,1), 1)
    cm_normalized = np.round(cm/row_sums, 2)
    sns.heatmap(cm_normalized, cmap="Blues", annot=True, xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()