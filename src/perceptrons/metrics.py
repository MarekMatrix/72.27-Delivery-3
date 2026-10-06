"""Evaluation metrics computed from predictions and targets, independent of any model."""

import numpy as np

def accuracy(cm: np.ndarray) -> float:
    return np.trace(cm) / np.sum(cm)

def recall(cm: np.ndarray) -> np.ndarray:
    row_sums = np.sum(cm, axis=1)

    return np.where(
        row_sums == 0,
        np.nan,
        np.diag(cm) / row_sums
    )
    
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    return cm