"""Report figures, built only from what experiment.py saved under results/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = "Confusion matrix") -> None:
    """Save a row-normalized confusion matrix: each row sums to 1, so the diagonal is per-class recall.

    Works for any number of classes (10 digits, or 2 for fraud). A class with no samples
    (e.g. 8 in digits.csv) is drawn as an all-zero row instead of dividing 0 by 0.
    """
    n_classes = cm.shape[0]
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_normalized = cm / row_sums

    fig, ax = plt.subplots(figsize=(1 + 0.6 * n_classes, 0.6 * n_classes))
    sns.heatmap(
        cm_normalized,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        xticklabels=range(n_classes),
        yticklabels=range(n_classes),
        cbar_kws={"label": "Fraction of actual class (row-normalized)"},
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{title}\n(n = {cm.sum()} samples; diagonal = recall)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_error_convergence(training_error: list[float], validation_error: list[float], path: Path, title: str = "Error Convergence") -> None:
    """Save training and validation error per epoch.

    The y-axis is logarithmic: on a linear axis the first few epochs dominate and the
    slow late-stage descent (is it still improving? do the curves diverge?) is invisible.
    """
    epochs = np.arange(1, len(training_error) + 1)

    fig, ax = plt.subplots()

    ax.plot(epochs, training_error, label="Training error")
    ax.plot(epochs, validation_error, label="Validation error")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"MSE $= \frac{1}{2\,n_{out}\,n_{samples}}\sum (O - Y)^2$ (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
# Testing plotting of numbers
#image = X_train[:, 0].reshape((28, 28))
#plt.imshow(image, cmap="gray")
#plt.show()