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
    ax.set_ylabel(r"MSE $= \frac{1}{2\,n_{samples}}\sum (O - Y)^2$ (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Fixed series colours, in this order, so the same slot always means the same thing
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

STUDY_LABELS = {
    "learning_rate": "Learning rate",
    "architecture": "Hidden layers",
    "optimizer": "Optimizer",
    "batch_size": "Batch size",
}


def study_value(config, study: str) -> str:
    if study == "learning_rate":
        return f"{config.learning_rate:g}"
    if study == "architecture":
        return "-".join(str(h) for h in config.hidden_layers)
    if study == "optimizer":
        return f"{config.optimizer_method.value}\n(lr {config.learning_rate:g})"
    return str(config.batch_size)


def _sort_key(config, study: str):
    if study == "architecture":
        return (len(config.hidden_layers), config.hidden_layers)
    if study == "optimizer":
        return ["gradient_descent", "momentum", "adam"].index(config.optimizer_method.value)
    return config.learning_rate if study == "learning_rate" else config.batch_size


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_study_accuracy(groups: dict, study: str, path: Path) -> None:
    """Mean validation accuracy of each variant, with the min-max over seeds as an error bar."""
    items = sorted(groups.values(), key=lambda item: _sort_key(item[0], study))
    labels = [study_value(c, study) for c, _ in items]
    accs = [np.array(a) * 100 for _, a in items]
    means = np.array([a.mean() for a in accs])
    err = np.array([[m - a.min() for m, a in zip(means, accs)], [a.max() - m for m, a in zip(means, accs)]])

    fig, ax = plt.subplots(figsize=(max(6, 1.2 + 1.1 * len(labels)), 4))
    x = np.arange(len(labels))
    ax.bar(x, means, width=0.6, color=SERIES[0])
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor="#52514e", capsize=4, linewidth=1.2)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.1f}", (xi, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    low = max(0, means.min() - 10)
    ax.set_ylim(low, 100)
    ax.set_xticks(x, labels)
    ax.set_xlabel(STUDY_LABELS[study])
    ax.set_ylabel("Validation accuracy (%)")
    n_seeds = len(accs[0])
    ax.set_title(f"{STUDY_LABELS[study]}: validation accuracy after training\n(mean of {n_seeds} seeds, bars = min/max)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_study_curves(runs: list, study: str, path: Path) -> None:
    """Validation loss per epoch for each variant (seed 0 only, so the lines stay readable)."""
    runs = sorted([r for r in runs if r[0].random_seed == 0], key=lambda r: _sort_key(r[0], study))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (config, result) in enumerate(runs):
        epochs = np.arange(1, len(result["valid_loss"]) + 1)
        ax.plot(epochs, result["valid_loss"], color=SERIES[i % len(SERIES)], linewidth=2,
                label=study_value(config, study))

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation MSE $= \frac{1}{2\,n_{samples}}\sum (O - Y)^2$ (log scale)")
    ax.set_title(f"{STUDY_LABELS[study]}: validation loss per epoch (seed 0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title=STUDY_LABELS[study], fontsize=8)
    _save(fig, path)


def plot_epoch_time(runs: list, study: str, path: Path) -> None:
    """Mean wall-clock seconds per epoch of each variant."""
    groups = {}
    for config, result in runs:
        groups.setdefault(study_value(config, study), (config, []))[1].append(np.mean(result["epoch_seconds"]))
    items = sorted(groups.items(), key=lambda kv: _sort_key(kv[1][0], study))

    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(items), 3.6))
    x = np.arange(len(items))
    secs = [np.mean(t) for _, (_, t) in items]
    ax.bar(x, secs, width=0.6, color=SERIES[0])
    for xi, sec in zip(x, secs):
        ax.annotate(f"{sec:.2f}", (xi, sec), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(x, [k for k, _ in items])
    ax.set_xlabel(STUDY_LABELS[study])
    ax.set_ylabel("Time per epoch (s)")
    ax.set_title(f"{STUDY_LABELS[study]}: training time per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_recall_valid_vs_test(valid_recall: list, test_recall: list, path: Path) -> None:
    """Per-class recall on the validation split and on digits_test.csv, side by side."""
    n_classes = len(test_recall)
    x = np.arange(n_classes)
    valid = np.array([np.nan if r is None else r * 100 for r in valid_recall])
    test = np.array([np.nan if r is None else r * 100 for r in test_recall])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, np.nan_to_num(valid), width=0.36, color=SERIES[0], label="Validation (digits.csv)")
    ax.bar(x + 0.2, np.nan_to_num(test), width=0.36, color=SERIES[1], label="Test (digits_test.csv)")
    for xi, v in zip(x, valid):
        if np.isnan(v):
            ax.annotate("no\nsamples", (xi - 0.2, 2), ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, [str(c) for c in range(n_classes)])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("Per-class recall of the best model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    _save(fig, path)


def plot_digit_grid(images: np.ndarray, captions: list[str], path: Path, title: str, cols: int = 8) -> None:
    """images has one 784-pixel sample per column, like X in data.py."""
    n = images.shape[1]
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.1 * cols, 1.3 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[:, i].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(captions[i], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_class_distribution(train_labels: np.ndarray, test_labels: np.ndarray, path: Path, n_classes: int = 10) -> None:
    """Share of each digit in digits.csv and in digits_test.csv."""
    x = np.arange(n_classes)
    train = np.bincount(train_labels, minlength=n_classes) / len(train_labels) * 100
    test = np.bincount(test_labels, minlength=n_classes) / len(test_labels) * 100

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, train, width=0.36, color=SERIES[0], label=f"digits.csv (n = {len(train_labels)})")
    ax.bar(x + 0.2, test, width=0.36, color=SERIES[1], label=f"digits_test.csv (n = {len(test_labels)})")
    for xi, t in zip(x, train):
        if t == 0:
            ax.annotate("0", (xi - 0.2, 0.3), ha="center", fontsize=9, color="#52514e")
    ax.set_xticks(x, [str(c) for c in range(n_classes)])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Share of samples (%)")
    ax.set_title("Class distribution: learning data vs production stand-in")
    ax.set_ylim(0, max(train.max(), test.max()) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    _save(fig, path)
