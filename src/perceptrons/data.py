"""Loading the datasets in data/, feature scaling, and splitting."""

# TODO: load fraud_dataset.csv, digits.csv, digits_test.csv, more_digits.csv
# TODO: Exercise 1 learning comparison uses every sample; only the generalization study splits
# TODO: the split strategy for the Exercise 1 generalization study
# TODO: Exercises 2 and 3 tune on digits.csv only; digits_test.csv stands in for production

import pandas as pd
from pathlib import Path
import numpy as np
import ast
from sklearn.model_selection import train_test_split


digits_path = Path(__file__).parent.parent.parent / "data" / "digits.csv"
more_digits_path = Path(__file__).parent.parent.parent / "data" / "more_digits.csv"
digits_test_path = Path(__file__).parent.parent.parent / "data" / "digits_test.csv"

def load_digits(path: Path) -> list[np.ndarray]:
    digits = pd.read_csv(path)
    Y = digits["label"].to_numpy()
    X = np.array(digits["image"].apply(ast.literal_eval).tolist()).T
    return X, Y

def split_train_and_validation(X: np.ndarray, Y: np.ndarray) -> list[np.ndarray]:
    split_index = int(X.shape[1] * 0.8)

    X_train = X[:, :split_index]
    Y_train = Y[:split_index]

    X_valid = X[:, split_index:]
    Y_valid = Y[split_index:]

    return X_train, X_valid, Y_train, Y_valid

def split_train_and_validation_stratified(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into 80% training and 20% validation, preserving class proportions."""
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X.T, Y,
        test_size=0.2,
        random_state=seed,
        stratify=Y,
    )

    return X_train.T, X_valid.T, Y_train, Y_valid

def oversample_training(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance observed classes by repeating training samples."""
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(Y, return_counts=True)
    target_count = counts.max()
    selected_indices = []

    for label in labels:
        class_indices = np.flatnonzero(Y == label)
        extra_count = target_count - len(class_indices)

        extra_indices = rng.choice(
            class_indices, size=extra_count, replace=True
        )
        selected_indices.append(
            np.concatenate([class_indices, extra_indices])
        )

    indices = np.concatenate(selected_indices)
    rng.shuffle(indices)

    return X[:, indices], Y[indices]

def add_unique_training_samples(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    extra_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Add extra images without duplicates or overlap with validation."""
    X_extra, Y_extra = load_digits(extra_path)

    seen = {
        image.tobytes()
        for image in np.concatenate([X_train, X_valid], axis=1).T
    }

    keep = []
    for i, image in enumerate(X_extra.T):
        key = image.tobytes()
        if key not in seen:
            keep.append(i)
            seen.add(key)

    return (
        np.concatenate([X_train, X_extra[:, keep]], axis=1),
        np.concatenate([Y_train, Y_extra[keep]]),
    )

def one_hot_encode(Y: np.ndarray, n_classes: int) -> np.ndarray:
    # n_classes is explicit: inferring it from Y.max() + 1 gives fewer rows when the
    # highest class happens to be missing from Y (e.g. a small batch or subset)
    one_hot_Y = np.zeros((Y.size, n_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)

