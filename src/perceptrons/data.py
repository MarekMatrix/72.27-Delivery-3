"""Loading the datasets in data/, feature scaling, and splitting."""

# TODO: load fraud_dataset.csv, digits.csv, digits_test.csv, more_digits.csv
# TODO: Exercise 1 learning comparison uses every sample; only the generalization study splits
# TODO: the split strategy for the Exercise 1 generalization study
# TODO: Exercises 2 and 3 tune on digits.csv only; digits_test.csv stands in for production

import pandas as pd
from pathlib import Path
import numpy as np
import ast


digits_path = Path(__file__).parent.parent.parent / "data" / "digits.csv"
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

def one_hot_encode(Y: np.ndarray, n_classes: int) -> np.ndarray:
    # n_classes is explicit: inferring it from Y.max() + 1 gives fewer rows when the
    # highest class happens to be missing from Y (e.g. a small batch or subset)
    one_hot_Y = np.zeros((Y.size, n_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)