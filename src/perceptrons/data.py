"""Loading the datasets in data/, feature scaling, and splitting."""

# TODO: load fraud_dataset.csv, digits.csv, digits_test.csv, more_digits.csv
# TODO: Exercise 1 learning comparison uses every sample; only the generalization study splits
# TODO: the split strategy for the Exercise 1 generalization study
# TODO: Exercises 2 and 3 tune on digits.csv only; digits_test.csv stands in for production

import pandas as pd
from pathlib import Path
import numpy as np
import ast


# Importing all training data: 
digits_path = Path(__file__).parent.parent.parent / "data" / "digits.csv"
digits = pd.read_csv(digits_path)

m, n = digits.shape
split_index = int(len(digits) * 0.8)

Y = digits["label"].to_numpy()
X = np.array(
    digits["image"].apply(ast.literal_eval).tolist()
)

# Importing additional Exercise 3 data:
more_digits_path = Path(__file__).parent.parent.parent / "data" / "more_digits.csv"
more_digits = pd.read_csv(more_digits_path)

# Own split index: more_digits has a different number of rows than digits
more_split_index = int(len(more_digits) * 0.8)

Y_more = more_digits["label"].to_numpy()
X_more = np.array(
    more_digits["image"].apply(ast.literal_eval).tolist()
)

# Split Exercise 3 training and validation data
# (checked: the file is not sorted by label, so an unshuffled split is fine)
Y_more_train = Y_more[:more_split_index]
X_more_train = X_more[:more_split_index].T

Y_more_valid = Y_more[more_split_index:]
X_more_valid = X_more[more_split_index:].T




# Split training and validation data
Y_train = Y[:split_index]
X_train = X[:split_index].T

Y_valid = Y[split_index:]
X_valid = X[split_index:].T


# Importing all test data:
digits_test_path = Path(__file__).parent.parent.parent / "data" / "digits_test.csv"
digits_test = pd.read_csv(digits_test_path)

Y_test = digits_test["label"].to_numpy()

X_test = np.array(
    digits_test["image"].apply(ast.literal_eval).tolist()
).T


def one_hot_encode(Y: np.ndarray, n_classes: int) -> np.ndarray:
    # n_classes is explicit: inferring it from Y.max() + 1 gives fewer rows when the
    # highest class happens to be missing from Y (e.g. a small batch or subset)
    one_hot_Y = np.zeros((Y.size, n_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)

