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
