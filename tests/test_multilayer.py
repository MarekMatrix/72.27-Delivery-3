"""Validation exercises for the multilayer perceptron: not submitted, but they catch implementation bugs."""

# TODO: one forward and backward step of [2, 2, 1] matches your hand calculation
# TODO: one forward and backward step of [2, 3, 2, 1] matches your hand calculation
# TODO: learns XOR: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [1, 1, -1, -1]
# test 

from perceptrons.metrics import confusion_matrix, accuracy
from perceptrons.plots import plot_confusion_matrix
from perceptrons.multilayer import multilayer, forward_propagation, one_hot_encoding, one_hot_decode
from perceptrons.data import X_valid, Y_valid
from pathlib import Path
import numpy as np
# Temporarily random seed: 
random_seed = np.random.default_rng(42)

def test_function(beta, function, X_eval, Y_eval, W, b, name="Validation"):
    _, _, O = forward_propagation(beta, function, X_eval, W, b)
    Y_pred = one_hot_decode(O)
    acc = accuracy(O, Y_pred)
    print("{} accuracy: {:.3f}".format(name, acc))
    cm = confusion_matrix(Y_eval, Y_pred, 10)
    path = Path("results/confusion_matrix_digits")
    plot_confusion_matrix(cm, path, "Confusion matrix of digits")

beta = 1
eta = 0.5
W, b = multilayer(eta, beta, "sigmoid", [64], 2000, random_seed)
# Tune on the validation split; digits_test.csv is only for the final result
test_function(beta, "sigmoid", X_valid, Y_valid, W, b)