"""Validation exercises for the multilayer perceptron: not submitted, but they catch implementation bugs."""

# TODO: one forward and backward step of [2, 2, 1] matches your hand calculation
# TODO: one forward and backward step of [2, 3, 2, 1] matches your hand calculation
# TODO: learns XOR: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [1, 1, -1, -1]
# test 

from perceptrons.metrics import confusion_matrix, accuracy
from perceptrons.plots import plot_confusion_matrix
from perceptrons.multilayer import multilayer, forward_propagation, one_hot_decode
from perceptrons.data import X_valid, Y_valid, X_train, Y_train
from perceptrons.optimizer import GradientDecent, Momentum, Adam
from pathlib import Path
import numpy as np


def test_multilayer():
    random_seed = np.random.default_rng(42)
    beta = 1
    eta = 0.5
    optimizer = GradientDecent(beta)
    W, b = multilayer(eta, beta, X_train, Y_train, optimizer, "sigmoid", [64], 2000, random_seed)
    _, _, O = forward_propagation(beta, "sigmoid", X_valid, W, b)
    Y_pred = one_hot_decode(O)
    cm = confusion_matrix(Y_valid, Y_pred, 10)
    acc = accuracy(cm)
    print("{} accuracy: {:.3f}".format("Validation", acc))
    path = Path("results/confusion_matrix_digits.png")
    plot_confusion_matrix(cm, path, "Confusion matrix of digits")
    assert acc > 0.6


# Tune on the validation split; digits_test.csv is only for the final result
