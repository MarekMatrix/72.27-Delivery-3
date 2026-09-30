"""Validation exercises for the multilayer perceptron: not submitted, but they catch implementation bugs."""

# TODO: one forward and backward step of [2, 2, 1] matches your hand calculation
# TODO: one forward and backward step of [2, 3, 2, 1] matches your hand calculation
# TODO: learns XOR: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [1, 1, -1, -1]
# test 

from perceptrons.metrics import confusion_matrix, accuracy
from perceptrons.plots import plot_confusion_matrix, plot_error_convergence
from perceptrons.multilayer import multilayer, forward_propagation, one_hot_decode, one_hot_encode
from perceptrons.data import X_valid, Y_valid, X_train, Y_train
from perceptrons.optimizers import GradientDescent, Momentum, Adam
from pathlib import Path
import numpy as np


def test_multilayer():
    random_seed = np.random.default_rng(42)
    beta = 1
    eta = 0.5
    optimizer = GradientDescent(eta)
    W, b, training_error, validation_error = multilayer(beta, X_train, one_hot_encode(Y_train), X_valid, one_hot_encode(Y_valid), optimizer, "sigmoid", [64], 2000, random_seed)
    _, _, O = forward_propagation(beta, "sigmoid", X_valid, W, b)
    
    plot_error_convergence(training_error, validation_error, Path("results/error_convergence.png"), "Error Convergence of Digits")
    
    Y_pred = one_hot_decode(O)
    cm = confusion_matrix(Y_valid, Y_pred, 10)
    acc = accuracy(cm)
    print("{} accuracy: {:.3f}".format("Validation", acc))
    plot_confusion_matrix(cm, Path("results/confusion_matrix_digits.png"), "Confusion Matrix of Digits")
    assert acc > 0.6


# Tune on the validation split; digits_test.csv is only for the final result
