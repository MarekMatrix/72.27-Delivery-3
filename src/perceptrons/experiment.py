"""Runs a configured experiment and records it under results/. Never plots."""

# TODO: report progress while training
# TODO: record loss per epoch, hyperparameters, and run time
# TODO: save a trained model with its config, and load it back to resume training

import numpy as np
from perceptrons.data import one_hot_encode, one_hot_decode
from perceptrons.multilayer import MLP
from perceptrons.losses import MSE
from perceptrons.optimizers import Optimizer
from perceptrons.training import History
from perceptrons.metrics import accuracy, recall, confusion_matrix
from perceptrons.plots import plot_confusion_matrix, plot_error_convergence

def run(config, X_train: np.ndarray, Y_train: np.ndarray, X_valid: np.ndarray, Y_valid: np.ndarray, rng: np.random.Generator) -> dict:
    n_classes = X_train.shape[0]
    Y_train = one_hot_encode(Y_train, n_classes)
    Y_valid = one_hot_encode(Y_valid, n_classes)
    n_features = Y_train.shape[0]
    
    layers = [n_features] + config.hidden_layers + [n_classes]
    
    multilayer = MLP()
    loss = MSE()
    optimizer = config.optimizer
    history = train()
    
    plot_error_convergence(training_error, validation_error, Path("results/error_convergence.png"), "Error Convergence of Digits")
    
    Y_pred = one_hot_decode(O)
    cm = confusion_matrix(Y_valid, Y_pred, 10)
    acc = accuracy(cm)
    print("{} accuracy: {:.3f}".format("Validation", acc))
    plot_confusion_matrix(cm, Path("results/confusion_matrix_digits.png"), "Confusion Matrix of Digits")
    assert acc > 0.6
    training_error.append(mse(O, Y))
    validation_error.append(mse(O_valid, Y_valid))
    
    return W, b, training_error, validation_error