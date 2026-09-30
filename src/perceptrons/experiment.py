"""Runs a configured experiment and records it under results/. Never plots."""

# TODO: report progress while training
# TODO: record loss per epoch, hyperparameters, and run time
# TODO: save a trained model with its config, and load it back to resume training

import numpy as np
from perceptrons.data import one_hot_encode, one_hot_decode

def run(config, X_train: np.ndarray, Y_train: np.ndarray, X_valid: np.ndarray, Y_valid: np.ndarray, rng: np.random.Generator) -> dict:
    n_classes = 10
    n_features = 784
    Y_train = one_hot_encode(Y_train, n_classes)
    Y_valid = one_hot_encode(Y_valid, n_classes)
    layers = [n_features] + config.hidden_layers + [n_classes]