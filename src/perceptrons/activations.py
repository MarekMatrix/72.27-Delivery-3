"""Activation functions and their derivatives, vectorized over numpy arrays."""

# TODO: step (validation: AND)
# TODO: identity (linear simple perceptron)
# TODO: tanh (validation: fit y = tanh(x))
# TODO: an activation for Exercise 1, whose output is a fraud probability in [0, 1]
# TODO: optional, Exercise 1 — ReLU, and how it changes the conclusions

import numpy as np

# Theta functions: 
def sigmoid(X: np.ndarray, beta: float) -> np.ndarray: 
    return 1 / (1 + np.exp(-2 * beta * X))

def tangent(X: np.ndarray, beta: float) -> np.ndarray: 
    return np.tanh(beta * X)

def dsigmoid(X: np.ndarray, beta: float) -> np.ndarray: 
    sigmoid_val = sigmoid(X, beta)
    return 2 * beta * sigmoid_val * (1 - sigmoid_val)

def dtangent(X: np.ndarray, beta: float) -> np.ndarray: 
    tangent_val = tangent(X, beta)
    return beta * (1 - tangent_val ** 2)

activation_functions = {
    "sigmoid": (sigmoid, dsigmoid),
    "tangent": (tangent, dtangent)
}