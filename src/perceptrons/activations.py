"""Activation functions and their derivatives, vectorized over numpy arrays."""
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