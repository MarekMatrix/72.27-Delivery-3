"""Multilayer perceptron with a configurable architecture, e.g. [2, 3, 2, 1]."""

# TODO: must learn XOR with [2, 2, 1] and [2, 3, 2, 1] (validation)
# TODO: forward and backward passes as matrix operations, not per-neuron loops
# TODO: 10-class digit classification (Exercises 2 and 3)

# starting making the perceptron multilayer with layer 0 being all inputs of the digits, 
# layer 1 being og 16 perceptrons and layer 2 as well, end layer will be of 10 perceptrons,
# one for each number

import matplotlib.pyplot as plt
import numpy as np
from perceptrons.data import X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# SHould be possible to have differrent numbers of hidden layers with different sizes for testing
# Should probably normalize the input
# SHould not initialize weights with so high weights
# Different types of gradient decent updating of weights should be explored (online, mini-batch, batch)

# Temporarily random seed: 
random_seed = np.random.default_rng(42)
hidden_layers = [16, 16]
input_size = 784 # Size x[0]
output_size = 10 # Size y.max

def init_params(layers: list[int], rnd: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Xavier initialization: the bound shrinks with layer size, so the signal neither
    # vanishes nor saturates as it passes through the layers
    W = []
    b = []

    for i in range(len(layers) - 1):
        cols = layers[i]
        rows = layers[i + 1]
        limit = np.sqrt(6 / (rows + cols))
        W.append(rnd.uniform(-limit, limit, size=(rows, cols)))
        b.append(np.zeros((rows, 1)))
    
    return W, b

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

def mse(Y_pred, Y_true):
    error = ((Y_pred - Y_true) ** 2).sum() / (2 * Y_pred.size) # Divide by batch_size?
    return error

def accuracy(Y_pred, Y_true): 
    acc = Y_pred.argmax(axis=0) == Y_true.argmax(axis=0)
    return acc.mean()

def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)

def forward_propagation(beta: float, function: str, layers: list[int], X: np.ndarray, W: list[np.ndarray], b: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    theta, _ = activation_functions[function]
    Z = []
    A = [X]
    
    for i in range(len(W)):
        Z.append(W[i] @ A[i] + b[i])
        A.append(theta(Z[i], beta))
    O = A[-1]
    
    return Z, A, O

def backward_propagation(beta: float, function: str, Y: np.ndarray, W: list[np.ndarray], Z: list[np.ndarray], O: np.ndarray) -> list[np.ndarray]:
    _, dtheta = activation_functions[function]
    delta = [None] * len(W)
    delta[-1] = (Y - O) * dtheta(Z[-1], beta)
    for i in range(len(W) - 2, -1, -1):
        delta[i] = W[i + 1].T @ delta[i + 1] * dtheta(Z[i], beta)
    return delta

def update_weights(eta: float, batch_size: int, W: list[np.ndarray], b: list[np.ndarray], A: list[np.ndarray], delta: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]: 
    for i in range(len(W)):
        W[i] = W[i] + eta * delta[i] @ A[i].T / batch_size
        b[i] = b[i] + eta * delta[i].sum(axis=1, keepdims=True) / batch_size
    return W, b

# Right now the function doesnt do anything, but would want it to choose the type of nonlinear function to be used 
def multilayer(eta: float, beta: float, function: str, hidden_layers: list[int], max_epocs: int, rnd: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X = X_train
    Y = one_hot(Y_train)
    batch_size = X_train.shape[1] # For now here, but should be a hyperparameter (and should also affect the amount of input samples are used to update per iteration)
    input_size = 784
    output_size = 10
    layers = np.concatenate([[input_size], hidden_layers, [output_size]])
    W, b = init_params(layers, rnd)
    # Doing the batch full type (i think)
    for _ in range(max_epocs):
        Z, A, O = forward_propagation(beta, function, layers, X, W, b)
        delta = backward_propagation(beta, function, Y, W, Z, O)
        W, b = update_weights(eta, batch_size, W, b, A, delta)
        
        # mse averages over all 10 outputs, most of them near 0, so 0.01 was reached
        # at ~90% training accuracy and stopped training early
        E = mse(O, Y)
        if E < 1e-4:
            break
    
    return W, b


from perceptrons.metrics import confusion_matrix
from perceptrons.metrics import plot_confusion_matrix
def test_function(beta, function, layers, X_eval, Y_eval, W, b, name="Validation"):
    _, _, O = forward_propagation(beta, function, layers, X_eval, W, b)
    acc = accuracy(O, one_hot(Y_eval))
    print("{} accuracy: {:.3f}".format(name, acc))
    confusion_matrix(Y_eval, one_hot_decode(O))
    plot_confusion_matrix(Y_eval, one_hot_decode(O), 10)

beta = 1
eta = 0.5
W, b = multilayer(eta, beta, "sigmoid", [64], 2000, random_seed)
# Tune on the validation split; digits_test.csv is only for the final result
test_function(beta, "sigmoid", [64], X_valid, Y_valid, W, b)


"""
# Different methods for updating, online, m   ini-batch and batch:
def update_weights(method):
    if method == "online":
        pass
    elif method == "mini-batch":
        pass
    elif method == "batch":
        pass
    else: 
        Exception "No method called " + method
    pass
    
"""