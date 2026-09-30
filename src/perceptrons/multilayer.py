"""Multilayer perceptron with a configurable architecture, e.g. [2, 3, 2, 1]."""

# TODO: must learn XOR with [2, 2, 1] and [2, 3, 2, 1] (validation)
# TODO: forward and backward passes as matrix operations, not per-neuron loops
# TODO: 10-class digit classification (Exercises 2 and 3)

# starting making the perceptron multilayer with layer 0 being all inputs of the digits, 
# layer 1 being og 16 perceptrons and layer 2 as well, end layer will be of 10 perceptrons,
# one for each number

import numpy as np
from perceptrons.activations import activation_functions
from perceptrons.optimizers import Optimizer

# Different types of gradient decent updating of weights should be explored (online, mini-batch, batch)

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

def mse(Y_pred, Y_true):
    error = ((Y_pred - Y_true) ** 2).sum() / (2 * Y_pred.size) # Divide by batch_size?
    return error

def one_hot_encode(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)

def forward_propagation(beta: float, function: str, X: np.ndarray, W: list[np.ndarray], b: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
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
    delta[-1] = (O - Y) * dtheta(Z[-1], beta)
    for i in range(len(W) - 2, -1, -1):
        delta[i] = W[i + 1].T @ delta[i + 1] * dtheta(Z[i], beta)
    return delta

def gradient(batch_size: int, A: list[np.ndarray], delta: list[np.ndarray]) -> list[np.ndarray]:
    grad_W = []
    grad_b = []
    for i in range(len(delta)):
        grad_W.append(delta[i] @ A[i].T / batch_size)
        grad_b.append(delta[i].sum(axis=1, keepdims=True) / batch_size)
    grads = grad_W + grad_b
    return grads

def multilayer(beta: float, X_train: np.ndarray, Y_train: np.ndarray, X_valid: np.ndarray, Y_valid: np.ndarray, optimizer: Optimizer, function: str, hidden_layers: list[int], max_epocs: int, rnd: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    X = X_train
    Y = Y_train
    batch_size = X.shape[1] # For now here, but should be a hyperparameter (and should also affect the amount of input samples are used to update per iteration)
    input_size = X.shape[0]
    output_size = Y.shape[0]
    layers = np.concatenate([[input_size], hidden_layers, [output_size]])
    W, b = init_params(layers, rnd)
    params = W + b
    training_error = []
    validation_error = []
    for _ in range(max_epocs):
        Z, A, O = forward_propagation(beta, function, X, W, b)
        delta = backward_propagation(beta, function, Y, W, Z, O)
        grads = gradient(batch_size, A, delta)
        optimizer.step(params, grads)
        training_error.append(mse(O, Y))
        _, _, O_valid = forward_propagation(beta, function, X_valid, W, b)
        validation_error.append(mse(O_valid, Y_valid))
    
    return W, b, training_error, validation_error