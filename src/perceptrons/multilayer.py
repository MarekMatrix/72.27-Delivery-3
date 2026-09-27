"""Multilayer perceptron with a configurable architecture, e.g. [2, 3, 2, 1]."""

# TODO: must learn XOR with [2, 2, 1] and [2, 3, 2, 1] (validation)
# TODO: forward and backward passes as matrix operations, not per-neuron loops
# TODO: 10-class digit classification (Exercises 2 and 3)

# starting making the perceptron multilayer with layer 0 being all inputs of the digits, 
# layer 1 being og 16 perceptrons and layer 2 as well, end layer will be of 10 perceptrons,
# one for each number

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import ast


# SHould be possible to have differrent numbers of hidden layers with different sizes for testing
# Should probably normalize the input
# SHould not initialize weights with so high weights
# Different types of gradient decent updating of weights should be explored (online, mini-batch, batch)
# Should get the data from the data.py file instead of creating them here (they are already there as well)






# Temporarily random seed: 
random_seed = np.random.default_rng(42)

# Importing all training data: 
# Importing all training data:
# SHOULD GET THE DATA D`FROM THE DATA FILE!
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

print(Y_train.shape)
print(X_train.shape)
print(Y_valid.shape)
print(X_valid.shape)


# Importing all test data:
digits_test_path = Path(__file__).parent.parent.parent / "data" / "digits_test.csv"
digits_test = pd.read_csv(digits_test_path)

Y_test = digits_test["label"].to_numpy()

X_test = np.array(
    digits_test["image"].apply(ast.literal_eval).tolist()
).T

print(Y_test.shape)
print(X_test.shape)


image = X_train[:, 0].reshape((28, 28))

# Testing plotting of numbers
#plt.imshow(image, cmap="gray")
#plt.show()

output_size = 10

# Weights matrices: 
def create_initial_weight_matrix(rows: int, cols: int, min_val: int, max_val: int, rnd: np.random.Generator) -> np.ndarray:
    W = rnd.uniform(min_val, max_val, size=(rows, cols))
    return W

# First working for one type of setup, then expanding for letting it be sent in as an input. 
def init_params(rnd: np.random.Generator): # what to write for output?
    N = X_train.shape[0]
    M = 16
    L = 10
    W1 = create_initial_weight_matrix(M, N, 0, 5, rnd)
    W2 = create_initial_weight_matrix(M, M, 0, 5, rnd)
    W3 = create_initial_weight_matrix(L, M, 0, 5, rnd)
    
    b1 = create_initial_weight_matrix(M, 1, 0, 5, rnd)
    b2 = create_initial_weight_matrix(M, 1, 0, 5, rnd)
    b3 = create_initial_weight_matrix(L, 1, 0, 5, rnd)
    
    return W1, W2, W3, b1, b2, b3

# Theta function: 
# ndarray or array???
def theta(X: np.array, beta: np.float32) -> np.ndarray: 
    return sigmoid(X, beta)

def dtheta(X: np.array, beta: np.float32) -> np.array: 
    return dsigmoid(X, beta)

def sigmoid(X: np.array, beta: np.float32) -> np.array: 
    return 1 / (1 + np.exp(-2 * beta * X))

def tangent(X: np.array, beta: np.float32) -> np.array: 
    return np.tanh(beta * X)

def dsigmoid(X: np.array, beta: np.float32) -> np.array: 
    sigmoid_val = sigmoid(X, beta)
    return 2 * beta * sigmoid_val * (1 - sigmoid_val)

def dtangent(X: np.array, beta: np.float32) -> np.array: 
    tangent_val = tangent(X, beta)
    return beta * (1 - tangent_val ** 2)

def mse(Y_pred, Y_true):
    error = ((Y_pred - Y_true) ** 2).sum() / (2 * Y_pred.size)
    return error

def accuracy(Y_pred, Y_true): 
    acc = Y_pred.argmax(axis=0) == Y_true.argmax(axis=0)
    return acc.mean()

def one_hot(Y:np.arange) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_propagation(beta: np.float32, X: np.array, W1: np.array, W2: np.array, W3: np.array, b1: np.array, b2: np.array, b3: np.array) -> np.ndarray:
    Z1 = W1 @ X + b1
    A1 = theta(Z1, beta)
    Z2 = W2 @ A1 + b2
    A2 = theta(Z2, beta)
    Z3 = W3 @ A2 + b3
    O = theta(Z3, beta)
    return Z1, A1, Z2, A2, Z3, O

def backward_propagation(beta: np.float32, T: np.array, Z1: np.array, Z2: np.array, Z3: np.array, W1: np.array, W2: np.array, W3: np.array, A1: np.array, A2: np.array, O: np.array) -> np.array:
    deltaO = (T - O) * dtheta(Z3, beta) # I think the problem might be that the onehot is not correct size? or that all the others are wrong?? 
    deltaZ2 = W3.T @ deltaO * dtheta(Z2, beta)
    deltaZ1 = W2.T @ deltaZ2 * dtheta(Z1, beta) # How to make sure I get the sum
    return deltaO, deltaZ2, deltaZ1

def update_weights(eta: np.float32, X: np.array, A1: np.array, A2: np.array, W1: np.array, W2: np.array, W3: np.array, b1: np.array, b2: np.array, b3: np.array, deltaO: np.array, deltaZ2: np.array, deltaZ1: np.array) -> np.array: 
    W1 = W1 + eta * deltaZ1 @ X.T
    W2 = W2 + eta * deltaZ2 @ A1.T
    W3 = W3 + eta * deltaO @ A2.T
    b1 = b1 + eta * deltaZ1.sum(axis=1, keepdims=True)
    b2 = b2 + eta * deltaZ2.sum(axis=1, keepdims=True)
    b3 = b3 + eta * deltaO.sum(axis=1, keepdims=True)
    return W1, W2, W3, b1, b2, b3

# Right now the function doesnt do anything, but would want it to choose the type of nonlinear function to be used 
def multilayer(beta: np.float32, eta: np.float32, max_epocs: int, function: str, rnd: np.random.Generator):
    X = X_train
    Y = Y_train
    T = one_hot(Y)
    W1, W2, W3, b1, b2, b3 = init_params(rnd)
    # Doing the batch full type (i think)
    for _ in range(max_epocs):
        Z1, A1, Z2, A2, Z3, O = forward_propagation(beta, X, W1, W2, W3, b1, b2, b3)
        deltaO, deltaZ1, deltaZ2 = backward_propagation(beta, T, Z1, Z2, Z3, W1, W2, W3, A1, A2, O)
        W1, W2, W3, b1, b2, b3 = update_weights(eta, X, A1, A2, W1, W2, W3, b1, b2, b3, deltaO, deltaZ2, deltaZ1)
        
        E = mse(O, Y)
        if E < 0.01: 
            break
    
    return W1, W2, W3, b1, b2, b3
    
def test_function(beta, X_test, Y_test, W1, W2, W3, b1, b2, b3): 
    Z1 = W1.dot(X_test) + b1
    A1 = theta(Z1, beta)
    Z2 = W2.dot(A1) + b2
    A2 = theta(Z2, beta)
    Z3 = W3.dot(A2) + b3
    O = theta(Z3, beta)
    test_acc = accuracy(one_hot(Y_test), O)
    print("Test accuracy: {}".format(test_acc))
    
beta = 0.1
eta = 0.5
W1, W2, W3, b1, b2, b3 = multilayer(beta, eta, 50, "sigmoid", random_seed)
test_function(beta, X_test, Y_test, W1, W2, W3, b1, b2, b3)


"""
# Different methods for updating, online, mini-batch and batch:
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