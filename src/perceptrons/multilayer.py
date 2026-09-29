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

hidden_layers = [16, 16]

# Testing plotting of numbers
#plt.imshow(image, cmap="gray")
#plt.show()

input_size = 784 # Size x[0]
output_size = 10 # Size y.max

def init_params(layers: list, rnd: np.random.Generator): # what to write for output?
    # TRY DIFFERENT TYPES OF INITIALIZATION: (random, Xavier...)
    W = []
    b = []
    
    for i in range(len(layers) - 1):
        cols = layers[i]
        rows = layers[i + 1]
        W.append(rnd.uniform(-0.05, 0.05, size=(rows, cols)))
        b.append(np.zeros((rows, 1)))
    
    return W, b

# Theta functions: 
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

def one_hot(Y: np.arange) -> np.array:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.array) -> np.array:
    return O.argmax(axis=0)

def forward_propagation(beta: np.float32, function: str, layers: list, X: np.array, W: np.array, b: np.array) -> np.array:
    theta, _ = activation_functions[function]
    Z = []
    A = [X]
    
    for i in range(len(W)):
        Z.append(W[i] @ A[i] + b[i])
        A.append(theta(Z[i], beta))
    O = A[-1]
    
    return Z, A, O

def backward_propagation(beta: np.float32, function: str, Y: np.array, W: np.array, Z: np.array, O: np.array) -> np.array:
    _, dtheta = activation_functions[function]
    delta = [None] * len(W)
    delta[-1] = (Y - O) * dtheta(Z[-1], beta)
    for i in range(len(W) - 2, -1, -1):
        delta[i] = W[i + 1].T @ delta[i + 1] * dtheta(Z[i], beta)
    return delta

def update_weights(eta: np.float32, batch_size: int, W: np.array, b: np.array, A: np.array, delta: np.array) -> np.array: 
    for i in range(len(W)):
        W[i] = W[i] + eta * delta[i] @ A[i].T / batch_size
        b[i] = b[i] + eta * delta[i].sum(axis=1, keepdims=True) / batch_size
    return W, b

# Right now the function doesnt do anything, but would want it to choose the type of nonlinear function to be used 
def multilayer(eta: np.float32, beta: np.float32, function: str, hidden_layers: list, max_epocs: int, rnd: np.random.Generator):
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
        
        E = mse(O, Y)
        if E < 0.01: 
            break
    
    return W, b



from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def confusion(y_true: np.array, y_pred: np.array):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    cm_normalized = np.round(cm/np.sum(cm, axis=1).reshape(-1,1), 2)
    sns.heatmap(cm_normalized, cmap="Blues", annot=True, xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    

def test_function(beta, function, layers, X_valid, Y_valid, W, b): 
    _, _, O = forward_propagation(beta, function, layers, X_valid, W, b)
    test_acc = accuracy(one_hot(Y_valid), O)
    print("Test accuracy: {}".format(test_acc))
    confusion(Y_valid, one_hot_decode(O))
    
beta = 1
eta = 0.05
W, b = multilayer(eta, beta, "sigmoid", [16, 16], 100, random_seed)
test_function(beta, "sigmoid", [16, 16] , X_test, Y_test, W, b)


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
