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

class MLP: 
    def __init__(self, layers: list[int], function: str, beta: float, rng: np.random.Generator) -> None: 
        self.theta, self.dtheta = activation_functions[function]
        self.beta = beta
        self.W = []
        self.b = []
    
        for i in range(len(layers) - 1):
            cols = layers[i]
            rows = layers[i + 1]
            limit = np.sqrt(6 / (rows + cols))
            self.W.append(rng.uniform(-limit, limit, size=(rows, cols)))
            self.b.append(np.zeros((rows, 1)))
        
        self.params = self.W + self.b
    
    def forward(self, X: np.ndarray) -> np.ndarray: 
        self.Z = []
        self.A = [X]
        
        for i in range(len(self.W)):
            self.Z.append(self.W[i] @ self.A[i] + self.b[i])
            self.A.append(self.theta(self.Z[i], self.beta))
        O = self.A[-1]
        
        return O
    
    def backward(self, dE_dO: np.ndarray) -> list[np.ndarray]: 
        delta = [None] * len(self.W)
        delta[-1] = (dE_dO) * self.dtheta(self.Z[-1], self.beta)
        for i in range(len(self.W) - 2, -1, -1):
            delta[i] = self.W[i + 1].T @ delta[i + 1] * self.dtheta(self.Z[i], self.beta)
        return delta
    
    def weight_gradients(self, delta: np.ndarray) -> list[np.ndarray]:
        grad_W = []
        grad_b = []
        for i in range(len(delta)):
            grad_W.append(delta[i] @ self.A[i].T / self.batch_size)
            grad_b.append(delta[i].sum(axis=1, keepdims=True) / self.batch_size)
        grads = grad_W + grad_b
        return grads

def one_hot_encode(Y: np.ndarray, n_classes: int) -> np.ndarray:
    # n_classes is explicit: inferring it from Y.max() + 1 gives fewer rows when the
    # highest class happens to be missing from Y (e.g. a small batch or subset)
    one_hot_Y = np.zeros((Y.size, n_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def one_hot_decode(O: np.ndarray) -> np.ndarray:
    return O.argmax(axis=0)
        