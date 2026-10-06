"""Multilayer perceptron with a configurable architecture, e.g. [2, 3, 2, 1]."""

import numpy as np
from perceptrons.activations import activation_functions

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
    
    def weight_gradients(self, delta: list[np.ndarray]) -> list[np.ndarray]:
        batch_size = self.A[0].shape[1]
        grad_W = []
        grad_b = []
        for i in range(len(delta)):
            grad_W.append(delta[i] @ self.A[i].T / batch_size)
            grad_b.append(delta[i].sum(axis=1, keepdims=True) / batch_size)
        grads = grad_W + grad_b
        return grads
