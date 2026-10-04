"""Simple perceptron: a single neuron with a step, linear, or non-linear activation."""

# TODO: step variant — must learn AND (validation)
# TODO: linear variant — must fit ~50 samples of y = x (validation)
# TODO: non-linear variant — must fit ~50 samples of y = tanh(x) (validation)
# TODO: Exercise 1 compares the linear and non-linear variants on fraud_dataset.csv

import numpy as np

def simple_step_perceptron(x, y, BOUND=1000, alpha = 0.1):
    """
    Simple Step Perceptron using Rosenblatt's learning rule
    with a step/sign activation function for binary classification tasks.
    """

    p = len(y)         
    N = x.shape[1]

    def ComputeError(x_data, y_data, weights, p_val):
        err_count = 0
        for idx in range(p_val):
            excitation = np.dot(x_data[idx], weights)
            activation = np.sign(excitation)
            if activation == 0: 
                activation = 1
            if activation != y_data[idx]:
                err_count += 1
        return err_count

    # Algoritmo 5: simple perceptron algorithm from the course notes

    i = 0
    w = np.zeros(N)             
    error = 1                    
    errormin = p * 2             
    wmin = np.copy(w)

    while error > 0 and i < BOUND:
        ix = int(np.ceil(np.random.rand() * p) - 1)
        
        excitation = np.dot(x[ix], w)

        activation = np.sign(excitation)
        if activation == 0:
            activation = 1
            
        delta_w = alpha * (y[ix] - activation) * x[ix]

        w = w + delta_w

        error = ComputeError(x, y, w, p)

        if error < errormin:
            errormin = error
            wmin = np.copy(w)
            
        i = i + 1

    return wmin, errormin, i

def simple_linear_perceptron(x, y, alpha=0.01, epochs=1000):
    """
    Simple Linear Perceptron using gradient descent (Delta / Widrow-Hoff rule)
    to solve linear systems (regression) by minimizing Sum of Squared Errors.
    """

    p = len(y)
    N = x.shape[1]
    w = np.zeros(N)

    for _ in range(epochs):
        for mu in range(p):
            excitation = np.dot(x[mu], w)
            activation = excitation
            
            error_delta = y[mu] - activation
            delta_w = alpha * error_delta * x[mu]
            w = w + delta_w
            
    # Compute final Mean Squared Error for tracking/validation
    predictions = np.dot(x, w)
    final_mse = np.mean((predictions - y) ** 2)

    return w, final_mse, epochs


def simple_nonlinear_perceptron(x, y, alpha=0.01, epochs=1000, beta=1.0,activation_fn=np.tanh, derivative_fn=lambda h: 1.0 - np.tanh(h)**2):
    """
    Simple Non-Linear Perceptron using gradient descent with a non-linear 
    activation function and the chain rule for regression tasks. Tanh-function 
    is set as the defaul function.
    """

    p = len(y)
    N = x.shape[1]
    w = np.zeros(N)

    for _ in range(epochs):
        for mu in range(p):
            h = np.dot(x[mu], w)

            activation = activation_fn(beta * h)
            derivative = beta * derivative_fn(h)

            error_delta = y[mu] - activation
            delta_w = alpha * error_delta * derivative * x[mu]
            w = w + delta_w
            
    predictions = activation_fn(beta * np.dot(x, w))
    final_mse = np.mean((predictions - y) ** 2)

    return w, final_mse, epochs