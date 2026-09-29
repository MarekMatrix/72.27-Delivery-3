"""Simple perceptron: a single neuron with a step, linear, or non-linear activation."""

# TODO: step variant — must learn AND (validation)
# TODO: linear variant — must fit ~50 samples of y = x (validation)
# TODO: non-linear variant — must fit ~50 samples of y = tanh(x) (validation)
# TODO: Exercise 1 compares the linear and non-linear variants on fraud_dataset.csv

import numpy as np

def simple_step_perceptron(x, y):

    p = len(y)         
    N = x.shape[1]
    BOUND=1000
    alpha = 0.1

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
