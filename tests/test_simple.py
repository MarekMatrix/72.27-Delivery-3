"""Validation exercises for the simple perceptron: not submitted, but they catch implementation bugs."""

import numpy as np

# TODO: step variant learns AND: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [-1, -1, -1, 1]
# TODO: linear variant fits ~50 samples of y = x
# TODO: non-linear variant fits ~50 samples of y = tanh(x)
# TODO: step variant fails on XOR, for comparison with the multilayer perceptron

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

    # --- Algoritmo 5: Simple Perceptron Algorithm ---

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

    print(f"Training finished in {i} iterations.")
    print(f"Minimum Error: {errormin}")
    print(f"Final Weights (wmin): {wmin}")


x = np.array([
    [-1,  1, 1],
    [ 1, -1, 1],
    [-1, -1, 1],
    [ 1,  1, 1]
])

y = np.array([-1, -1, -1, 1])

def main(): 
    simple_step_perceptron(x, y)
main()