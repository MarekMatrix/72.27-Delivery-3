"""Validation exercises for the simple perceptron: not submitted, but they catch implementation bugs."""

# TODO: step variant learns AND: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [-1, -1, -1, 1]
# TODO: linear variant fits ~50 samples of y = x
# TODO: non-linear variant fits ~50 samples of y = tanh(x)
# TODO: step variant fails on XOR, for comparison with the multilayer perceptron

import numpy as np
from perceptrons.simple_tests import (
    simple_step_perceptron, 
    simple_linear_perceptron, 
    simple_nonlinear_perceptron
)

def test_step_variant_learns_and():
    #AND: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [-1, -1, -1, 1]
    x = np.array([
        [-1,  1, 1],
        [ 1, -1, 1],
        [-1, -1, 1],
        [ 1,  1, 1]
    ])
    y = np.array([-1, -1, -1, 1])
    
    wmin, errormin, iterations = simple_step_perceptron(x, y)
    assert errormin == 0, f"Simple step perceptron failed to learn AND gate, minimum error: {errormin}"

def test_linear_variant_fits_linear_data():
    # Generate ~50 samples of a linear function (e.g., y = x)
    P = 50
    x_vals = np.linspace(-1, 1, P)
    
    # Matrix X: Column 0: x_vals, Column 1: bias term (1s) (shape: P x N)
    X = np.column_stack((x_vals, np.ones(P)))
    
    # Target values y = x
    y = x_vals  
    
    w, final_mse, epochs = simple_linear_perceptron(X, y)

    predictions = np.dot(X, w)
    mse = np.mean((predictions - y) ** 2)
    
    # Assert that the Mean Squared Error is close to zero, proving it fit the data
    assert mse < 1e-3, f"Linear perceptron failed to fit y=x, final MSE: {mse}"

def test_nonlinear_variant_fits_nonlinear_data():
    # Generate ~50 samples of a non-linear function (e.g., y = tanh(x))
    P = 50
    x_vals = np.linspace(-1, 1, P)
    
    # Matrix X: Column 0: x_vals, Column 1: bias term (1s) (shape: P x N)
    X = np.column_stack((x_vals, np.ones(P)))
    
    # Target values y = tanh(x)
    y = np.tanh(x_vals)  
    
    w, final_mse, epochs = simple_nonlinear_perceptron(X, y)
    
    predictions = np.tanh(np.dot(X, w))
    mse = np.mean((predictions - y) ** 2)
    
    # Assert that the Mean Squared Error is close to zero, proving it fit the data
    assert mse < 1e-3, f"Non-linear perceptron failed to fit y=tanh(x), final MSE: {mse}"

if __name__ == "__main__":
    test_step_variant_learns_and()
    test_linear_variant_fits_linear_data()
    test_nonlinear_variant_fits_nonlinear_data()
    print("All validation tests passed successfully!")