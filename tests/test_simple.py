"""Validation exercises for the simple perceptron: not submitted, but they catch implementation bugs."""

# TODO: step variant learns AND: x = [[-1, 1], [1, -1], [-1, -1], [1, 1]], y = [-1, -1, -1, 1]
# TODO: linear variant fits ~50 samples of y = x
# TODO: non-linear variant fits ~50 samples of y = tanh(x)
# TODO: step variant fails on XOR, for comparison with the multilayer perceptron

import numpy as np
from src.perceptrons.simple import simple_step_perceptron

def test_step_variant_learns_and():
    x = np.array([
        [-1,  1, 1],
        [ 1, -1, 1],
        [-1, -1, 1],
        [ 1,  1, 1]
    ])
    y = np.array([-1, -1, -1, 1])
    
    wmin, errormin, iterations = simple_step_perceptron(x, y)
    assert errormin == 0, f"Simple step perceptron failed to learn AND gate, minimum error: {errormin}"

test_step_variant_learns_and()