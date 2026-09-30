"""Backpropagation in the MLP class must match a numerical derivative of the loss."""

import numpy as np
import pytest

from perceptrons.losses import MSE
from perceptrons.multilayer import MLP

LAYERS = [4, 3, 2]
BETA = 1.0


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(LAYERS[0], 5))
    Y = rng.uniform(size=(LAYERS[-1], 5))
    return X, Y


def test_gradients_match_numerical_derivative_of_loss(data):
    """The whole chain loss.gradient -> backward -> weight_gradients is dE/dparams of loss.value."""
    X, Y = data
    model = MLP(LAYERS, "sigmoid", BETA, np.random.default_rng(1))
    loss = MSE()
    grads = model.weight_gradients(model.backward(loss.gradient(model.forward(X), Y)))

    eps = 1e-6
    for p, g in zip(model.params, grads):
        for idx in np.ndindex(p.shape):
            original = p[idx]
            p[idx] = original + eps
            e_plus = loss.value(model.forward(X), Y)
            p[idx] = original - eps
            e_minus = loss.value(model.forward(X), Y)
            p[idx] = original
            assert np.isclose(g[idx], (e_plus - e_minus) / (2 * eps), atol=1e-7)
