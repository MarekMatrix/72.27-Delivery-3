"""The MLP class must reproduce the function-based implementation in multilayer.py."""

import numpy as np
import pytest

from perceptrons.losses import MSE
from perceptrons.multilayer import backward_propagation, forward_propagation, gradient, init_params
from perceptrons.multilayer_class import MLP

LAYERS = [4, 3, 2]
BETA = 1.0


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(LAYERS[0], 5))
    Y = rng.uniform(size=(LAYERS[-1], 5))
    return X, Y


@pytest.mark.parametrize("function", ["sigmoid", "tangent"])
def test_class_matches_functions(data, function):
    X, Y = data
    W, b = init_params(LAYERS, np.random.default_rng(42))
    model = MLP(LAYERS, function, BETA, np.random.default_rng(42))

    Z, A, O = forward_propagation(BETA, function, X, W, b)
    O_class = model.forward(X)
    assert np.allclose(O_class, O)

    grads = gradient(X.shape[1], A, backward_propagation(BETA, function, Y, W, Z, O))
    grads_class = model.weight_gradients(model.backward(MSE().gradient(O_class, Y)))
    assert len(grads_class) == len(model.params) == 2 * (len(LAYERS) - 1)
    for g_class, g, p in zip(grads_class, grads, model.params):
        assert g_class.shape == p.shape
        assert np.allclose(g_class, g)


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
