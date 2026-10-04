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


def test_one_step_of_2_2_1_matches_hand_calculation():
    # sigmoid with beta = 1: theta(h) = 1 / (1 + exp(-2h)), theta'(h) = 2 theta(h) (1 - theta(h))
    # x = (1, -1), zeta = 1, biases 0
    #   h1 = 0.5 + 0.5 = 1.0     V1 = 0.880797
    #   h2 = 0.25 - 0.75 = -0.5  V2 = 0.268941
    #   h_out = 0.5 * V1 - V2 = 0.171457,  O = 0.584898
    #   delta_out = (O - zeta) * theta'(h_out) = -0.201567
    #   dE/dW2 = delta_out * (V1, V2) = (-0.177540, -0.054210)
    #   delta1 = 0.5 * delta_out * theta'(1.0) = -0.021163
    #   delta2 = -1.0 * delta_out * theta'(-0.5) = 0.079261
    #   dE/dW1 = (delta1, delta2)^T (1, -1)
    model = MLP([2, 2, 1], "sigmoid", 1.0, np.random.default_rng(0))
    model.W[0][:] = [[0.5, -0.5], [0.25, 0.75]]
    model.W[1][:] = [[0.5, -1.0]]
    X = np.array([[1.0], [-1.0]])
    Y = np.array([[1.0]])

    O = model.forward(X)
    assert O[0, 0] == pytest.approx(0.584898, abs=1e-6)

    grad_W1, grad_W2, grad_b1, grad_b2 = model.weight_gradients(model.backward(MSE().gradient(O, Y)))
    np.testing.assert_allclose(grad_W2, [[-0.177540, -0.054210]], atol=1e-6)
    np.testing.assert_allclose(grad_W1, [[-0.021163, 0.021163], [0.079261, -0.079261]], atol=1e-6)
    np.testing.assert_allclose(grad_b2, [[-0.201567]], atol=1e-6)
    np.testing.assert_allclose(grad_b1, [[-0.021163], [0.079261]], atol=1e-6)


@pytest.mark.parametrize("layers", [[2, 2, 1], [2, 3, 2, 1]])
def test_learns_xor(layers):
    from perceptrons.optimizers import GradientDescent
    from perceptrons.training import train

    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float).T
    Y = np.array([[1, 1, -1, -1]], dtype=float)

    model = MLP(layers, "tangent", 1.0, np.random.default_rng(3))
    train(model, MSE(), GradientDescent(0.1), X, Y, epochs=5000, batch_size=4,
          rng=np.random.default_rng(0), target_loss=1e-3)

    assert np.array_equal(np.sign(model.forward(X)), Y)
