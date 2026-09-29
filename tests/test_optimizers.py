"""Optimizer tests: each update rule is checked on functions whose minimum is known, with no model involved.

Interface these tests assume (agree it with the multilayer.py owner):
    optimizer = GradientDescent(learning_rate=...)
    optimizer = Momentum(learning_rate=..., alpha=0.9)
    optimizer = Adam(learning_rate=..., beta1=0.9, beta2=0.999, epsilon=1e-8)
    optimizer.step(params, grads)

- params and grads are lists of numpy arrays; grads[i] has the same shape as params[i].
- grads[i] is dE/dparams[i]: the optimizer SUBTRACTS it to go downhill.
- step() updates the arrays in params IN PLACE and returns nothing.
- Optimizer state (momentum's previous step, Adam's m, v and t) lives inside the optimizer.

Most tests minimise f(w) = (w - 3)^2, whose gradient is 2(w - 3) and whose minimum is w = 3.
"""

import numpy as np
import pytest

from perceptrons.optimizers import Adam, GradientDescent, Momentum

TARGET = 3.0


def quadratic_grad(w: np.ndarray) -> np.ndarray:
    """Gradient of f(w) = sum((w - 3)^2)."""
    return 2 * (w - TARGET)


def run(optimizer, w0: float, steps: int) -> list[float]:
    """Minimise the 1-D quadratic from w0 and return w after every step."""
    w = np.array([w0])
    trajectory = []
    for _ in range(steps):
        optimizer.step([w], [quadratic_grad(w)])
        trajectory.append(w[0])
    return trajectory


# --------------------------------------------------------------------------- #
# Gradient descent
# --------------------------------------------------------------------------- #

def test_gd_first_two_steps_match_hand_calculation():
    # w = 0: g = -6, w = 0 - 0.1 * -6 = 0.6
    # w = 0.6: g = -4.8, w = 0.6 + 0.48 = 1.08
    trajectory = run(GradientDescent(learning_rate=0.1), w0=0.0, steps=2)
    assert trajectory == pytest.approx([0.6, 1.08])


def test_gd_converges_to_minimum():
    trajectory = run(GradientDescent(learning_rate=0.1), w0=0.0, steps=200)
    assert trajectory[-1] == pytest.approx(TARGET, abs=1e-6)


def test_gd_diverges_when_learning_rate_too_large():
    # Each step multiplies the distance to the minimum by (1 - 2 * eta) = -1.2.
    trajectory = run(GradientDescent(learning_rate=1.1), w0=0.0, steps=20)
    assert abs(trajectory[-1] - TARGET) > abs(0.0 - TARGET)


def test_gd_updates_params_in_place():
    w = np.array([0.0])
    original = w
    GradientDescent(learning_rate=0.1).step([w], [quadratic_grad(w)])
    assert w is original
    assert w[0] == pytest.approx(0.6)


def test_gd_handles_several_params_of_different_shapes():
    # Like one layer of the MLP: a weight matrix and a bias column.
    W = np.zeros((2, 3))
    b = np.zeros((2, 1))
    optimizer = GradientDescent(learning_rate=0.1)
    for _ in range(200):
        optimizer.step([W, b], [quadratic_grad(W), quadratic_grad(b)])
    np.testing.assert_allclose(W, np.full((2, 3), TARGET), atol=1e-6)
    np.testing.assert_allclose(b, np.full((2, 1), TARGET), atol=1e-6)


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #

def test_momentum_first_three_steps_match_hand_calculation():
    # delta(t) = -eta * g + alpha * delta(t - 1), w = w + delta(t)
    # step 1: g = -6,    delta = 0.6,                  w = 0.6
    # step 2: g = -4.8,  delta = 0.48 + 0.9 * 0.6 = 1.02,   w = 1.62
    # step 3: g = -2.76, delta = 0.276 + 0.9 * 1.02 = 1.194, w = 2.814
    trajectory = run(Momentum(learning_rate=0.1, alpha=0.9), w0=0.0, steps=3)
    assert trajectory == pytest.approx([0.6, 1.62, 2.814])


def test_momentum_keeps_moving_when_gradient_is_zero():
    # The remembered step carries on: 0.6, then 0.9 * 0.6 = 0.54 with no gradient at all.
    w = np.array([0.0])
    optimizer = Momentum(learning_rate=0.1, alpha=0.9)
    optimizer.step([w], [np.array([-6.0])])
    optimizer.step([w], [np.array([0.0])])
    assert w[0] == pytest.approx(0.6 + 0.54)


def test_momentum_with_alpha_zero_is_gradient_descent():
    momentum = run(Momentum(learning_rate=0.1, alpha=0.0), w0=0.0, steps=10)
    gd = run(GradientDescent(learning_rate=0.1), w0=0.0, steps=10)
    assert momentum == pytest.approx(gd)


def test_momentum_converges_to_minimum():
    trajectory = run(Momentum(learning_rate=0.1, alpha=0.9), w0=0.0, steps=500)
    assert trajectory[-1] == pytest.approx(TARGET, abs=1e-6)


def test_momentum_keeps_separate_state_per_param():
    # Only the first param ever gets a gradient; the second must never move.
    a = np.array([0.0])
    b = np.array([0.0])
    optimizer = Momentum(learning_rate=0.1, alpha=0.9)
    for _ in range(3):
        optimizer.step([a, b], [np.array([-6.0]), np.array([0.0])])
    assert a[0] != 0.0
    assert b[0] == 0.0


def test_momentum_beats_gd_on_a_narrow_valley():
    # f(x, y) = x^2 + 25 y^2: steep in y, flat in x. The steep direction forces a small
    # learning rate, so plain gradient descent crawls along x. This is why momentum exists.
    def valley_grad(p):
        return p * np.array([2.0, 50.0])

    def steps_to_converge(optimizer, max_steps=10_000):
        p = np.array([10.0, 1.0])
        for step in range(1, max_steps + 1):
            optimizer.step([p], [valley_grad(p)])
            if np.linalg.norm(p) < 1e-3:
                return step
        return max_steps

    gd_steps = steps_to_converge(GradientDescent(learning_rate=0.03))
    momentum_steps = steps_to_converge(Momentum(learning_rate=0.03, alpha=0.9))
    assert momentum_steps < gd_steps


# --------------------------------------------------------------------------- #
# Adam
# --------------------------------------------------------------------------- #

def test_adam_first_step_has_size_learning_rate():
    # m_hat = g and v_hat = g^2 after bias correction, so the step is eta * g / |g| = eta.
    trajectory = run(Adam(learning_rate=0.1), w0=0.0, steps=1)
    assert trajectory[0] == pytest.approx(0.1)


@pytest.mark.parametrize("gradient", [1e-3, 1.0, 1e3])
def test_adam_first_step_does_not_depend_on_gradient_scale(gradient):
    # Without bias correction the first step would be far from eta (about 3.2 * eta).
    w = np.array([0.0])
    Adam(learning_rate=0.1).step([w], [np.array([gradient])])
    assert w[0] == pytest.approx(-0.1, rel=1e-4)


def test_adam_second_step_matches_hand_calculation():
    # step 2, from w = 0.1: g = -5.8
    #   m = 0.9 * -0.6 + 0.1 * -5.8          = -1.12
    #   v = 0.999 * 0.036 + 0.001 * 33.64    = 0.069604
    #   m_hat = -1.12 / (1 - 0.9^2)          = -5.894737
    #   v_hat = 0.069604 / (1 - 0.999^2)     = 34.819410
    #   w = 0.1 + 0.1 * 5.894737 / sqrt(34.819410) = 0.199897
    trajectory = run(Adam(learning_rate=0.1), w0=0.0, steps=2)
    assert trajectory == pytest.approx([0.1, 0.199897], abs=1e-6)


def test_adam_counts_steps_not_params():
    # t must go up once per step() call. If it went up once per param, the second param's
    # first update would use t = 2 in the bias correction and would not have size eta.
    a = np.array([0.0])
    b = np.array([0.0])
    Adam(learning_rate=0.1).step([a, b], [np.array([5.0]), np.array([5.0])])
    assert a[0] == pytest.approx(-0.1, rel=1e-4)
    assert b[0] == pytest.approx(-0.1, rel=1e-4)


def test_adam_converges_to_minimum():
    trajectory = run(Adam(learning_rate=0.1), w0=0.0, steps=1000)
    assert trajectory[-1] == pytest.approx(TARGET, abs=1e-3)
