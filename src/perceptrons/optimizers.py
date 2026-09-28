"""Weight update rules, kept separate from the models so they can be swapped.

Interface shared by every optimizer:
    optimizer.step(params, grads)

- params and grads are lists of numpy arrays; grads[i] has the same shape as params[i].
- grads[i] is dE/dparams[i]: the optimizer SUBTRACTS it to go downhill.
- step() updates the arrays in params IN PLACE and returns nothing.
- Optimizer state (momentum's previous step, Adam's m, v and t) lives inside the optimizer.
"""

import numpy as np


class GradientDescent:
    def __init__(self, learning_rate: float) -> None:
        self.eta = learning_rate

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: w <- w - eta * g."""
        for p, g in zip(params, grads):
            p -= self.eta * g


class Momentum:
    def __init__(self, learning_rate: float, alpha: float = 0.9) -> None:
        self.eta = learning_rate
        self.alpha = alpha
        self.deltas = None  # one previous step per param, created on the first call

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: delta(t) = -eta * g + alpha * delta(t - 1), w <- w + delta(t)."""
        if self.deltas is None:
            self.deltas = [np.zeros_like(p, dtype=float) for p in params]
        for p, g, delta in zip(params, grads, self.deltas):
            delta *= self.alpha
            delta -= self.eta * g
            p += delta


class Adam:
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None  # running average of the gradient, one per param
        self.v = None  # running average of the squared gradient, one per param

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: w <- w - eta * m_hat / (sqrt(v_hat) + epsilon)."""
        if self.m is None:
            self.m = [np.zeros_like(p, dtype=float) for p in params]
            self.v = [np.zeros_like(p, dtype=float) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p -= self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon)
