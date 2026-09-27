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
        # TODO: store the hyperparameters
        ...

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: w <- w - eta * g."""
        # TODO: apply the update rule to each (param, grad) pair
        raise NotImplementedError


class Momentum:
    def __init__(self, learning_rate: float, alpha: float = 0.9) -> None:
        # TODO: store the hyperparameters; no state arrays yet
        ...

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: delta(t) = -eta * g + alpha * delta(t - 1), w <- w + delta(t)."""
        # TODO: on the first call, create one zero array per param (np.zeros_like)
        # TODO: for each (param, grad, previous delta): compute the new delta, apply it, store it
        raise NotImplementedError


class Adam:
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        # TODO: store the hyperparameters; no state arrays yet; step counter t starts at 0
        ...

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Update params in place: w <- w - eta * m_hat / (sqrt(v_hat) + epsilon)."""
        # TODO: on the first call, create zero arrays m and v, one of each per param
        # TODO: increase t once per call, not once per param
        # TODO: for each (param, grad, m, v): update m and v, bias-correct them, apply the step
        raise NotImplementedError
