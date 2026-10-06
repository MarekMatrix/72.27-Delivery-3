"""Loss functions and their gradients with respect to the network output."""

import numpy as np


class MSE:
    """Squared error, averaged over samples: E = 1/(2n) * sum((O - Y)^2), n = number of samples.

    Summed over the output neurons, averaged over the samples, so the value does not
    depend on the batch size and train/valid losses of different sizes are comparable.

    Convention: gradient() returns the per-sample derivative O - Y, WITHOUT the 1/n.
    The model's gradient step divides by the batch size exactly once, so the
    optimizer ends up with dE/dW of the E returned by value().
    """

    def value(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        n_samples = Y_pred.shape[1]
        return float(((Y_pred - Y_true) ** 2).sum() / (2 * n_samples))

    def gradient(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return Y_pred - Y_true
