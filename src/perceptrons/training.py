"""Training loop: epochs, shuffling, and batching, driving the model, loss, and optimizer."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import time

import numpy as np

# Assumed interfaces (not defined here):
#   model.forward(X) -> output           caches whatever backward() needs
#   model.backward(dE_dO) -> list[grads]  same order as model.params
#   model.params -> list[np.ndarray]      the arrays the optimizer updates in place
#   loss.value(Y_pred, Y_true) -> float
#   loss.gradient(Y_pred, Y_true) -> np.ndarray   dE/dO, same shape as Y_pred
#   optimizer.step(params, grads)         see optimizers.py
#
# Convention: samples are COLUMNS, X has shape (n_features, n_samples), as in data.py.


@dataclass
class History:
    """What one training run produced, handed to experiment.py for recording."""
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)


def iterate_minibatches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, Y_batch) pairs covering every sample exactly once, in a random order.

    batch_size = 1 is online, 1 < batch_size < n_samples is mini-batch,
    batch_size >= n_samples is full batch. The last batch may be smaller; it is kept
    so that every sample is seen once per epoch.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    n_samples = X.shape[1]
    order = rng.permutation(n_samples)

    for start in range(0, n_samples, batch_size):
        idx = order[start:start + batch_size]
        yield X[:, idx], Y[..., idx]


def train_epoch(
    model,
    loss,
    optimizer,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Run one pass over the data, one optimizer step per batch. Return the epoch's training loss.

    The epoch loss is the mean of the batch losses, weighted by batch size, so a small
    last batch does not count as much as a full one.
    """
    total_loss = 0.0
    n_seen = 0

    for X_batch, Y_batch in iterate_minibatches(X, Y, batch_size, rng):
        output = model.forward(X_batch)
        grads = model.weight_gradients(model.backward(loss.gradient(output, Y_batch)))
        optimizer.step(model.params, grads)

        n = X_batch.shape[1]
        total_loss += loss.value(output, Y_batch) * n
        n_seen += n

    return total_loss / n_seen


def evaluate_loss(model, loss, X: np.ndarray, Y: np.ndarray) -> float:
    """Loss of the current model on (X, Y), with no weight updates."""
    return loss.value(model.forward(X), Y)


def train(
    model,
    loss,
    optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    X_valid: np.ndarray | None = None,
    Y_valid: np.ndarray | None = None,
    target_loss: float | None = None,
    on_epoch: Callable[[int, History], None] | None = None,
) -> History:
    """Train for up to `epochs` epochs and return the per-epoch history.

    Stops early once the training loss drops below target_loss (if given).
    on_epoch is called after every epoch so experiment.py can report progress;
    this module never prints or plots.
    """
    history = History()

    for epoch in range(epochs):
        start = time.perf_counter()
        train_loss = train_epoch(model, loss, optimizer, X_train, Y_train, batch_size, rng)
        history.epoch_seconds.append(time.perf_counter() - start)
        history.train_loss.append(train_loss)

        if X_valid is not None and Y_valid is not None:
            history.valid_loss.append(evaluate_loss(model, loss, X_valid, Y_valid))

        if on_epoch is not None:
            on_epoch(epoch, history)

        if target_loss is not None and train_loss < target_loss:
            break

    return history
