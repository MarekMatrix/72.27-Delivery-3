"""Training loop: epochs, shuffling, and batching, driving the model, loss, and optimizer."""

# TODO: batch size as a hyperparameter, covering online, mini-batch, and full-batch updates
# TODO: hand per-epoch loss and timing to experiment.py for recording

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
    # TODO: anything else you want per epoch (e.g. accuracy), or a stop reason


def iterate_minibatches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, Y_batch) pairs covering every sample exactly once, in a random order.

    batch_size = 1 is online, 1 < batch_size < n_samples is mini-batch,
    batch_size >= n_samples is full batch.
    """
    # TODO: validate batch_size (what should 0 or a negative value do?)
    # TODO: draw a random permutation of the sample indices using rng
    # TODO: walk through the permutation in steps of batch_size
    # TODO: slice X and Y along the SAMPLE axis with those indices and yield them
    # TODO: decide what happens to the last, smaller batch (keep it or drop it) and be able to justify it


def train_epoch(
    model,
    loss,
    optimizer,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Run one pass over the data, one optimizer step per batch. Return the epoch's training loss."""
    # TODO: for each batch from iterate_minibatches:
    #   TODO: forward pass
    #   TODO: loss value for this batch (for reporting)
    #   TODO: dE/dO from the loss
    #   TODO: backward pass -> grads
    #   TODO: optimizer.step(model.params, grads)
    # TODO: combine the per-batch losses into one epoch number (plain mean, or weighted by batch size?)


def evaluate_loss(model, loss, X: np.ndarray, Y: np.ndarray) -> float:
    """Loss of the current model on (X, Y), with no weight updates."""
    # TODO: forward pass on the whole set
    # TODO: return the loss value


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

    on_epoch is called after every epoch so experiment.py can report progress;
    this module never prints or plots.
    """
    # TODO: create an empty History
    # TODO: for each epoch:
    #   TODO: start a timer
    #   TODO: train_epoch -> training loss
    #   TODO: stop the timer; record loss and time
    #   TODO: if validation data was given, record the validation loss
    #   TODO: call on_epoch if given
    #   TODO: stop early if target_loss is set and reached (on which loss: train or valid?)
    # TODO: return the history