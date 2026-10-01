"""Runs a configured experiment and records it under results/. Never plots."""

# TODO: save a trained model with its config, and load it back to resume training

import time

import numpy as np

from perceptrons.config import Config, OptimizerMethod
from perceptrons.data import one_hot_decode, one_hot_encode
from perceptrons.losses import MSE
from perceptrons.metrics import accuracy, confusion_matrix, recall
from perceptrons.multilayer import MLP
from perceptrons.optimizers import Adam, GradientDescent, Momentum
from perceptrons.training import History, train

# Name in the config -> optimizer class. A new optimizer is built for every run:
# Momentum and Adam keep state between steps, which must not carry over to the next run.
OPTIMIZERS = {
    OptimizerMethod.GradientDescent: GradientDescent,
    OptimizerMethod.Momentum: Momentum,
    OptimizerMethod.Adam: Adam,
}


def run_experiment(config: Config, X_train: np.ndarray, Y_train: np.ndarray, X_valid: np.ndarray, Y_valid: np.ndarray) -> dict:
    """Train one model from `config`, evaluate it on the validation split, return what to save.

    Y_train / Y_valid are digit labels; they are one-hot encoded here for training, and the
    labels are kept for the confusion matrix. The returned dict only holds plain Python
    types, so it can be written to JSON as is.
    """
    # The only source of randomness, so the seed saved in the config reproduces the run
    rng = np.random.default_rng(config.random_seed)
    Y_train_one_hot = one_hot_encode(Y_train, config.n_classes)
    Y_valid_one_hot = one_hot_encode(Y_valid, config.n_classes)

    layers = [config.n_features] + config.hidden_layers + [config.n_classes]

    model = MLP(layers, config.activation_method, config.activation_parameter, rng)
    loss = MSE()
    optimizer = OPTIMIZERS[config.optimizer_method](config.learning_rate)

    def report_progress(epoch: int, history: History) -> None:
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{config.epochs}  train loss {history.train_loss[-1]:.4f}  "
                  f"valid loss {history.valid_loss[-1]:.4f}")

    start = time.perf_counter()
    history = train(
        model, loss, optimizer, X_train, Y_train_one_hot,
        epochs=config.epochs,
        batch_size=config.batch_size,
        rng=rng,
        X_valid=X_valid,
        Y_valid=Y_valid_one_hot,
        on_epoch=report_progress,
    )
    total_seconds = time.perf_counter() - start

    Y_pred = one_hot_decode(model.forward(X_valid))  # labels, not one-hot
    cm = confusion_matrix(Y_valid, Y_pred, config.n_classes)  # compares labels with labels
    acc = accuracy(cm)
    print("{} accuracy: {:.3f}".format("Validation", acc))

    return {
        "train_loss": history.train_loss,
        "valid_loss": history.valid_loss,
        "epoch_seconds": history.epoch_seconds,
        "total_seconds": total_seconds,
        "accuracy": float(acc),
        # A class missing from the validation split (8 in digits.csv) has recall NaN,
        # which is not valid JSON; store it as null
        "recall": [None if np.isnan(r) else float(r) for r in recall(cm)],
        "confusion_matrix": cm.tolist(),
    }
