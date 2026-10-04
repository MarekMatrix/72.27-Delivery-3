"""Runs a configured experiment and records it under results/. Never plots."""

# TODO: save a trained model with its config, and load it back to resume training

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from perceptrons.config import ActivationMethod, Config, OptimizerMethod
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


def evaluate(model: MLP, X: np.ndarray, Y: np.ndarray, n_classes: int) -> dict:
    Y_pred = one_hot_decode(model.forward(X))
    cm = confusion_matrix(Y, Y_pred, n_classes)
    return {
        "accuracy": float(accuracy(cm)),
        # A class missing from the split (8 in digits.csv) has recall NaN, store it as null
        "recall": [None if np.isnan(r) else float(r) for r in recall(cm)],
        "confusion_matrix": cm.tolist(),
    }


def run_experiment(
    config: Config,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Train one model from `config`, evaluate it on the validation split, return what to save.

    Y_train / Y_valid are digit labels; they are one-hot encoded here for training, and the
    labels are kept for the confusion matrix. The returned dict only holds plain Python
    types, so it can be written to JSON as is.

    The test set is optional and only meant for the final model, after model selection.
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
        if verbose and (epoch + 1) % 10 == 0:
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

    valid = evaluate(model, X_valid, Y_valid, config.n_classes)
    if verbose:
        print(f"validation accuracy: {valid['accuracy']:.3f}")

    result = {
        "train_loss": history.train_loss,
        "valid_loss": history.valid_loss,
        "epoch_seconds": history.epoch_seconds,
        "total_seconds": total_seconds,
        **valid,
    }
    if X_test is not None and Y_test is not None:
        result["test"] = evaluate(model, X_test, Y_test, config.n_classes)
        result["test"]["predictions"] = one_hot_decode(model.forward(X_test)).tolist()
        if verbose:
            print(f"test accuracy: {result['test']['accuracy']:.3f}")
    return result


def save_run(config: Config, result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # default=str turns the enums into their values
    with open(path, "w") as f:
        json.dump({"config": asdict(config), "result": result}, f, indent=2, default=str)


def load_run(path: Path) -> tuple[Config, dict]:
    with open(path) as f:
        record = json.load(f)
    config = Config(**record["config"])
    config.optimizer_method = OptimizerMethod(config.optimizer_method)
    config.activation_method = ActivationMethod(config.activation_method)
    return config, record["result"]
