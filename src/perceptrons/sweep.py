"""Exercise 2 hyperparameter sweeps: one field varied at a time, everything else fixed."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from perceptrons.cli import DATASET_PATHS, run_name
from perceptrons.config import ActivationMethod, Config, Dataset, OptimizerMethod
from perceptrons.data import load_digits, split_train_and_validation
from perceptrons.experiment import run_experiment


def sweep(base: Config, field: str, values: list, output_dir: Path) -> None:
    """Run `base` once per value of `field`, everything else fixed. Save every run and its weights."""
    # The data is loaded once for the whole sweep, so every run must use the same dataset
    if field == "dataset":
        raise ValueError("sweep loads the data once; run one sweep per dataset instead")

    # Loading is the slow part (~30 s), training a small net takes seconds
    X, Y = load_digits(DATASET_PATHS[base.dataset])
    X_train, X_valid, Y_train, Y_valid = split_train_and_validation(X, Y)
    output_dir.mkdir(parents=True, exist_ok=True)

    for value in values:
        # replace() returns a copy, so base stays the same for the next value
        config = replace(base, **{field: value})
        name = run_name(config)
        result = run_experiment(config, X_train, Y_train, X_valid, Y_valid,
                                model_path=output_dir / f"{name}.npz")

        # keep in sync with main() in cli.py
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"config": asdict(config), "result": result}, f, indent=2, default=str)

        print(f"{field}={value}  valid accuracy {result['accuracy']:.4f}  saved {path}")


if __name__ == "__main__":
    base = Config(
        dataset=Dataset.Digits,
        hidden_layers=[128],
        activation_parameter=1.0,
        learning_rate=0.3,
        epochs=50,
        random_seed=0,
    )

    # Done (all on validation, see notebooks/exercise2_results.ipynb):
    #   learning rate      GD, [16], bs 32: 0.01 / 0.03 / 0.1 / 0.3 / 1.0             -> 0.3
    #   architecture       GD, lr 0.3, bs 32: [16] / [64] / [128] / [64, 32]          -> [128]
    #   optimizer          [64], bs 32: GD 0.3, Momentum 0.003-0.3, Adam 0.0003-0.01  -> GD
    #   activation         tanh at lr 0.03 / 0.1 / 0.3                                -> sigmoid
    #   batch size         8 / 32 / 128                                               -> 8
    #   lr at batch size 8 0.03 / 0.1 / 0.3 / 1.0 / 3.0                               -> 1.0
    #   seeds 0, 1, 2 for the top candidates

    # Final configuration: retrained for seeds 0, 1, 2 so its weights are saved for the test set.
    # Same seeds as before, so the validation results are identical to the earlier runs.
    final = replace(base, batch_size=8, learning_rate=1.0)
    sweep(final, "random_seed", [0, 1, 2], Path("results"))
