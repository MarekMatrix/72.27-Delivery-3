"""Exercise 2: evaluate the final configuration ONCE on digits_test.csv ("production").

Ran after model selection is finished: nothing measured here is used to
change the configuration. It loads the models that sweep.py saved for each seed of
FINAL_CONFIG; it does not train anything.

    uv run python -m perceptrons.evaluate_test
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from perceptrons.cli import run_name
from perceptrons.config import Config, Dataset
from perceptrons.data import digits_test_path, load_digits, one_hot_decode
from perceptrons.experiment import load_model
from perceptrons.metrics import accuracy, confusion_matrix, recall

# Chosen on the validation set only (see notebooks/ex2_final_results.ipynb)
FINAL_CONFIG = Config(
    dataset=Dataset.Digits,
    hidden_layers=[128],
    activation_parameter=1.0,
    learning_rate=1.0,
    batch_size=8,
    epochs=50,
)
SEEDS = [0, 1, 2]


def main(results_dir: Path = Path("results")) -> None:
    X_test, Y_test = load_digits(digits_test_path)

    accuracies = []
    for seed in SEEDS:
        config = replace(FINAL_CONFIG, random_seed=seed)
        name = run_name(config)
        model_path = results_dir / f"{name}.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found: train the final configuration with sweep.py first")

        model, _ = load_model(model_path)
        cm = confusion_matrix(Y_test, one_hot_decode(model.forward(X_test)), config.n_classes)
        result = {
            "accuracy": float(accuracy(cm)),
            # A class with no test samples would have recall NaN, which is not valid JSON
            "recall": [None if np.isnan(r) else float(r) for r in recall(cm)],
            "confusion_matrix": cm.tolist(),
        }
        accuracies.append(100 * result["accuracy"])

        # "test_" prefix keeps these apart from the training runs (digits_*.json)
        path = results_dir / f"test_{name}.json"
        with open(path, "w") as f:
            json.dump({"config": asdict(config), "test": result}, f, indent=2, default=str)
        print(f"seed {seed}  test accuracy {100 * result['accuracy']:.2f} %  saved {path}")

    print(f"\ntest accuracy over {len(SEEDS)} seeds: {np.mean(accuracies):.2f} ± {np.std(accuracies, ddof=1):.2f} %")


if __name__ == "__main__":
    main()
