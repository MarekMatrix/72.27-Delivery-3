"""Command-line entry point for running experiments and generating figures.

    uv run python -m perceptrons.cli --learning-rate 0.5 --activation-parameter 1 \
        --hidden-layers 64 --epochs 50 --batch-size 32
"""

# TODO: one command per deliverable: validation, Exercise 1, Exercise 2, Exercise 3

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from perceptrons.config import ActivationMethod, Config, Dataset, OptimizerMethod
from perceptrons.data import digits_path, load_digits, more_digits_path, split_train_and_validation
from perceptrons.experiment import run_experiment



DATASET_PATHS = {
    Dataset.Digits: digits_path,
    Dataset.MoreDigits: more_digits_path,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the classifier model with perceptrons.")
    parser.add_argument("--number-of-classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--number-of-features", type=int, default=784, help="Number of features of the input vector.")

    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs used as stopping criteria for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per weight update: 1 = online, >= number of samples = full batch.")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate for updating the weights.")
    parser.add_argument("--activation-parameter", type=float, required=True, help="Beta for changing the activation function.")
    parser.add_argument("--output-dir", default="results", help="Where to write outputs.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", required=True, help="Size of each hidden layer, e.g. --hidden-layers 64 32.")

    parser.add_argument("--optimizer", choices=[m.value for m in OptimizerMethod], default=OptimizerMethod.GradientDescent.value, help="Type of optimizer to update the weights.")
    parser.add_argument("--activation", choices=[m.value for m in ActivationMethod], default=ActivationMethod.Sigmoid.value, help="Type of activation function used by each perceptron")
    parser.add_argument("--dataset", choices=[m.value for m in Dataset], default=Dataset.Digits.value, help="Dataset to use for training.")

    parser.add_argument("--seed", type=int, default=0)
    return parser


def run_name(config: Config) -> str:
    """A file name that says which run it is, e.g. digits_gradient_descent_lr0.5_h64_bs32_seed0."""
    hidden = "-".join(str(h) for h in config.hidden_layers)
    return (
        f"{config.dataset.value}_{config.optimizer_method.value}_lr{config.learning_rate}_h{hidden}"
        f"_bs{config.batch_size}_{config.activation_method.value}_seed{config.random_seed}"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    config = Config(
        n_classes=args.number_of_classes,
        n_features=args.number_of_features,
        dataset=Dataset(args.dataset),
        learning_rate=args.learning_rate,
        activation_parameter=args.activation_parameter,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        optimizer_method=OptimizerMethod(args.optimizer),
        activation_method=ActivationMethod(args.activation),
        random_seed=args.seed,
    )

    X, Y = load_digits(DATASET_PATHS[config.dataset])
    X_train, X_valid, Y_train, Y_valid = split_train_and_validation(X, Y)
    result = run_experiment(config, X_train, Y_train, X_valid, Y_valid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name(config)}.json"

    # default=str turns the enums and any numpy scalars into JSON-friendly values
    with open(path, "w") as f:
        json.dump({"config": asdict(config), "result": result}, f, indent=2, default=str)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
