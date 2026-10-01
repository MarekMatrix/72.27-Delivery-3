"""Command-line entry point for running experiments and generating figures."""

# TODO: one command per deliverable: validation, Exercise 1, Exercise 2, Exercise 3

"""Command-line entry point: `ga-triangles --image ... --triangles ...`.

Argument parsing is boilerplate and implemented in full here. It calls
straight into engine.run_ga.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from perceptrons.config import OptimizerMethod, BatchMethod, ActivationMethod, Config
from perceptrons.experiment import run_experiment
from perceptrons.data import load_digits, split_train_and_validation, digits_path, digits_test_path

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the classifier model with perceptrones.")
    parser.add_argument("--number-of-classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--number-of-features", type=int, default=784, help="Number of features of the input vector.")

    parser.add_argument("--epochs", type=int, help="The number of epochs used as stopping criteria for training.")
    parser.add_argument("--batch-size", type=int, help="The size of the batch for configuration mini-batch.")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate for updating the weights.")
    parser.add_argument("--activation-parameter", type=float, required=True, help="Beta for changing the activation function.")
    parser.add_argument("--output-dir", default="results", help="Where to write outputs.")
    parser.add_argument("--hidden-layers", type=list[int], required=True, help="Size of each hidden layer.")
    
    parser.add_argument("--optimizer", choices=[m.value for m in OptimizerMethod], default=OptimizerMethod.GradientDescent.value, help="Type of optimizer to update the weights.")
    parser.add_argument("--batch", choices=[m.value for m in BatchMethod], default=BatchMethod.Batch.value, help="Type of method for choosing how many samples that are used for creating the gradient.")
    parser.add_argument("--activation", choices=[m.value for m in ActivationMethod], default=ActivationMethod.Sigmoid.value, help="Type of activation function used by each perceptron")
    
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    config = Config(
        n_classess=args.number_of_classes,
        n_features=args.number_of_features,
        learning_rate=args.learning_rate,
        activation_parameter=args.activation_parameter,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        optimizer_method=OptimizerMethod(args.optimizer),
        batch_method=BatchMethod(args.batch),
        activation_method=ActivationMethod(args.activation),
        random_seed=args.seed,
    )

    X, Y = load_digits(digits_path)
    X_train, X_valid, Y_train, Y_valid= split_train_and_validation(X, Y)
    result = run_experiment(config, X_train, Y_train)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.history.plot(str(output_dir / "---.png"))

    with open(output_dir / "---.json", "w") as f:
        json.dump({"config": config.__dict__, "stop_reason": result.stop_reason}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
