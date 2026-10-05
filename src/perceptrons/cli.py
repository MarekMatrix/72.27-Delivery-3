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
from perceptrons.data import (
    digits_path,
    more_digits_path,
    load_digits,
    split_train_and_validation,
    split_train_and_validation_stratified,
    oversample_training,
    add_unique_training_samples,
    augment_training,
)
from perceptrons.experiment import run_experiment


DATASET_PATHS = {
    Dataset.Digits: digits_path,
    Dataset.MoreDigits: more_digits_path,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the classifier model with perceptrons."
    )
    parser.add_argument(
        "--number-of-classes",
        type=int,
        default=10,
        help="Number of classes.",
    )
    parser.add_argument(
        "--number-of-features",
        type=int,
        default=784,
        help="Number of features of the input vector.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="The number of epochs used as stopping criteria for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Samples per weight update: 1 = online, >= number of samples = full batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Learning rate for updating the weights.",
    )
    parser.add_argument(
        "--activation-parameter",
        type=float,
        required=True,
        help="Beta for changing the activation function.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Where to write outputs.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        required=True,
        help="Size of each hidden layer, e.g. --hidden-layers 64 32.",
    )
    parser.add_argument(
        "--optimizer",
        choices=[m.value for m in OptimizerMethod],
        default=OptimizerMethod.GradientDescent.value,
        help="Type of optimizer to update the weights.",
    )
    parser.add_argument(
        "--activation",
        choices=[m.value for m in ActivationMethod],
        default=ActivationMethod.Sigmoid.value,
        help="Type of activation function used by each perceptron.",
    )
    parser.add_argument(
        "--dataset",
        default=Dataset.Digits.value,
        help="Dataset name (digits or more_digits) or path to either CSV file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["sequential", "stratified"],
        default="sequential",
    )
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--extra-dataset", type=Path, default=None)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Add randomly shifted copies of training images.",
    )
    return parser


def run_name(config: Config) -> str:
    """A descriptive filename that includes the dataset and configuration."""
    hidden = "-".join(str(h) for h in config.hidden_layers)
    return (
        f"{config.dataset.value}_{config.optimizer_method.value}"
        f"_lr{config.learning_rate}_h{hidden}"
        f"_bs{config.batch_size}_{config.activation_method.value}"
        f"_seed{config.random_seed}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Support dataset names and the CSV paths used in earlier commands.
    dataset_name = Path(args.dataset).stem
    try:
        dataset = Dataset(dataset_name)
    except ValueError:
        parser.error(
            "--dataset must be digits, more_digits, "
            "or a path to digits.csv or more_digits.csv."
        )

    dataset_path = (
        DATASET_PATHS[dataset]
        if args.dataset in [item.value for item in Dataset]
        else Path(args.dataset)
    )

    config = Config(
        n_classes=args.number_of_classes,
        n_features=args.number_of_features,
        dataset=dataset,
        learning_rate=args.learning_rate,
        activation_parameter=args.activation_parameter,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        optimizer_method=OptimizerMethod(args.optimizer),
        activation_method=ActivationMethod(args.activation),
        random_seed=args.seed,
    )

    config.extra["dataset"] = str(dataset_path)
    config.extra["split"] = args.split
    config.extra["oversample"] = args.oversample
    config.extra["augment"] = args.augment
    config.extra["extra_dataset"] = (
        str(args.extra_dataset)
        if args.extra_dataset is not None
        else None
    )

    X, Y = load_digits(dataset_path)

    if args.split == "stratified":
        X_train, X_valid, Y_train, Y_valid = (
            split_train_and_validation_stratified(
                X, Y, seed=args.seed
            )
        )
    else:
        X_train, X_valid, Y_train, Y_valid = (
            split_train_and_validation(X, Y)
        )

    if args.extra_dataset is not None:
        X_train, Y_train = add_unique_training_samples(
            X_train, Y_train, X_valid, args.extra_dataset
        )
        print(f"Training samples after adding data: {Y_train.size}")

    if args.oversample:
        X_train, Y_train = oversample_training(
            X_train, Y_train, seed=args.seed
        )

    output_dir = Path(args.output_dir)
    if args.augment:
        X_train, Y_train = augment_training(
            X_train, Y_train, seed=args.seed
        )
        print(f"Training samples after augmentation: {Y_train.size}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        output_dir / f"{run_name(config)}.npz"
        if args.save_model
        else None
    )

    result = run_experiment(
        config,
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        model_path=model_path,
    )

    path = output_dir / f"{run_name(config)}.json"

    # Convert enums and NumPy scalars into JSON-friendly values.
    with open(path, "w") as f:
        json.dump(
            {"config": asdict(config), "result": result},
            f,
            indent=2,
            default=str,
        )

    print(f"saved {path}")


if __name__ == "__main__":
    main()