"""Exercise 2: the variant studies on digits.csv, and the final model on digits_test.csv.

    uv run python -m perceptrons.ex2 run       # trains everything, writes results/ex2/
    uv run python -m perceptrons.ex2 figures   # reads results/ex2/, writes the figures
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from perceptrons.cli import run_name
from perceptrons.config import ActivationMethod, Config, OptimizerMethod
from perceptrons.data import digits_path, digits_test_path, load_digits, split_train_and_validation
from perceptrons.experiment import load_run, run_experiment, save_run

RESULTS = Path(__file__).parent.parent.parent / "results" / "ex2"
SEEDS = [0, 1, 2]

BASELINE = Config(
    optimizer_method=OptimizerMethod.GradientDescent,
    activation_method=ActivationMethod.Sigmoid,
    activation_parameter=1.0,
    batch_size=32,
    hidden_layers=[64],
    learning_rate=0.5,
    epochs=60,
)


def studies() -> dict[str, list[Config]]:
    # one factor at a time, everything else stays at BASELINE
    return {
        "learning_rate": [replace(BASELINE, learning_rate=lr) for lr in [0.01, 0.1, 0.5, 1.0, 3.0]],
        "architecture": [
            replace(BASELINE, hidden_layers=h)
            for h in [[16], [32], [64], [128], [256], [64, 32], [128, 64]]
        ],
        # each optimizer gets a learning rate that suits it, plain GD's 0.5 makes Adam blow up
        "optimizer": [
            replace(BASELINE, optimizer_method=OptimizerMethod.GradientDescent, learning_rate=0.5),
            replace(BASELINE, optimizer_method=OptimizerMethod.Momentum, learning_rate=0.05),
            replace(BASELINE, optimizer_method=OptimizerMethod.Adam, learning_rate=0.001),
        ],
        "batch_size": [replace(BASELINE, batch_size=bs) for bs in [1, 8, 32, 128, 512]],
    }


def load_split():
    X, Y = load_digits(digits_path)
    return split_train_and_validation(X, Y)


def train_one(job: tuple[Config, Path]) -> str:
    config, path = job
    if path.exists():
        return f"skip {path.name}"
    X_train, X_valid, Y_train, Y_valid = load_split()
    result = run_experiment(config, X_train, Y_train, X_valid, Y_valid, verbose=False)
    save_run(config, result, path)
    return f"{path.parent.name}/{path.name}  valid acc {result['accuracy']:.4f}"


def read_study(study: str) -> list[tuple[Config, dict]]:
    return [load_run(p) for p in sorted((RESULTS / study).glob("*.json"))]


def mean_accuracy(runs: list[tuple[Config, dict]]) -> dict[str, tuple[Config, list[float]]]:
    """Group the seeds of each configuration together, keyed by the run name without the seed."""
    groups = {}
    for config, result in runs:
        key = run_name(config).rsplit("_seed", 1)[0]
        groups.setdefault(key, (config, []))[1].append(result["accuracy"])
    return groups


def pick_best() -> Config:
    best, best_acc = None, -1.0
    for study in studies():
        for config, accs in mean_accuracy(read_study(study)).values():
            if np.mean(accs) > best_acc:
                best, best_acc = config, float(np.mean(accs))
    print(f"best config: {run_name(best)}  mean valid acc {best_acc:.4f}")
    return best


def run(workers: int) -> None:
    # numpy would otherwise start a thread pool in every worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = []
    for study, configs in studies().items():
        for config in configs:
            for seed in SEEDS:
                c = replace(config, random_seed=seed)
                jobs.append((c, RESULTS / study / f"{run_name(c)}.json"))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(train_one, jobs):
            print(line)

    # Only now does digits_test.csv get used, once, on the configuration chosen on validation
    best = replace(pick_best(), random_seed=0)
    X_train, X_valid, Y_train, Y_valid = load_split()
    X_test, Y_test = load_digits(digits_test_path)
    result = run_experiment(best, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    save_run(best, result, RESULTS / "best.json")


def figures() -> None:
    from perceptrons import plots

    out = RESULTS / "figures"
    for study in studies():
        runs = read_study(study)
        plots.plot_study_accuracy(mean_accuracy(runs), study, out / f"{study}_accuracy.png")
        plots.plot_study_curves(runs, study, out / f"{study}_valid_loss.png")
    plots.plot_epoch_time(read_study("batch_size"), "batch_size", out / "batch_size_time.png")
    plots.plot_epoch_time(read_study("architecture"), "architecture", out / "architecture_time.png")

    best, result = load_run(RESULTS / "best.json")
    plots.plot_error_convergence(result["train_loss"], result["valid_loss"], out / "best_convergence.png",
                                 title=f"Best model: {run_name(best)}")
    plots.plot_confusion_matrix(np.array(result["confusion_matrix"]), out / "best_valid_confusion.png",
                                title="Validation split (digits.csv)")
    plots.plot_confusion_matrix(np.array(result["test"]["confusion_matrix"]), out / "best_test_confusion.png",
                                title="Production stand-in (digits_test.csv)")
    plots.plot_recall_valid_vs_test(result["recall"], result["test"]["recall"], out / "best_recall.png")

    X, Y = load_digits(digits_path)
    X_test, Y_test = load_digits(digits_test_path)
    plots.plot_class_distribution(Y, Y_test, out / "class_distribution.png")
    first = [int(np.flatnonzero(Y_test == d)[0]) for d in range(10)]
    plots.plot_digit_grid(X_test[:, first], [f"label {d}" for d in range(10)], out / "digit_samples.png",
                          "One sample of each digit (28 x 28 pixels, values in [0, 1])", cols=5)

    predictions = np.array(result["test"]["predictions"])
    eights = np.flatnonzero(Y_test == 8)[:16]
    plots.plot_digit_grid(X_test[:, eights], [f"predicted {p}" for p in predictions[eights]],
                          out / "misclassified_eights.png", "Test samples labelled 8")

    summary = {
        "config": run_name(best),
        "valid_accuracy": result["accuracy"],
        "test_accuracy": result["test"]["accuracy"],
        "test_recall": result["test"]["recall"],
    }
    print(json.dumps(summary, indent=2))
    print(f"figures in {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exercise 2 studies.")
    parser.add_argument("command", choices=["run", "figures"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.workers)
    else:
        figures()


if __name__ == "__main__":
    main()
