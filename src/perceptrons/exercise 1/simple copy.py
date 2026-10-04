"""
TP3 - Exercise 1: simple linear vs. simple non-linear (sigmoid) perceptron
Knowledge distillation: learn BigModel's fraud probability from the features.

How to run (macOS, VS Code):
  1. Put this file in the same folder as fraud_dataset.csv
  2. In the terminal: pip3 install numpy pandas matplotlib
  3. Run: python3 ex1_perceptron.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "fraud_dataset.csv"
TARGET = "big_model_fraud_probability"
GROUND_TRUTH = "flagged_fraud"          # never used for training
DROP_COLUMNS = []                       # e.g. ["timestamp", "device_screen_resolution"]

EPOCHS = 2000
ALPHA_LINEAR = 0.1
ALPHA_SIGMOID = 0.5
SEED = 42


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))


def sigmoid_prime(h):
    s = sigmoid(h)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def standardize(x, mean=None, std=None):
    """Scale every column to mean 0 and std 1. Pass mean/std to reuse a fit."""
    if mean is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)   # avoid division by zero
    return (x - mean) / std, mean, std


def add_bias(x):
    """Append a column of ones (after scaling) so w[0] acts as the bias."""
    return np.hstack([np.ones((x.shape[0], 1)), x])


def load_fraud_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns
                    if c not in [TARGET, GROUND_TRUTH] + DROP_COLUMNS]
    x_raw = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    flagged = df[GROUND_TRUTH].to_numpy(dtype=int)
    return x_raw, y, flagged, feature_cols


# ---------------------------------------------------------------------------
# Perceptrons
# ---------------------------------------------------------------------------
def simple_step_perceptron(x, y, bound=1000, alpha=0.1, seed=SEED):
    """Rosenblatt rule with sign activation (for the AND validation)."""
    rng = np.random.default_rng(seed)
    p, n = x.shape
    w = np.zeros(n)
    w_min, error_min = w.copy(), p * 2
    i, error = 0, 1

    def compute_error(weights):
        predictions = np.where(x @ weights >= 0, 1, -1)
        return int(np.sum(predictions != y))

    while error > 0 and i < bound:
        ix = rng.integers(p)
        activation = 1 if x[ix] @ w >= 0 else -1
        w = w + alpha * (y[ix] - activation) * x[ix]
        error = compute_error(w)
        if error < error_min:
            error_min, w_min = error, w.copy()
        i += 1
    return w_min, error_min, i


def simple_linear_perceptron(x, y, alpha=ALPHA_LINEAR, epochs=EPOCHS):
    """Identity activation, batch gradient descent on the MSE."""
    p, n = x.shape
    w = np.zeros(n)
    history = []
    for _ in range(epochs):
        a = x @ w
        history.append(np.mean((a - y) ** 2))
        w += alpha * (x.T @ (y - a)) / p
    return w, history


def simple_nonlinear_perceptron(x, y, alpha=ALPHA_SIGMOID, epochs=EPOCHS,
                                beta=1.0, activation_fn=sigmoid,
                                derivative_fn=sigmoid_prime):
    """Sigmoid activation by default, batch gradient descent on the MSE."""
    p, n = x.shape
    w = np.zeros(n)
    history = []
    for _ in range(epochs):
        h = beta * (x @ w)
        a = activation_fn(h)
        history.append(np.mean((a - y) ** 2))
        delta = (y - a) * beta * derivative_fn(h)   # derivative at beta*h
        w += alpha * (x.T @ delta) / p
    return w, history


# ---------------------------------------------------------------------------
# Validation (quick sanity checks, not part of the deliverable)
# ---------------------------------------------------------------------------
def run_validation():
    print("=== Validation ===")
    # AND with the step perceptron
    x_and = add_bias(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float))
    y_and = np.array([-1, -1, -1, 1])
    w, err, it = simple_step_perceptron(x_and, y_and)
    print(f"AND  step perceptron   -> errors: {err}, iterations: {it}")

    # y = x with the linear perceptron
    xs = np.linspace(-2, 2, 50).reshape(-1, 1)
    w, hist = simple_linear_perceptron(add_bias(xs), xs.ravel(), epochs=500)
    print(f"y=x  linear perceptron -> MSE: {hist[-1]:.6f}")

    # y = tanh(x) with the non-linear perceptron (tanh here, sigmoid below)
    w, hist = simple_nonlinear_perceptron(
        add_bias(xs), np.tanh(xs.ravel()), alpha=0.1, epochs=2000,
        activation_fn=np.tanh, derivative_fn=lambda h: 1 - np.tanh(h) ** 2)
    print(f"tanh non-linear        -> MSE: {hist[-1]:.6f}\n")


# ---------------------------------------------------------------------------
# Exercise 1, part 1: learning comparison on ALL samples
# ---------------------------------------------------------------------------
def main():
    run_validation()

    x_raw, y, flagged, feature_cols = load_fraud_data()
    x_scaled, _, _ = standardize(x_raw)    # all samples: no split yet
    x = add_bias(x_scaled)
    print(f"=== Exercise 1 === {x.shape[0]} samples, features: {feature_cols}")

    w_lin, hist_lin = simple_linear_perceptron(x, y)
    w_sig, hist_sig = simple_nonlinear_perceptron(x, y)

    pred_lin = x @ w_lin
    pred_sig = sigmoid(x @ w_sig)

    for name, pred, hist in [("Linear", pred_lin, hist_lin),
                             ("Sigmoid", pred_sig, hist_sig)]:
        print(f"{name:8s} final MSE: {hist[-1]:.5f} | "
              f"MAE: {np.mean(np.abs(pred - y)):.4f} | "
              f"pred range: [{pred.min():.3f}, {pred.max():.3f}] | "
              f"outside [0,1]: {np.mean((pred < 0) | (pred > 1)) * 100:.1f}%")

    print("\nWeights (bias first):")
    for name, wl, ws in zip(["bias"] + feature_cols, w_lin, w_sig):
        print(f"  {name:30s} linear: {wl:+.4f}   sigmoid: {ws:+.4f}")

    # Plots: loss curves and predicted vs. BigModel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(hist_lin, label="Linear")
    axes[0].plot(hist_sig, label="Sigmoid")
    axes[0].set(title="Training MSE per epoch", xlabel="Epoch", ylabel="MSE",
                yscale="log")
    axes[0].legend()

    for ax, pred, name in [(axes[1], pred_lin, "Linear"),
                           (axes[2], pred_sig, "Sigmoid")]:
        ax.scatter(y, pred, s=3, alpha=0.3, c=flagged, cmap="coolwarm")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"{name}: prediction vs. BigModel",
               xlabel="BigModel probability", ylabel="TinyModel prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
