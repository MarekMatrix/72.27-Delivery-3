"""Experiment configuration: everything needed to reproduce a run, stored with its results.

Pure data holder -- no logic lives here.
"""

from dataclasses import dataclass, field
from enum import Enum


class OptimizerMethod(str, Enum):
    GradientDescent = "gradient_descent"
    Momentum = "momentum"
    Adam = "adam"


class ActivationMethod(str, Enum):
    Sigmoid = "sigmoid"
    Tangent = "tangent"


@dataclass
class Config:
    # Problem parameters (not hyperparameters)
    n_classes: int = 10
    n_features: int = 784

    # Hyperparameters
    optimizer_method: OptimizerMethod = OptimizerMethod.GradientDescent
    activation_method: ActivationMethod = ActivationMethod.Sigmoid
    activation_parameter: float = 0.5

    # Batch size alone decides the update type: 1 = online, >= n_samples = full batch,
    # anything in between = mini-batch
    batch_size: int = 32
    hidden_layers: list[int] = field(default_factory=lambda: [16, 16])
    learning_rate: float = 0.1
    epochs: int = 50

    # The rng of a run is created from this seed, so the saved config reproduces the run
    random_seed: int = 0

    extra: dict = field(default_factory=dict)  # escape hatch for one-off experiment knobs
