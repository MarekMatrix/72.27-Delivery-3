"""Experiment configuration: everything needed to reproduce a run, stored with its results."""

# TODO: the hyperparameters Exercise 2 varies — learning rate, architecture, optimizer — plus any others
# TODO: write to and read from disk


"""Hyperparameter container for a GA run.

Pure data holder -- no logic lives here. Whoever owns engine.py decides
which of these are actually read; add fields as the engine needs them.
"""

from dataclasses import dataclass, field
from enum import Enum


class OptimizerMethod(str, Enum):
    GradientDescent = "gradient_descent"
    Momentum = "momentum"
    Adam = "adam"


class BatchMethod(str, Enum):
    Online = "online"
    MiniBatch = "mini_batch"
    Batch = "batch"


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
    batch_method: BatchMethod = BatchMethod.Batch
    activation_method: ActivationMethod = ActivationMethod.Sigmoid
    activation_parameter: float = 0.5
    
    batch_size: int = 32
    hidden_layers: list = [16, 16]
    learning_rate: float = 0.1

    random_seed: int | None = None

    extra: dict = field(default_factory=dict)  # escape hatch for one-off experiment knobs
