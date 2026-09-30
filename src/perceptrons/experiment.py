"""
Runs a configured experiment and records it under results/.

This module is responsible for storing the outputs of an experiment.
Plotting and analysis are handled separately in plots.py.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass

class ExperimentResult:

    """

    Stores the results produced by one experiment run.

    Attributes

    ----------

    train_loss : list[float]

        Training loss recorded after each epoch.

    validation_loss : list[float]

        Validation loss recorded after each epoch.

    train_accuracy : list[float]

        Training accuracy recorded after each epoch.

    validation_accuracy : list[float]

        Validation accuracy recorded after each epoch.

    test_loss : float | None

        Final loss on the held-out test set.

        This should only be computed after model selection.

    test_accuracy : float | None

        Final accuracy on the held-out test set.

        This should only be computed after model selection.

    training_time : float

        Total training time in seconds.

    """

    train_loss: list[float]
    validation_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]
    test_loss: float | None
    test_accuracy: float | None
    training_time: float


def save_result(result: ExperimentResult, path: str | Path) -> None:
    """
    Save an experiment result as JSON.

    Parameters
    ----------
    result : ExperimentResult
        Results produced by an experiment.

    path : str or Path
        File where the results should be stored.
    """
    path = Path(path)

    # Create the directory if it does not exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=4)


# TODO: integrate with training.py when Exercise 2 is ready
# TODO: report progress while training
# TODO: save model weights
# TODO: load model and configuration to resume training