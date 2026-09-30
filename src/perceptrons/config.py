"""Experiment configuration: everything needed to reproduce a run, stored with its results."""



from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class ExperimentConfig:
    exercise: int
    seed: int
    learning_rate: float
    epochs: int
    batch_size: int
    architecture: list[int]
    activation: str
    optimizer: str

    def save(self, path: str | Path) -> None:

        """
        Save the experiment configuration to a JSON file.

        """

        path = Path(path)

        # Create parent directories if they do not already exist.

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert the dataclass to a dictionary and store it as JSON.

        with path.open("w", encoding="utf-8") as file:

            json.dump(asdict(self), file, indent=4)

    @classmethod

    def load(cls, path: str | Path) -> "ExperimentConfig":

        """
        Load an experiment configuration from a JSON file.
        """

        path = Path(path)

        with path.open("r", encoding="utf-8") as file:

            data = json.load(file)

        return cls(**data)

    '''
    Example usage:
    config = ExperimentConfig(
    exercise=3,
    seed=42,
    learning_rate=0.001,
    epochs=100,
    batch_size=64,
    architecture=[784, 128, 64, 10],
    activation="relu",
    optimizer="adam"
    )

    config.save("results/test_config.json")

    loaded_config = ExperimentConfig.load("results/test_config.json")

    print(config)
    print(loaded_config)
    print("Configurations are equal:", config == loaded_config)
        
    
    
    
    '''