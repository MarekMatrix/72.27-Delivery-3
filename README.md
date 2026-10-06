# TP3 - Perceptrons

Sistemas de Inteligencia Artificial (72.27), ITBA, 2Q 2026. Group 13.

Simple and multilayer perceptrons written from scratch with numpy. The simple perceptron is used for the fraud dataset (exercise 1) and the multilayer perceptron for classifying handwritten digits (exercises 2 and 3).

## Setup

You need [uv](https://docs.astral.sh/uv/).

```
make install
```

## Running

Train a network on the digits from the command line:

```
uv run python -m perceptrons.cli --learning-rate 1.0 --activation-parameter 1 \
    --hidden-layers 128 --batch-size 8 --epochs 50
```

Use `--help` to see the other options (dataset, optimizer, activation, split, augmentation, saving the model). Each run is saved as a json file in `results/`.

Exercise 1:

```
uv run python src_ex1/perceptron_exercise1/simple.py
```

Exercise 2:

```
uv run python -m perceptrons.sweep           # train the final configuration on digits.csv (seeds 0, 1, 2)
uv run python -m perceptrons.evaluate_test   # final model on digits_test.csv
```

The results are analysed in the notebooks in `notebooks/`.

## Code

The multilayer perceptron is in `src/perceptrons/`:

- `activations.py`, `losses.py`: sigmoid and tanh with their derivatives, mean squared error
- `multilayer.py`: multilayer perceptron, forward and backward pass with matrices
- `optimizers.py`: gradient descent, momentum and Adam
- `training.py`: training loop with epochs, shuffling and mini-batches
- `data.py`: loading the csv files, splitting, one-hot encoding, oversampling and augmentation
- `metrics.py`: accuracy, recall and confusion matrix
- `config.py`, `experiment.py`, `cli.py`: configuration, running and saving experiments
- `sweep.py`, `evaluate_test.py`: exercise 2 training runs and final evaluation on `digits_test.csv`

Exercise 1 is in `src_ex1/perceptron_exercise1/simple.py` (step, linear and non-linear simple perceptrons). The datasets are in `data/`.
