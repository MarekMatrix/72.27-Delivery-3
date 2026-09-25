# TP3: Simple and Multilayer Perceptrons

Sistemas de Inteligencia Artificial (72.27), ITBA, second semester 2026. Group of six.

We implement the simple perceptron (step, linear, non-linear) and the multilayer perceptron from scratch with numpy, validate them on toy problems, and apply them to fraud-probability estimation and handwritten-digit classification for a fictional client, CompanyX.

## The assignment

| Part | Problem | Model | Data | Submitted |
|---|---|---|---|---|
| Validation | AND; fit `y = x`; fit `y = tanh(x)`; XOR with `[2, 2, 1]` and `[2, 3, 2, 1]` | Simple perceptron (step, linear, non-linear); MLP | Synthetic | No, but they are our tests |
| Exercise 1 | Probability that an online transaction is fraudulent, with a small "TinyModel" replacing the client's expensive "BigModel" (knowledge distillation) | Simple perceptron, linear vs non-linear | `fraud_dataset.csv` | Yes |
| Exercise 2 | Classify handwritten digits 0–9 | MLP | `digits.csv` for training and tuning; `digits_test.csv` as the production stand-in | Yes |
| Exercise 3 | Exercise 2 again, reaching accuracy ≥ 98% | MLP | Adds `more_digits.csv` | Yes |

### Questions to answer

The labels are used in the work areas below.

**Exercise 1**

- Learning study, using every sample:
  **1L-a** is there underfitting?
  **1L-b** is there capacity saturation?
  **1L-c** which perceptron goes on to the generalization study?
- Generalization study, on the chosen perceptron only:
  **1G-a** which evaluation metrics, and why?
  **1G-b** how is the dataset handled, and how is the best training set chosen?
  **1G-c** best model for the client, plus a recommended fraud-detection threshold.
- Optional: ReLU in the non-linear perceptron (practical); features to build or discard (theory); calibration (theory).
- The assignment stresses exploring `fraud_dataset.csv` before modelling: column documentation, value ranges, composition, cleanliness.

**Exercise 2**

- **2-a** how do we evaluate the system?
- **2-b** which variants did we try? At minimum: learning rate, architecture, optimization method.

**Exercise 3**

- **3-a** best result with the new data.
- **3-b** techniques used to improve on Exercise 2.
- **3-c** other factors behind the change in performance.

**Optional for Exercises 2 and 3:** robustness to noise (e.g. Gaussian) on the test set; interpretability through attribution methods.

Optional parts start only once every required part is done; the assignment is explicit about this.

The assignment also recommends matrix operations, progress reporting, a stored and extensible configuration, saving and loading models together with their configuration, and keeping experiment runs separate from analysis. The structure below is built around those.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # uv sync: creates .venv with numpy, matplotlib, pytest
make test      # runs the validation exercises in tests/
```

The course materials are committed in `data/`, under their original filenames, so `make install` is all you need — there is nothing to download:

| File | Used by |
|---|---|
| `fraud_dataset.csv` | Exercise 1 |
| `digits.csv` | Exercise 2 (training and tuning) |
| `digits_test.csv` | Exercises 2 and 3 (production stand-in) |
| `more_digits.csv` | Exercise 3 |
| `fraud_dataset_documentation.pdf` | column documentation for Exercise 1 |
| `digit_dataset_loader.py` | the course's reference loader for the digit CSVs; reference only, `data.py` is ours |

These files are committed deliberately, as an exception to the rule below about generated artefacts: they are inputs, they never change, and having them in the repository means everyone trains on byte-identical data. Keep the names as they are — `data.py` and the report both refer to them. Anything else you drop into `data/` stays gitignored.

Experiments will run through `cli.py`, which is not implemented yet (area A6).

## Repository structure

```
.
├── src/perceptrons/
│   ├── activations.py    activation functions and derivatives            A1
│   ├── losses.py         loss functions and their gradients              A1
│   ├── simple.py         simple perceptron: step, linear, non-linear     A1
│   ├── multilayer.py     multilayer perceptron                           A2
│   ├── optimizers.py     weight update rules                             A3
│   ├── training.py       epochs, shuffling, batching                     A3
│   ├── data.py           loading, scaling, splitting                     A4
│   ├── metrics.py        evaluation metrics                              A5
│   ├── plots.py          report figures, built from results/ only        A5
│   ├── config.py         experiment configuration                        A6
│   ├── experiment.py     runs and records experiments                    A6
│   └── cli.py            command-line entry point                        A6
├── tests/                validation exercises and per-module tests
├── notebooks/            exploration only; never imported by src/
├── data/                 course datasets and their documentation (committed)
├── results/              run outputs and figures (gitignored)
├── docs/                 report and slides
├── pyproject.toml        dependencies, managed with uv (uv.lock is committed)
├── Makefile              install, test, clean
└── CLAUDE.md             settings for members who use Claude Code
```

Conventions:

- `src/` never imports from `notebooks/`. Anything that matters moves into `src/` and is imported back.
- Generated files (figures, results, model weights) are never committed. Commit the script or config that produces them.
- Code, identifiers and comments are in English.
- `CLAUDE.md` only affects Claude Code sessions; its structure and commit conventions are the parts that apply to everyone.

## Pipeline

```mermaid
flowchart TD
    CSV[("data/*.csv")]
    TESTS["tests/<br/>validation exercises"]

    subgraph RUN["Run: writes to results/"]
        CLI["cli.py (A6)<br/>choose exercise and config"]
        CFG["config.py (A6)<br/>hyperparameters"]
        DATA["data.py (A4)<br/>load, scale, split"]
        EXP["experiment.py (A6)<br/>orchestrate, report progress, save and load"]
        TRAIN["training.py (A3)<br/>epochs, shuffling, batches"]
        MODEL["simple.py (A1)<br/>multilayer.py (A2)"]
        ACT["activations.py (A1)"]
        LOSS["losses.py (A1)"]
        OPT["optimizers.py (A3)"]
        MET["metrics.py (A5)"]
    end

    RES[("results/run-id/<br/>config, per-epoch history, timing, weights")]

    subgraph ANALYSE["Analyse: reads results/ only"]
        PLOT["plots.py (A5)"]
    end

    DOCS["docs/<br/>report and slides"]

    CLI --> CFG
    CFG --> EXP
    CSV --> DATA
    DATA -->|"training, validation and test arrays"| EXP
    EXP -->|"model, data, config"| TRAIN
    TRAIN <-->|"forward and backward pass"| MODEL
    ACT -.-> MODEL
    LOSS -.-> TRAIN
    TRAIN -->|"gradients"| OPT
    OPT -->|"updated weights"| MODEL
    TRAIN -->|"per-epoch loss and time"| EXP
    EXP -->|"predictions and targets"| MET
    MET -->|"scores"| EXP
    EXP --> RES
    RES --> PLOT
    PLOT -->|"figures, not committed"| DOCS
    TESTS -.->|"call directly"| MODEL
```

An experiment starts from `cli.py` with a config. `experiment.py` gets the data from `data.py`, passes model, data and config to the training loop, scores the result with `metrics.py`, and writes everything about the run to `results/<run-id>/`. `plots.py` only reads from `results/`, so figures can be redrawn without retraining. The validation tests call the models directly and skip the pipeline.

## Work areas

Each member owns one area: a set of modules in phase 1 (implementation) and the report questions closest to those modules in phase 2 (experiments). Areas that are heavy on implementation carry lighter experiment work, and the other way round.

| Area | Owner | Modules | Phase 2 questions |
|---|---|---|---|
| A1: Simple perceptron and neuron maths | _TBD_ | `activations.py`, `losses.py`, `simple.py` | 1L-a, 1L-b, 1L-c; optional ReLU |
| A2: Multilayer perceptron | _TBD_ | `multilayer.py` | 2-b architecture; optional interpretability |
| A3: Optimization and training loop | _TBD_ | `optimizers.py`, `training.py` | 2-b learning rate and optimizers |
| A4: Data | _TBD_ | `data.py`, exploration notebooks | 1G-b, 3-c; optional features |
| A5: Evaluation and figures | _TBD_ | `metrics.py`, `plots.py` | 1G-a, 1G-c, 2-a; optional calibration |
| A6: Experiment infrastructure | _TBD_ | `config.py`, `experiment.py`, `cli.py` | 3-a, 3-b; optional noise robustness |

Every area also owns the tests for its modules.

### A1: Simple perceptron and neuron maths

Owns `activations.py`, `losses.py`, `simple.py`, `tests/test_simple.py`.

Phase 1 is done when:

- the step variant learns AND, the linear variant fits `y = x`, the non-linear variant fits `y = tanh(x)`, and the step variant fails on XOR, all as passing tests;
- activations and losses, with their derivatives and gradients, are merged into `develop` early, because A2 and A3 build on them.

Phase 2: 1L-a, 1L-b, 1L-c; optional ReLU. The answer to 1L-c decides which perceptron A4 and A5 use for the generalization study.

### A2: Multilayer perceptron

Owns `multilayer.py`, `tests/test_multilayer.py`.

Phase 1 is done when:

- one forward and backward step for `[2, 2, 1]` and for `[2, 3, 2, 1]` has been calculated by hand, as the assignment recommends, and a test reproduces the numbers;
- XOR is learned with both architectures;
- forward and backward passes are matrix operations, not per-neuron loops.

The hand calculations can start on day one; running code needs A1's activations.

Phase 2: 2-b architecture variants; optional interpretability.

### A3: Optimization and training loop

Owns `optimizers.py`, `training.py`, `tests/test_optimizers.py` (to create).

Phase 1 is done when:

- plain gradient descent and each alternative we compare in Exercise 2 minimize a function whose minimum is known in advance (e.g. a one-dimensional quadratic), with no model involved;
- the training loop handles epochs, shuffling and batch size, and hands per-epoch loss and timing to `experiment.py`.

Phase 2: 2-b learning-rate and optimizer variants.

### A4: Data

Owns `data.py`, the exploration notebooks, `tests/test_data.py` (to create).

Phase 1 is done when:

- all four CSVs load into arrays with the agreed shapes and target encoding;
- a notebook explores `fraud_dataset.csv` along the assignment's questions, and the digit datasets (class balance, value ranges);
- scaling and splitting utilities exist for the Exercise 1 generalization study and for tuning on `digits.csv`, with `digits_test.csv` kept out of all tuning.

Phase 2: 1G-b, 3-c; optional feature construction and removal.

### A5: Evaluation and figures

Owns `metrics.py`, `plots.py`, `tests/test_metrics.py` (to create).

Phase 1 is done when:

- every metric matches values computed by hand on small made-up predictions;
- plotting helpers produce figures with labelled axes and units from a results record. Use a hand-written fake record until A6's format is merged.

Phase 2: 1G-a, 1G-c, 2-a; optional calibration.

### A6: Experiment infrastructure

Owns `config.py`, `experiment.py`, `cli.py`, `tests/test_experiment.py` (to create).

Phase 1 is done when:

- a config written to disk and read back is unchanged;
- a run writes its config, per-epoch history, timing and weights to `results/<run-id>/`, and reports progress while training;
- a saved model loads with its config and continues training;
- the CLI runs any exercise from a config file.

Build against a dummy model until A1 and A2 merge.

Phase 2: 3-a, 3-b, drawing on A2 and A3; optional noise robustness.

## Before anyone branches: agree on the contracts

Parallel work only holds if the boundaries between areas are fixed first. Settle these at a kickoff and record each decision in the docstring of the module that provides it.

| Contract | Provided by → used by | To decide |
|---|---|---|
| Array shapes | A4 → everyone | Shapes of input and output matrices; how digit labels are encoded |
| Activation | A1 → A2 | How a function and its derivative are exposed |
| Loss | A1 → A3 | Loss value, and its gradient with respect to the network output |
| Model | A1, A2 → A3, A6 | Prediction; gradients for every parameter; access to parameters for updating and saving |
| Optimizer | A3 → A6 | How parameters and gradients go in; where optimizer state (e.g. running averages) lives |
| Epoch record | A3 → A6 | What the training loop reports after each epoch |
| Results record | A6 → A5 | What a run writes to `results/`, and in which format |
| Config fields | everyone → A6 | Each area lists the hyperparameters it needs |
| Randomness | everyone | How seeds are set so runs are reproducible |

These are design questions for the whole group rather than one owner, since everyone defends them orally:

- Is the simple perceptron a special case of the multilayer perceptron, or a separate implementation?
- Which update mode is the default: online, mini-batch or full batch?
- Which output activation and loss for Exercise 1, and which for Exercises 2 and 3?

## Git workflow

### Branches

- `main`: reviewed, working states only. Updated from `develop` at milestones; nobody commits to it directly.
- `develop`: the integration branch. Every work branch starts here and comes back through a pull request.
- Work branches, created from `develop`:
  - `feat/<short-name>` for implementation, e.g. `feat/mlp-backprop`
  - `fix/<short-name>` for bug fixes
  - `exp/<short-name>` for experiment studies, e.g. `exp/ex2-learning-rate`

### One-time setup

Done once, by whoever creates the GitHub repository:

```bash
git add .
git commit -m "chore: scaffold TP3 project structure and tooling"
git remote add origin <repository-url>
git push -u origin main
git switch -c develop
git push -u origin develop
```

Then protect `main` and `develop` in the GitHub repository settings so changes only land through pull requests.

### Everyday flow

```bash
git switch develop
git pull
git switch -c feat/mlp-backprop
# work, commit
git fetch origin
git rebase origin/develop
make test
git push -u origin feat/mlp-backprop   # add --force-with-lease if the branch was already pushed
# open a pull request into develop
```

### Rules

- Rebase only branches that you alone push to. On a branch shared with someone, merge `develop` into it instead.
- Every pull request is reviewed by someone from a **different** area. We all defend the whole project orally, so review is also how everyone learns the other areas.
- `make test` passes before merging.
- Keep pull requests small and within your own area's files. If you have to change another area's module, its owner reviews.
- Commit messages start with `feat:`, `fix:`, `refactor:`, `docs:`, `chore:` or `test:`, use the imperative mood, keep the subject under about 72 characters, and cover one logical change.
- Nothing in `results/` is ever committed. In `data/`, only the original course materials are; anything else you put there stays ignored.

## Milestones

1. **Kickoff:** owners assigned, contracts and design questions settled, `develop` created.
2. **Foundations:** A1's activations and losses, A4's loaders and A6's config merged into `develop` as small, early pull requests, since everyone else depends on them.
3. **Validation passes:** every test passes on `develop`; merge `develop` into `main`.
4. **Experiments:** Exercise 1 (A1, A4, A5) and Exercise 2 (A2, A3, A5) run in parallel. The Exercise 1 generalization study starts once 1L-c has picked a perceptron. Exercise 3 (A6, A4) starts from the best Exercise 2 setup. Merge into `main` after each exercise.
5. **Report and slides** in `docs/`, then optional parts if time allows. Final state merged into `main`.
