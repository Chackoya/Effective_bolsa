# TensorGP

TensorGP is a general-purpose Genetic Programming engine that uses TensorFlow with Python to accelerate fitness evaluation through operator vectorization.

## Installation

Import the engine with:

```python
from engine import *
```

You can use the [pip](https://pip.pypa.io/en/stable/) package manager to install the list of dependencies provided in "requirements.txt":
```bash
pip install -r requirements.txt
```

## Usage

The Python files ending with "_example.py" demonstrate how to call the engine through examples:

"pagie_example.py" exemplifies a classical symbolic regression problem.

"nima_example.py" exemplifies the application of TensorGP to an evolutionary art problem.

A general evolutionary run must first be initialized with the Engine() constructor.
```python
engine = Engine(fitness_func = rmse_function,
                population_size = 30,
                tournament_size = 3,
                mutation_rate = 0.1,
                crossover_rate = 0.9,
                max_tree_depth = 20,
                target_dims=[1000, 1000, 3],
                method='ramped half-and-half',
                objective='maximizing',
                device='/gpu:0',
                stop_value=number_generations)
```


After the run() method may be called to run the remaining evolutionary process, as so:

```python
engine.run()

```

