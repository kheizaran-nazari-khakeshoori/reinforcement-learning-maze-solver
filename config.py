"""Central hyperparameters and defaults."""

from typing import Dict, Union

DEFAULT: Dict[str, Union[int, float]] = {
    "size": 5,
    "episodes": 5000,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "decay": 0.995,
    "min_eps": 0.05,
    "seed": 0,  # reproducible default
}

# Reproducibility: seed controls numpy/random and env seeding via train.py --seed
# All CLI defaults are taken from DEFAULT so config is single source of truth.
