"""Central hyperparameters and defaults with dataclass."""

from dataclasses import asdict, dataclass
from typing import Dict, Union

@dataclass
class Config:
    size: int = 5
    episodes: int = 5000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    decay: float = 0.995
    min_eps: float = 0.05
    seed: int = 0

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

# dataclass instance for typed access
CONFIG = Config()

def get_config_dict() -> Dict[str, Union[int, float]]:
    return asdict(CONFIG)

# Reproducibility: seed controls numpy/random and env seeding via train.py --seed
# All CLI defaults are taken from DEFAULT so config is single source of truth.

# seed logged for reproducibility, used by train.py --seed
# defaults synced

# hydra-compatible: Config dataclass can be used with hydra if installed
