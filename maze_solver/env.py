"""Maze environment module."""

import gymnasium as gym
import numpy as np


class MazeEnv(gym.Env):
    """Custom Gridworld maze environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

