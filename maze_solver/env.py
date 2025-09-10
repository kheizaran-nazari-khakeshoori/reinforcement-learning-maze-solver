"""Maze environment module."""

import gymnasium as gym
import numpy as np


class MazeEnv(gym.Env):
    """Custom Gridworld maze environment.

    Grid size N x N, start at (0,0), goal at (N-1,N-1).
    Supports walls, traps, and discrete actions 0:up,1:right,2:down,3:left.
    Observation is single Discrete state index = r*N + c.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=5, walls=None):
        super().__init__()
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = set(walls) if walls else set()
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("walls cannot overlap start or goal")
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(size * size)

    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def _is_terminal(self, pos):
        return pos == self.goal or pos in getattr(self, "traps", set())

