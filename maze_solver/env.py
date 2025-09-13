"""Maze environment module."""

import gymnasium as gym
import numpy as np

DEFAULT_SIZE = 5

__all__ = ["MazeEnv", "DEFAULT_SIZE"]


class MazeEnv(gym.Env):
    """Custom Gridworld maze environment.

    Grid size N x N, start at (0,0), goal at (N-1,N-1).
    Supports walls, traps, and discrete actions 0:up,1:right,2:down,3:left.
    Observation is single Discrete state index = r*N + c.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=DEFAULT_SIZE, walls=None, traps=None):
        super().__init__()
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.walls = set(walls) if walls else set()
        self.traps = set(traps) if traps else set()
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("walls cannot overlap start or goal")
        self.agent_pos = self.start
        self.grid = np.zeros((size, size), dtype=int)
        for w in self.walls:
            self.grid[w] = 1
        for t in self.traps:
            self.grid[t] = 2
        self.grid[self.goal] = 3
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(size * size)

    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def _is_valid(self, pos):
        r, c = pos
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if pos in self.walls or self.grid[pos] == 1:
            return False
        return True

    def _is_terminal(self, pos):
        return pos == self.goal or pos in getattr(self, "traps", set())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        r, c = self.agent_pos
        cand = (r + dr, c + dc)
        self.agent_pos = cand
        return self._pos_to_state(self.agent_pos), 0.0, False, False, {}

