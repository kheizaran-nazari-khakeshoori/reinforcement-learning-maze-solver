"""Maze environment module.

Gymnasium-compatible Gridworld with seeding support.
"""

import gymnasium as gym
import numpy as np
try:
    from maze_solver.gui import MazeGUI
    _HAS_GUI=True
except: _HAS_GUI=False

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

DEFAULT_SIZE = 5

__all__ = ["MazeEnv", "DEFAULT_SIZE"]


class MazeEnv(gym.Env):
    """Custom Gridworld maze environment with seed support.

    Grid size N x N, start at (0,0), goal at (N-1,N-1).
    Supports walls, traps, and discrete actions 0:up,1:right,2:down,3:left.
    Observation is single Discrete state index = r*N + c.
    Seed handling uses Gymnasium's seeding via super().reset(seed=seed).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=DEFAULT_SIZE, walls=None, traps=None, seed=None):
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
        # Seed action/observation spaces for reproducibility
        if seed is not None:
            self.reset(seed=seed)

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

    def _render_text(self):
        rows = []
        border = "+" + "--" * self.size + "+"
        rows.append(border)
        for r in range(self.size):
            cells = [self._ansi_cell((r, c)) for c in range(self.size)]
            rows.append("|" + " ".join(cells) + "|")
        rows.append(border)
        return "\n".join(rows)

    def _plot_grid(self):
        if not _HAS_MPL:
            return None
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        ax.grid(True)
        ax.invert_yaxis()
        return fig, ax

    def _color_for(self, pos):
        if pos == self.agent_pos:
            return "blue"
        if pos == self.goal:
            return "gold"
        if pos in self.traps:
            return "red"
        if pos in self.walls:
            return "gray"
        return "white"

    def render(self, mode="human"):
        if mode=="pygame" and _HAS_GUI:
            if not hasattr(self, "gui") or self.gui is None:
                self.gui=MazeGUI(self)
            return self.gui.loop_once()
        if mode in ("human", "ansi"):
            text = self._render_text()
            print(text)
            return text
        if mode == "visual":
            return self._plot_grid()
        return self._render_text()

    def _ansi_cell(self, pos):
        if pos == self.agent_pos:
            return "\033[94mA\033[0m"
        if pos == self.goal:
            return "\033[92mG\033[0m"
        if pos in self.traps:
            return "\033[91mX\033[0m"
        if pos in self.walls:
            return "\033[90m#\033[0m"
        return "."

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium handles seeding of self.np_random and spaces
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos), {}

    def step(self, action):
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        r, c = self.agent_pos
        cand = (r + dr, c + dc)
        if self._is_valid(cand):
            self.agent_pos = cand
        if self.agent_pos in self.traps:
            reward = -1.0
        elif self.agent_pos == self.goal:
            reward = 1.0
        else:
            reward = -0.01
        terminated = self._is_terminal(self.agent_pos)
        return self._pos_to_state(self.agent_pos), reward, terminated, False, {}


    def get_state(self):
        return self._pos_to_state(self.agent_pos)

    def distance_to_goal(self):
        return abs(self.agent_pos[0]-self.goal[0])+abs(self.agent_pos[1]-self.goal[1])

    def render_text(self):
        return self._render_text()
