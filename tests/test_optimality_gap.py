"""Optimality gap: learned path vs BFS optimal."""

from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from maze_solver.utils import bfs, get_learned_path
from train import train


def test_optimality_gap_empty_grid():
    size = 5
    walls = set()
    opt = bfs(size, walls, (0, 0), (size - 1, size - 1))
    assert opt is not None
    assert len(opt) == 9  # 8 steps -> 9 positions

    env = MazeEnv(size=size, seed=0)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    train(agent, env, episodes=1000, seed=0)
    learned = get_learned_path(env, agent)
    # Learned path should be within 4 steps of optimal after 1k episodes
    assert len(learned) <= len(opt) + 4


def test_bfs_with_walls():
    from maze_solver.utils import bfs

    walls = {(1, 1), (1, 2)}
    path = bfs(3, walls, (0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path
