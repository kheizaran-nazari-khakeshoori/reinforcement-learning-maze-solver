"""Reproducibility: same seed gives same training trajectory."""

from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from train import train, set_seed


def test_reproducibility_same_seed():
    seed = 42
    env1 = MazeEnv(size=5, seed=seed)
    ag1 = QLearningAgent(env1.observation_space.n, env1.action_space.n)
    set_seed(seed)
    r1, _ = train(ag1, env1, episodes=10, seed=seed)

    env2 = MazeEnv(size=5, seed=seed)
    ag2 = QLearningAgent(env2.observation_space.n, env2.action_space.n)
    set_seed(seed)
    r2, _ = train(ag2, env2, episodes=10, seed=seed)

    assert r1 == r2


def test_reproducibility_config_seed():
    from config import DEFAULT

    assert "seed" in DEFAULT
    assert isinstance(DEFAULT["seed"], int)
