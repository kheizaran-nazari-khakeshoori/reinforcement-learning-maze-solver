"""CLI and config: DEFAULT used by training CLI."""

import config
from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent


def test_config_default_has_required_keys():
    for key in ["size", "episodes", "alpha", "gamma", "epsilon", "decay", "min_eps", "seed"]:
        assert key in config.DEFAULT


def test_config_defaults_match_cli():
    import argparse
    from train import set_seed

    # Simulate CLI defaults from config
    assert config.DEFAULT["size"] == 5
    assert config.DEFAULT["seed"] == 0
    env = MazeEnv(size=config.DEFAULT["size"], seed=config.DEFAULT["seed"])
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=config.DEFAULT["alpha"])
    assert agent.alpha == 0.1


def test_env_seed_via_config():
    env = MazeEnv(size=5, seed=config.DEFAULT["seed"])
    s1, _ = env.reset(seed=config.DEFAULT["seed"])
    s2, _ = env.reset(seed=config.DEFAULT["seed"])
    assert s1 == s2 == 0
