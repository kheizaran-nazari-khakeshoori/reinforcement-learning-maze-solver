"""Sanity checks for CI and basic assertions."""

def test_sanity_arithmetic():
    assert 1 + 2 == 3


def test_sanity_imports():
    from maze_solver.env import MazeEnv
    from maze_solver.agents import QLearningAgent

    env = MazeEnv(size=3)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    assert env.observation_space.n == 9
    assert agent.q_table.shape == (9, 4)
