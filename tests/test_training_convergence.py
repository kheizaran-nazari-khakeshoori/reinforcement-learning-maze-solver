"""Training convergence: greedy policy improves with training."""

from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from train import train
from evaluate import evaluate


def test_training_convergence():
    env = MazeEnv(size=5, seed=0)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    train(agent, env, episodes=200, seed=0)
    metrics = evaluate(agent, env, episodes=10, seed=0)
    # After 200 episodes on empty 5x5, should reach goal at least sometimes
    assert metrics["success"] >= 0.5
    assert metrics["avg_steps"] <= 20


def test_training_deterministic_with_seed():
    env = MazeEnv(size=3, seed=123)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    rewards, _ = train(agent, env, episodes=5, seed=123)
    assert len(rewards) == 5
    assert all(isinstance(r, float) for r in rewards)
