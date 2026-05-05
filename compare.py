"""Compare Q-Learning vs SARSA training curves."""

from typing import Tuple, List

from maze_solver.agents import QLearningAgent, SarsaAgent
from maze_solver.env import MazeEnv
from train import run_episode


def run_compare(episodes: int = 100) -> Tuple[List[float], List[float]]:
    """Joint training comparison with shared epsilon schedule."""
    env_q = MazeEnv(size=4)
    env_s = MazeEnv(size=4)
    q_agent = QLearningAgent(env_q.observation_space.n, env_q.action_space.n)
    s_agent = SarsaAgent(env_s.observation_space.n, env_s.action_space.n)

    q_rewards: List[float] = []
    s_rewards: List[float] = []
    epsilon = 1.0

    for _ in range(episodes):
        reward_q, _ = run_episode(env_q, q_agent, epsilon)
        q_rewards.append(reward_q)

        reward_s, _ = run_episode(env_s, s_agent, epsilon)
        s_rewards.append(reward_s)

        epsilon = max(0.05, epsilon * 0.995)

    return q_rewards, s_rewards


def quick_compare() -> Tuple[List[float], List[float]]:
    return run_compare(episodes=10)


def compare_lengths():
    from maze_solver.utils import bfs

    return bfs(3, set(), (0, 0), (2, 2))
