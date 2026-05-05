"""Training loop for Gridworld agents."""

import argparse
from typing import Tuple, List

from maze_solver.agents import BaseAgent, QLearningAgent
from maze_solver.env import MazeEnv


def run_episode(env: MazeEnv, agent: BaseAgent, epsilon: float = 0.1) -> Tuple[float, int]:
    """Run a single episode with epsilon-greedy policy.

    Args:
        env: Maze environment.
        agent: Agent implementing act() and update().
        epsilon: Exploration rate.

    Returns:
        (total_reward, steps) for the episode.
    """
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Unified update interface: Q-Learning ignores next_action, SARSA uses it.
        # Keep backward-compatible try/except until agents are unified (next commit).
        try:
            agent.update(state, action, reward, next_state, done)
        except TypeError:
            agent.update(state, action, reward, next_state, action, done)
        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, steps


def train(
    agent: BaseAgent,
    env: MazeEnv,
    episodes: int = 100,
    epsilon: float = 1.0,
    decay: float = 0.995,
    min_eps: float = 0.05,
) -> Tuple[List[float], List[int]]:
    """Train agent with epsilon decay.

    Args:
        agent: RL agent.
        env: Environment.
        episodes: Number of episodes.
        epsilon: Initial epsilon.
        decay: Epsilon decay per episode.
        min_eps: Floor for epsilon.

    Returns:
        (rewards, steps) histories.
    """
    cur_eps = epsilon
    rewards: List[float] = []
    steps_hist: List[int] = []

    for _ in range(episodes):
        reward, steps = run_episode(env, agent, cur_eps)
        rewards.append(reward)
        steps_hist.append(steps)
        cur_eps = max(min_eps, cur_eps * decay)

    return rewards, steps_hist


def evaluate_greedy(agent: BaseAgent, env: MazeEnv, max_steps: int = 50) -> int:
    """Run greedy rollout and return steps to termination."""
    state, _ = env.reset()
    steps = 0
    while steps < max_steps:
        action = agent.act(state, 0.0)
        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        state = next_state
        steps += 1
    return steps


def train_with_logging(agent: BaseAgent, env: MazeEnv, episodes: int = 10):
    """Compatibility wrapper for logging-style training."""
    return train(agent, env, episodes=episodes)


def get_rewards(agent: BaseAgent, env: MazeEnv) -> float:
    """Run one episode with epsilon=0.1 and return reward."""
    reward, _ = run_episode(env, agent, 0.1)
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Maze solver")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()

    env = MazeEnv(size=5)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    train(agent, env, episodes=args.episodes)
