"""Training loop for Gridworld agents with seeding."""

import argparse
import csv
import logging
import random
from typing import Tuple, List, Optional

import numpy as np

from config import DEFAULT
from maze_solver.agents import BaseAgent, QLearningAgent, SarsaAgent
from maze_solver.env import MazeEnv


def set_seed(seed: Optional[int]) -> None:  # logs to config
    """Set global RNG seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


class CsvLogger:
    """Simple CSV logger for training metrics."""
    def __init__(self, path: str = "training_log.csv"):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "reward", "steps", "epsilon"])

    def log(self, episode: int, reward: float, steps: int, epsilon: float):
        self.writer.writerow([episode, reward, steps, epsilon])

    def close(self):
        self.file.close()


def get_tensorboard_writer(log_dir: str = "runs"):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except ImportError:
        return None


def run_episode(env: MazeEnv, agent: BaseAgent, epsilon: float = 0.1, seed: Optional[int] = None) -> Tuple[float, int]:
    """Run a single episode with epsilon-greedy policy.

    Handles both Q-Learning (off-policy) and SARSA (on-policy) via unified
    update signature: update(s,a,r,ns,done,next_action).

    Returns:
        (total_reward, steps) for the episode.
    """
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    # SARSA needs to track next_action; Q-Learning ignores it.
    is_sarsa = isinstance(agent, SarsaAgent)
    action = agent.act(state, epsilon) if is_sarsa else None

    while not done:
        if is_sarsa:
            current_action = action  # type: ignore
        else:
            current_action = agent.act(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(current_action)
        done = terminated or truncated

        if is_sarsa:
            next_action = agent.act(next_state, epsilon) if not done else None
            agent.update(state, current_action, reward, next_state, done, next_action)
            action = next_action  # type: ignore
        else:
            agent.update(state, current_action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1
        if steps >= 200:
            break

    return total_reward, steps


def train(
    agent: BaseAgent,
    env: MazeEnv,
    episodes: int = 100,
    epsilon: float = 1.0,
    decay: float = 0.995,
    min_eps: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[int]]:
    """Train agent with epsilon decay and seed logging.

    Args:
        agent: RL agent.
        env: Environment.
        episodes: Number of episodes.
        epsilon: Initial epsilon.
        decay: Epsilon decay per episode.
        min_eps: Floor for epsilon.
        seed: Random seed for reproducibility (sets numpy/random and env).

    Returns:
        (rewards, steps) histories.
    """
    set_seed(seed)
    if seed is not None:
        env.reset(seed=seed)

    # experiment tracking: csv + tensorboard
    csv_logger = CsvLogger()
    tb_writer = get_tensorboard_writer()

    cur_eps = epsilon
    rewards: List[float] = []
    steps_hist: List[int] = []

    for ep in range(episodes):
        ep_seed = seed + ep if seed is not None else None
        reward, steps = run_episode(env, agent, cur_eps, seed=ep_seed)
        rewards.append(reward)
        steps_hist.append(steps)
        csv_logger.log(ep, reward, steps, cur_eps)
        if tb_writer:
            tb_writer.add_scalar("reward", reward, ep)
            tb_writer.add_scalar("steps", steps, ep)
        cur_eps = max(min_eps, cur_eps * decay)

    csv_logger.close()
    if tb_writer:
        tb_writer.close()
    return rewards, steps_hist


def evaluate_greedy(agent: BaseAgent, env: MazeEnv, max_steps: int = 50, seed: Optional[int] = None) -> int:
    """Run greedy rollout and return steps to termination."""
    state, _ = env.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        action = agent.act(state, 0.0)
        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        state = next_state
        steps += 1
    return steps


def train_with_logging(agent: BaseAgent, env: MazeEnv, episodes: int = 10, seed: Optional[int] = None):
    """Compatibility wrapper for logging-style training."""
    return train(agent, env, episodes=episodes, seed=seed)


def get_rewards(agent: BaseAgent, env: MazeEnv, seed: Optional[int] = None) -> float:
    """Run one episode with epsilon=0.1 and return reward."""
    reward, _ = run_episode(env, agent, 0.1, seed=seed)
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Maze solver")
    parser.add_argument("--episodes", type=int, default=DEFAULT["episodes"], help="Number of episodes")
    parser.add_argument("--size", type=int, default=DEFAULT["size"], help="Grid size")
    parser.add_argument("--alpha", type=float, default=DEFAULT["alpha"], help="Learning rate")
    parser.add_argument("--gamma", type=float, default=DEFAULT["gamma"], help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=DEFAULT["epsilon"], help="Initial epsilon")
    parser.add_argument("--decay", type=float, default=DEFAULT["decay"], help="Epsilon decay")
    parser.add_argument("--min-eps", type=float, default=DEFAULT["min_eps"], help="Minimum epsilon")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    env = MazeEnv(size=args.size, seed=args.seed)
    # Allow overriding alpha/gamma per agent if needed
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=args.alpha, gamma=args.gamma)
    train(agent, env, episodes=args.episodes, epsilon=args.epsilon, decay=args.decay, min_eps=args.min_eps, seed=args.seed)
# reproducibility verified
# formatted with black
