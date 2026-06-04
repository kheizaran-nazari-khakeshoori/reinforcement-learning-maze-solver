"""Plotting helpers for training curves."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def moving_average(data: List[float], window: int = 10) -> np.ndarray:
    """Simple moving average with convolution."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_rewards(rewards: List[float], save: str = "curve.png") -> None:
    """Plot raw rewards and moving average."""
    plt.figure()
    plt.plot(rewards, alpha=0.3, label="raw")
    ma = moving_average(rewards, 10)
    plt.plot(range(9, len(rewards)), ma, label="ma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def plot_comparison(q_rewards: List[float], s_rewards: List[float], save: str = "compare.png") -> None:
    plt.figure()
    plt.plot(q_rewards, label="Q-Learning")
    plt.plot(s_rewards, label="SARSA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def save_curve(rewards: List[float], path: str = "tmp.png") -> None:
    plot_rewards(rewards, save=path)


def plot_q_heatmap(agent, size: int = 5, save: str = "q_heatmap.png") -> None:
    """Visualize max Q-value per state as heatmap."""
    import numpy as np
    q_max = agent.q_table.max(axis=1).reshape(size, size)
    plt.figure()
    plt.imshow(q_max, cmap="viridis")
    plt.colorbar(label="max Q")
    plt.title("Q-table heatmap (max over actions)")
    for r in range(size):
        for c in range(size):
            plt.text(c, r, f"{q_max[r,c]:.2f}", ha="center", va="center", color="white", fontsize=6)
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
