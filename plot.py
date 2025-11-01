"""Plot."""
import numpy as np
import matplotlib.pyplot as plt
def moving_average(data, w=10):
    return np.convolve(data, np.ones(w)/w, mode="valid")
def plot_rewards(rewards, save="curve.png"):
    plt.figure(); plt.plot(rewards, alpha=0.3, label="raw")
    ma=moving_average(rewards, 10)
    plt.plot(range(9, len(rewards)), ma, label="ma")
    plt.legend(); plt.tight_layout(); plt.savefig(save)

def plot_comparison(q,s):
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(q, label="q"); plt.plot(s, label="s"); plt.legend()

def save_curve(rewards, path="tmp.png"):
    plot_rewards(rewards, save=path)
def plot_06(): pass
def plot_14(): pass
def plot_22(): pass
def plot_30(): pass
def plot_38(): pass
