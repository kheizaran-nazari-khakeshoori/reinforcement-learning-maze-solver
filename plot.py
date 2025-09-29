"""Plot."""
import numpy as np
import matplotlib.pyplot as plt
def moving_average(data, w=10):
    return np.convolve(data, np.ones(w)/w, mode="valid")
def plot_rewards(rewards):
    plt.figure(); plt.plot(rewards, alpha=0.3)
    ma=moving_average(rewards, 10)
    plt.plot(range(9, len(rewards)), ma)
    plt.savefig("curve.png")
