"""Plot."""
import matplotlib.pyplot as plt
def plot_rewards(rewards):
    plt.figure(); plt.plot(rewards); plt.savefig("curve.png")
