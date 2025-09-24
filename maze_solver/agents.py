"""Agents module."""
import numpy as np
class BaseAgent:
    """Base."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states=n_states; self.n_actions=n_actions
        self.alpha=alpha; self.gamma=gamma
        self.q_table=np.zeros((n_states,n_actions))
