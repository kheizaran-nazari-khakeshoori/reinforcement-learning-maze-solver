"""Agents module."""
import numpy as np
class BaseAgent:
    """Base."""
    def __init__(self, n_states, n_actions):
        self.n_states=n_states; self.n_actions=n_actions
        self.q_table=np.zeros((n_states,n_actions))
