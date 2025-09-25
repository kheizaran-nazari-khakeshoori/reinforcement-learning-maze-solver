"""Agents module."""
import numpy as np
class BaseAgent:
    """Base."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.n_states=n_states; self.n_actions=n_actions
        self.alpha=alpha; self.gamma=gamma
        self.q_table=np.zeros((n_states,n_actions))
    def random_action(self):
        return np.random.randint(self.n_actions)
    def greedy_action(self, state):
        return int(np.argmax(self.q_table[state]))
    def act(self, state, epsilon=0.1):
        if np.random.rand()<epsilon:
            return self.random_action()
        return self.greedy_action(state)

class QLearningAgent(BaseAgent):
    """Q-Learning."""
    def update(self, s,a,r,ns,done):
        best=0.0 if done else float(np.max(self.q_table[ns]))
        self.q_table[s,a]+=self.alpha*(r+self.gamma*best-self.q_table[s,a])
