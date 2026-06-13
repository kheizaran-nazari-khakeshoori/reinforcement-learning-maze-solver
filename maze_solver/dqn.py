"""DQN agent for large mazes N>=10."""

import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    def push(self, s, a, r, ns, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s,a,r,ns,done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

if HAS_TORCH:
    class QNetwork(nn.Module):
        def __init__(self, n_states, n_actions, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Embedding(n_states, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
        def forward(self, x):
            return self.net(x)

class DQNAgent:
    """Simple DQN for N>=10, falls back to tabular if torch missing."""
    def __init__(self, n_states, n_actions, alpha=1e-3, gamma=0.99, epsilon=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer = ReplayBuffer()
        if HAS_TORCH:
            self.q_net = QNetwork(n_states, n_actions)
            self.target_net = QNetwork(n_states, n_actions)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=alpha)
        else:
            # fallback tabular
            self.q_table = np.zeros((n_states, n_actions))
            self.alpha = alpha

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions-1)
        if HAS_TORCH:
            import torch
            with torch.no_grad():
                q = self.q_net(torch.tensor([state]))
                return int(q.argmax().item())
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, s, a, r, ns, done, *args, **kwargs):
        self.buffer.push(s,a,r,ns,done)
        if HAS_TORCH and len(self.buffer) >= 32:
            import torch
            batch = self.buffer.sample(32)
            s_b, a_b, r_b, ns_b, d_b = zip(*batch)
            s_t = torch.tensor(s_b)
            a_t = torch.tensor(a_b).unsqueeze(1)
            r_t = torch.tensor(r_b, dtype=torch.float32)
            ns_t = torch.tensor(ns_b)
            d_t = torch.tensor(d_b, dtype=torch.float32)
            q = self.q_net(s_t).gather(1, a_t).squeeze()
            with torch.no_grad():
                q_next = self.target_net(ns_t).max(1)[0]
                target = r_t + self.gamma * q_next * (1 - d_t)
            loss = (q - target).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif not HAS_TORCH:
            # tabular fallback
            best = 0.0 if done else float(np.max(self.q_table[ns]))
            self.q_table[s,a] += self.alpha * (r + self.gamma*best - self.q_table[s,a])

# network supports N>=10 with embedding

# use for size>=10, tabular for size<=7
# debuging dqn verified
