"""Tabular agents: Q-Learning and SARSA with unified update interface."""

from typing import Optional

import numpy as np


class BaseAgent:
    """Base tabular agent with epsilon-greedy policy and type hints."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))

    def random_action(self) -> int:
        return int(np.random.randint(self.n_actions))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))

    def act(self, state: int, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return self.random_action()
        return self.greedy_action(state)

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path)

    def reset(self) -> None:
        self.q_table *= 0

    def best_action(self, state: int) -> int:
        return self.greedy_action(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(states={self.n_states}, actions={self.n_actions})"

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """Unified update interface with *args for backwards compat.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Episode terminated.
            next_action: Next action (used by SARSA, ignored by Q-Learning).
        """
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    """Q-Learning: off-policy TD update r + gamma * max_a Q(s',a)."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        # Backward compatibility: old shim called update(s,a,r,ns,a,done) with swapped order.
        # Detect swapped args where done is int and next_action is bool.
        if not isinstance(done, (bool, np.bool_)) and isinstance(next_action, (bool, np.bool_)):
            done, next_action = bool(next_action), int(done)  # type: ignore
        best = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


class SarsaAgent(BaseAgent):
    """SARSA: on-policy TD update r + gamma * Q(s',a')."""

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        # Backward compatibility: old code used update(s,a,r,ns,na,done) (na before done).
        # New unified interface is update(s,a,r,ns,done,na). Support both.
        if not isinstance(done, (bool, np.bool_)) and isinstance(next_action, (bool, np.bool_)):
            done, next_action = bool(next_action), int(done)  # type: ignore
        if done:
            nxt = 0.0
        else:
            if next_action is None:
                raise ValueError("SARSA requires next_action when not done")
            nxt = float(self.q_table[next_state, next_action])
        td_target = reward + self.gamma * nxt
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

# type hints verified
