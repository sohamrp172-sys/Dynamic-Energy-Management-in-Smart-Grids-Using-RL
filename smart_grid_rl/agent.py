"""Tabular Q-Learning agent for the Smart Grid environment."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from config import Config, get_config


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent.

    Maintains a Q-table of shape ``[num_states, num_actions]`` and updates it
    via the temporal-difference Bellman equation.
    """

    def __init__(self, num_states: int, num_actions: int,
                 config: Optional[Config] = None,
                 seed: Optional[int] = None) -> None:
        """Initialize the agent and its Q-table.

        Args:
            num_states: Number of discrete states.
            num_actions: Number of discrete actions.
            config: Project configuration (defaults to global singleton).
            seed: Optional RNG seed for reproducible exploration.
        """
        self.cfg: Config = config if config is not None else get_config()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions),
                                dtype=np.float64)

        self.alpha = self.cfg.train.alpha
        self.gamma = self.cfg.train.gamma
        self.epsilon = self.cfg.train.epsilon_start
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Select an action using an epsilon-greedy policy.

        Args:
            state: Discrete state index.
            greedy: If True, always exploit (used for evaluation).

        Returns:
            Chosen action index.
        """
        state = int(np.clip(state, 0, self.num_states - 1))
        if (not greedy) and (self._rng.random() < self.epsilon):
            return int(self._rng.integers(0, self.num_actions))
        q_values = self.q_table[state]
        # Break ties randomly among the best actions.
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """Apply the Q-learning TD update.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current discrete state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting discrete state.
            done: Whether the episode terminated (no bootstrap if True).
        """
        state = int(np.clip(state, 0, self.num_states - 1))
        next_state = int(np.clip(next_state, 0, self.num_states - 1))
        action = int(np.clip(action, 0, self.num_actions - 1))

        best_next = 0.0 if done else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self, episode: int) -> None:
        """Linearly decay epsilon from start to end over the schedule.

        Args:
            episode: Zero-based episode index.
        """
        t = self.cfg.train
        frac = min(1.0, episode / max(1, t.epsilon_decay_episodes))
        self.epsilon = t.epsilon_start + frac * (t.epsilon_end -
                                                 t.epsilon_start)
        self.epsilon = float(max(t.epsilon_end, self.epsilon))

    def save(self, path: str = "q_table.npy") -> None:
        """Persist the Q-table to disk as a ``.npy`` file."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str = "q_table.npy") -> None:
        """Load a Q-table from disk, validating its shape."""
        table = np.load(path)
        if table.shape != (self.num_states, self.num_actions):
            raise ValueError(
                f"Loaded Q-table shape {table.shape} does not match "
                f"expected {(self.num_states, self.num_actions)}.")
        self.q_table = table.astype(np.float64)
