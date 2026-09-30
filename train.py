"""Training loop for the tabular Q-learning smart-grid agent."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent import QLearningAgent
from config import Config, get_config
from data_loader import load_or_generate_data, split_train_test
from smart_grid_env import SmartGridEnv


def moving_average(values: List[float], window: int) -> np.ndarray:
    """Compute a trailing simple moving average.

    Args:
        values: Sequence of scalar values.
        window: Averaging window length.

    Returns:
        Array of the same length as ``values``; early entries average over
        the available prefix.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        out[i] = arr[lo:i + 1].mean()
    return out


def train(config: Optional[Config] = None,
          artifact_dir: str = "artifacts",
          verbose: bool = True
          ) -> Tuple[QLearningAgent, Dict[str, List[float]]]:
    """Train a Q-learning agent for the configured number of episodes.

    Args:
        config: Project configuration (defaults to global singleton).
        artifact_dir: Directory where ``q_table.npy`` and logs are written.
        verbose: If True, print milestone progress.

    Returns:
        ``(trained_agent, history)`` where history contains per-episode
        rewards, moving averages, epsilon values and step penalties.
    """
    cfg = config if config is not None else get_config()
    os.makedirs(artifact_dir, exist_ok=True)

    load_kw, pv_kw = load_or_generate_data()
    train_load, train_pv, _, _ = split_train_test(load_kw, pv_kw, test_days=1)

    env = SmartGridEnv(load_kw=train_load, pv_kw=train_pv, config=cfg,
                       episode_days=1, random_start=True, seed=cfg.train.seed)
    agent = QLearningAgent(cfg.disc.num_states, cfg.num_actions,
                           config=cfg, seed=cfg.train.seed)

    history: Dict[str, List[float]] = {
        "episode": [], "reward": [], "epsilon": [], "penalty": [],
    }

    for ep in range(cfg.train.num_episodes):
        obs, _ = env.reset()
        state = env.discretize(obs)
        total_reward = 0.0
        total_penalty = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.discretize(next_obs)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            total_penalty += info["degradation"] + info["soc_penalty"]

        agent.decay_epsilon(ep)

        history["episode"].append(ep + 1)
        history["reward"].append(total_reward)
        history["epsilon"].append(agent.epsilon)
        history["penalty"].append(total_penalty)

        if verbose and (ep + 1) in cfg.train.milestone_episodes:
            ma = moving_average(history["reward"],
                                cfg.train.moving_average_window)[-1]
            print(f"[train] Episode {ep + 1:>3d} | "
                  f"Epsilon {agent.epsilon:5.3f} | "
                  f"Raw Reward {total_reward:9.3f} | "
                  f"20-Ep MA {ma:9.3f}")

    history["moving_avg"] = moving_average(
        history["reward"], cfg.train.moving_average_window).tolist()

    q_path = os.path.join(artifact_dir, "q_table.npy")
    agent.save(q_path)
    np.save(os.path.join(artifact_dir, "history_reward.npy"),
            np.asarray(history["reward"]))
    if verbose:
        print(f"[train] Training complete. Q-table saved to '{q_path}'.")

    return agent, history


if __name__ == "__main__":
    train()
