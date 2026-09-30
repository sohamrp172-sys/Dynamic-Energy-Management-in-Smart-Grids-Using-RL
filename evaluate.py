"""Evaluation, baselines, plotting and quantitative comparison.

Compares three strategies on a held-out 24-hour test profile:
    1. Random agent
    2. Rule-based greedy self-consumption heuristic
    3. Trained tabular Q-learning agent

Generates three publication-ready figures (dpi=300) and prints a Markdown
comparison table.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless-safe backend.
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False

from agent import QLearningAgent
from config import Config, get_config
from data_loader import load_or_generate_data, split_train_test
from smart_grid_env import SmartGridEnv


# --------------------------------------------------------------------------- #
# Baseline policies
# --------------------------------------------------------------------------- #
def random_policy_factory(env: SmartGridEnv, seed: int = 0
                          ) -> Callable[[np.ndarray], int]:
    """Return a uniform-random action selector."""
    rng = np.random.default_rng(seed)

    def policy(_obs: np.ndarray) -> int:
        return int(rng.integers(0, env.action_space.n))

    return policy


def rule_based_policy_factory(env: SmartGridEnv
                              ) -> Callable[[np.ndarray], int]:
    """Return a greedy self-consumption + peak-arbitrage heuristic.

    Logic:
        * Surplus solar (PV > Load): charge the battery.
        * Deficit (Load > PV): discharge to cover the shortfall.
        * During peak hours (18:00-22:00): prioritize discharging.
    Action indices: 0 fast-charge, 1 slow-charge, 2 idle,
                    3 slow-discharge, 4 fast-discharge.
    """
    cfg = env.cfg

    def policy(obs: np.ndarray) -> int:
        p_load, p_pv, soc, tariff = obs[0], obs[1], obs[2], obs[3]
        net = p_load - p_pv  # >0 deficit, <0 surplus
        is_peak = abs(tariff - cfg.tariff.peak_rate) < 1e-6

        if is_peak and soc > cfg.battery.soc_safe_min:
            # Economic arbitrage: discharge stored energy during peak.
            return 4 if net > 1.5 else 3
        if net < -0.1 and soc < cfg.battery.soc_safe_max:
            # Surplus solar -> charge.
            return 0 if net < -1.5 else 1
        if net > 0.1 and soc > cfg.battery.soc_safe_min:
            # Deficit -> discharge to self-consume.
            return 4 if net > 1.5 else 3
        return 2  # Idle.

    return policy


def q_policy_factory(env: SmartGridEnv, agent: QLearningAgent
                     ) -> Callable[[np.ndarray], int]:
    """Return a greedy policy backed by a trained Q-table."""
    def policy(obs: np.ndarray) -> int:
        state = env.discretize(obs)
        return agent.select_action(state, greedy=True)

    return policy


# --------------------------------------------------------------------------- #
# Rollout & metrics
# --------------------------------------------------------------------------- #
def run_episode(env: SmartGridEnv, policy: Callable[[np.ndarray], int]
                ) -> Dict[str, np.ndarray]:
    """Run one full episode under ``policy`` and collect per-step traces.

    Returns:
        Dict of numpy arrays keyed by trace name (hour, p_load, p_pv, ...).
    """
    obs, _ = env.reset()
    traces: Dict[str, List[float]] = {
        "hour": [], "p_load": [], "p_pv": [], "p_batt": [], "p_grid": [],
        "soc": [], "tariff": [], "cost": [], "reward": [], "violation": [],
    }
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        traces["hour"].append(info["hour"])
        traces["p_load"].append(info["p_load"])
        traces["p_pv"].append(info["p_pv"])
        traces["p_batt"].append(info["p_batt"])
        traces["p_grid"].append(info["p_grid"])
        traces["soc"].append(info["soc"])
        traces["tariff"].append(info["tariff"])
        traces["cost"].append(info["cost"])
        traces["reward"].append(reward)
        traces["violation"].append(1.0 if info["soc_violation"] else 0.0)
    return {k: np.asarray(v, dtype=np.float64) for k, v in traces.items()}


def compute_metrics(traces: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary economic and technical metrics for an episode.

    Returns:
        Dict with total bill (Rs), peak grid draw (kW), solar
        self-consumption (%) and SoC boundary-violation count.
    """
    total_bill = float(traces["cost"].sum())
    peak_grid_draw = float(max(0.0, traces["p_grid"].max()))

    # Solar self-consumption: fraction of generated PV energy that is used
    # locally (by load or to charge the battery) rather than exported.
    # Per step, PV-attributed export cannot exceed the PV generated that step;
    # battery-sourced export is excluded from the PV export figure.
    pv_step = traces["p_pv"]
    export_step = np.clip(-traces["p_grid"], 0.0, None)
    pv_export_step = np.minimum(export_step, pv_step)  # PV-attributed export
    pv_total = float(pv_step.sum())
    pv_exported = float(pv_export_step.sum())
    self_consumed = max(0.0, pv_total - pv_exported)
    self_consumption_rate = (100.0 * self_consumed / pv_total
                             if pv_total > 1e-9 else 100.0)
    self_consumption_rate = float(min(100.0, self_consumption_rate))

    violations = int(traces["violation"].sum())

    return {
        "total_bill": total_bill,
        "peak_grid_draw": peak_grid_draw,
        "self_consumption": self_consumption_rate,
        "violations": violations,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_convergence(history: Dict[str, List[float]], out_path: str,
                     window: int = 20) -> None:
    """Figure 1: training reward curve + moving average."""
    rewards = np.asarray(history["reward"], dtype=np.float64)
    episodes = np.arange(1, rewards.size + 1)
    ma = np.asarray(history.get("moving_avg"), dtype=np.float64) \
        if history.get("moving_avg") is not None else rewards

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, rewards, color="#9ecae1", alpha=0.7, linewidth=1.0,
            label="Raw Episodic Reward")
    ax.plot(episodes, ma, color="#08519c", linewidth=2.2,
            label=f"{window}-Episode Moving Average")
    ax.set_title("Training Convergence: Tabular Q-Learning Agent",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episodic Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_power_dispatch(traces: Dict[str, np.ndarray], out_path: str) -> None:
    """Figure 2: 24-hour power dispatch (load, PV, battery, grid)."""
    hours = np.arange(traces["p_load"].size) * 0.5
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hours, traces["p_load"], label="P_load (kW)", color="#d62728",
            linewidth=2.0)
    ax.plot(hours, traces["p_pv"], label="P_pv (kW)", color="#ff7f0e",
            linewidth=2.0)
    ax.plot(hours, traces["p_batt"], label="P_batt (kW)  (+dis / -chg)",
            color="#2ca02c", linewidth=2.0)
    ax.plot(hours, traces["p_grid"], label="P_grid (kW)  (+imp / -exp)",
            color="#1f77b4", linewidth=2.0, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("24-Hour Power Dispatch — Trained RL Agent",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Time of Day (hours)")
    ax.set_ylabel("Power (kW)")
    ax.set_xlim(0, max(24, hours.max()))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_soc_tariff(traces: Dict[str, np.ndarray], out_path: str) -> None:
    """Figure 3: dual-axis SoC vs Time-of-Day tariff (arbitrage)."""
    hours = np.arange(traces["soc"].size) * 0.5
    fig, ax1 = plt.subplots(figsize=(11, 6))

    color_soc = "#08519c"
    ax1.plot(hours, traces["soc"] * 100.0, color=color_soc, linewidth=2.4,
             label="Battery SoC (%)")
    ax1.axhline(20.0, color="grey", linestyle=":", linewidth=1.2,
                label="Safe SoC bounds")
    ax1.axhline(80.0, color="grey", linestyle=":", linewidth=1.2)
    ax1.set_xlabel("Time of Day (hours)")
    ax1.set_ylabel("Battery State-of-Charge (%)", color=color_soc)
    ax1.tick_params(axis="y", labelcolor=color_soc)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(24, hours.max()))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_tar = "#d62728"
    ax2.step(hours, traces["tariff"], where="post", color=color_tar,
             linewidth=2.0, label="ToD Tariff (Rs/kWh)")
    ax2.set_ylabel("Time-of-Day Tariff (Rs/kWh)", color=color_tar)
    ax2.tick_params(axis="y", labelcolor=color_tar)
    ax2.set_ylim(0, max(14.0, float(traces["tariff"].max()) + 2.0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax1.set_title("Battery SoC vs Time-of-Day Tariff — Peak Shaving & "
                  "Economic Arbitrage", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def build_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Render a Markdown comparison table from per-strategy metrics."""
    header = ("| Strategy | Total Energy Bill (Rs) | Peak Grid Draw (kW) "
              "| Solar Self-Consumption (%) | SoC Boundary Violations |\n"
              "|---|---:|---:|---:|---:|")
    rows = [header]
    for name, m in results.items():
        rows.append(
            f"| {name} | {m['total_bill']:.2f} | {m['peak_grid_draw']:.2f} "
            f"| {m['self_consumption']:.1f} | {m['violations']} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(history: Optional[Dict[str, List[float]]] = None,
             config: Optional[Config] = None,
             artifact_dir: str = "artifacts",
             figures_dir: str = "figures") -> Dict[str, Dict[str, float]]:
    """Run full evaluation: baselines, RL agent, plots and comparison table.

    Args:
        history: Optional training history for the convergence plot. If None,
            an attempt is made to load ``history_reward.npy``.
        config: Project configuration.
        artifact_dir: Directory containing ``q_table.npy``.
        figures_dir: Directory to which figures are written.

    Returns:
        Mapping of strategy name to its computed metrics.
    """
    cfg = config if config is not None else get_config()
    os.makedirs(figures_dir, exist_ok=True)

    load_kw, pv_kw = load_or_generate_data()
    _, _, test_load, test_pv = split_train_test(load_kw, pv_kw, test_days=1)

    def make_env() -> SmartGridEnv:
        return SmartGridEnv(load_kw=test_load, pv_kw=test_pv, config=cfg,
                            episode_days=1, random_start=False,
                            seed=cfg.train.seed)

    # --- Trained agent ---
    agent = QLearningAgent(cfg.disc.num_states, cfg.num_actions, config=cfg,
                           seed=cfg.train.seed)
    q_path = os.path.join(artifact_dir, "q_table.npy")
    if os.path.isfile(q_path):
        agent.load(q_path)
    else:
        print(f"[evaluate] Warning: '{q_path}' not found; using untrained "
              f"agent. Run train.py first.")

    strategies: Dict[str, Callable[[SmartGridEnv], Callable]] = {
        "Baseline (Random)": lambda e: random_policy_factory(e,
                                                             seed=cfg.train.seed),
        "Rule-Based Greedy": rule_based_policy_factory,
        "Trained RL Agent": lambda e: q_policy_factory(e, agent),
    }

    results: Dict[str, Dict[str, float]] = {}
    rl_traces: Optional[Dict[str, np.ndarray]] = None
    for name, factory in strategies.items():
        env = make_env()
        policy = factory(env)
        traces = run_episode(env, policy)
        results[name] = compute_metrics(traces)
        if name == "Trained RL Agent":
            rl_traces = traces

    # --- Figures ---
    if history is None:
        hist_path = os.path.join(artifact_dir, "history_reward.npy")
        if os.path.isfile(hist_path):
            rewards = np.load(hist_path).tolist()
            history = {"reward": rewards, "moving_avg": None}
    if history is not None:
        if history.get("moving_avg") is None:
            from train import moving_average
            history["moving_avg"] = moving_average(
                history["reward"], cfg.train.moving_average_window).tolist()
        plot_convergence(history,
                         os.path.join(figures_dir, "figure_1_convergence.png"),
                         window=cfg.train.moving_average_window)

    if rl_traces is not None:
        plot_power_dispatch(
            rl_traces, os.path.join(figures_dir, "figure_2_power_dispatch.png"))
        plot_soc_tariff(
            rl_traces,
            os.path.join(figures_dir, "figure_3_soc_tariff_arbitrage.png"))

    # --- Table ---
    table = build_comparison_table(results)
    print("\n=== Quantitative Strategy Comparison "
          "(held-out 24-hour profile) ===\n")
    print(table)
    print()
    return results


if __name__ == "__main__":
    evaluate()
