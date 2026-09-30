"""Gymnasium environment for microgrid battery dispatch optimization.

Implements ``SmartGridEnv`` following the Farama Gymnasium API together with a
state-discretization helper that maps the continuous observation into one of
27 discrete states for tabular Q-learning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # Gymnasium is the primary dependency.
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback if gymnasium missing at import
    gym = None
    spaces = None

from config import Config, get_config
from data_loader import load_or_generate_data


class SmartGridEnv(gym.Env if gym is not None else object):
    """A microgrid battery-dispatch environment.

    Observation (continuous Box, shape (6,)):
        [P_load (kW), P_pv (kW), SoC (0..1), Tariff (Rs/kWh),
         sin(time_of_day), cos(time_of_day)]

    Action (Discrete(5)):
        0: Fast Charge   (-3.0 kW)
        1: Slow Charge   (-1.5 kW)
        2: Idle          ( 0.0 kW)
        3: Slow Discharge(+1.5 kW)
        4: Fast Discharge(+3.0 kW)

    Reward:
        R = -(Cost_grid + alpha * Degradation + beta * SoC_violation)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 load_kw: Optional[np.ndarray] = None,
                 pv_kw: Optional[np.ndarray] = None,
                 config: Optional[Config] = None,
                 episode_days: int = 1,
                 random_start: bool = True,
                 seed: Optional[int] = None) -> None:
        """Initialize the environment.

        Args:
            load_kw: Household load series (kW). Auto-loaded if ``None``.
            pv_kw: PV generation series (kW). Auto-loaded if ``None``.
            config: Project configuration; defaults to the global singleton.
            episode_days: Number of days per episode.
            random_start: If True, each episode starts at a random day offset.
            seed: Optional RNG seed.
        """
        super().__init__()
        self.cfg: Config = config if config is not None else get_config()

        if load_kw is None or pv_kw is None:
            load_kw, pv_kw = load_or_generate_data()
        self.load_kw = np.asarray(load_kw, dtype=np.float64)
        self.pv_kw = np.asarray(pv_kw, dtype=np.float64)

        self.steps_per_day = self.cfg.env.steps_per_day
        self.episode_len = episode_days * self.steps_per_day
        self.random_start = random_start
        self._rng = np.random.default_rng(seed)

        self.actions_kw = np.asarray(self.cfg.env.battery_actions_kw,
                                     dtype=np.float64)

        # --- Gymnasium spaces ---
        high = np.array([np.inf, np.inf, 1.0, np.inf, 1.0, 1.0],
                        dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,
                                             dtype=np.float32)
        self.action_space = spaces.Discrete(self.cfg.num_actions)

        # --- Episode state ---
        self._start_idx: int = 0
        self._t: int = 0
        self.soc: float = self.cfg.battery.soc_init

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #
    def _hour_of_day(self, step_in_day: int) -> float:
        """Return the hour of day (0..24) for a within-day step index."""
        return (step_in_day * self.cfg.env.dt_hours) % 24.0

    def _time_features(self, hour: float) -> Tuple[float, float]:
        """Cyclical (sin, cos) encoding of the hour of day."""
        angle = 2.0 * math.pi * (hour / 24.0)
        return math.sin(angle), math.cos(angle)

    def _current_index(self) -> int:
        """Absolute index into the data arrays for the current step."""
        return (self._start_idx + self._t) % self.load_kw.size

    def _get_obs(self) -> np.ndarray:
        """Build the continuous observation vector for the current step."""
        idx = self._current_index()
        step_in_day = (self._start_idx + self._t) % self.steps_per_day
        hour = self._hour_of_day(step_in_day)
        p_load = float(self.load_kw[idx])
        p_pv = float(self.pv_kw[idx])
        tariff = self.cfg.tariff.buy_rate(hour)
        sin_t, cos_t = self._time_features(hour)
        return np.array([p_load, p_pv, self.soc, tariff, sin_t, cos_t],
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the start of a (random) episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)

        num_days = max(1, self.load_kw.size // self.steps_per_day)
        if self.random_start and num_days > 1:
            day = int(self._rng.integers(0, num_days))
            self._start_idx = (day * self.steps_per_day) % self.load_kw.size
        else:
            self._start_idx = 0

        self._t = 0
        self.soc = self.cfg.battery.soc_init
        return self._get_obs(), {}

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the simulation by one 30-minute step.

        Args:
            action: Discrete action index in ``[0, num_actions)``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        action = int(action)
        action = max(0, min(action, self.cfg.num_actions - 1))

        batt = self.cfg.battery
        dt = self.cfg.env.dt_hours
        idx = self._current_index()
        step_in_day = (self._start_idx + self._t) % self.steps_per_day
        hour = self._hour_of_day(step_in_day)

        p_load = float(self.load_kw[idx])
        p_pv = float(self.pv_kw[idx])

        # Requested battery power (kW): negative charges, positive discharges.
        p_batt_req = float(self.actions_kw[action])

        # --- Apply battery physics with capacity & rate limits ---
        p_batt, soc_next = self._apply_battery(p_batt_req, dt)

        # --- Grid balance ---
        p_grid = p_load - p_pv - p_batt  # >0 import, <0 export

        # --- Cost / revenue over the interval ---
        if p_grid >= 0.0:
            cost = p_grid * self.cfg.tariff.buy_rate(hour) * dt
        else:
            cost = p_grid * self.cfg.tariff.feed_in_rate * dt  # negative

        # --- Degradation penalty (penalize high-rate cycling) ---
        degradation = self.cfg.env.alpha_degradation * (p_batt ** 2)

        # --- SoC boundary violation penalty (quadratic) ---
        violation = 0.0
        if soc_next < batt.soc_safe_min:
            violation = (batt.soc_safe_min - soc_next) ** 2
        elif soc_next > batt.soc_safe_max:
            violation = (soc_next - batt.soc_safe_max) ** 2
        soc_penalty = self.cfg.env.beta_soc_violation * violation
        is_violation = violation > 0.0

        reward = -(cost + degradation + soc_penalty)

        # Commit state transition.
        self.soc = soc_next
        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_len

        info = {
            "p_load": p_load,
            "p_pv": p_pv,
            "p_batt": p_batt,
            "p_grid": p_grid,
            "soc": self.soc,
            "hour": hour,
            "tariff": self.cfg.tariff.buy_rate(hour),
            "cost": cost,
            "degradation": degradation,
            "soc_penalty": soc_penalty,
            "soc_violation": bool(is_violation),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _apply_battery(self, p_batt_req: float, dt: float
                       ) -> Tuple[float, float]:
        """Apply rate/capacity limits and efficiency to a battery request.

        Args:
            p_batt_req: Requested battery power (kW). Negative = charge.
            dt: Time-step length (hours).

        Returns:
            ``(p_batt_actual, soc_next)`` where power is clipped to feasible
            limits and ``soc_next`` is bounded to [hard_min, hard_max].
        """
        batt = self.cfg.battery
        cap = batt.capacity_kwh

        # A realistic Battery Management System (BMS) confines operation to
        # the safe SoC band. We use it as the operational limit so charge /
        # discharge power is curtailed as the band edges are approached.
        soc_ceiling = batt.soc_safe_max
        soc_floor = batt.soc_safe_min

        if p_batt_req < 0.0:  # Charging
            p = max(p_batt_req, -batt.max_charge_kw)
            # Energy actually stored respects charge efficiency.
            energy_in = (-p) * dt * batt.efficiency
            headroom = (soc_ceiling - self.soc) * cap
            energy_in = min(energy_in, max(headroom, 0.0))
            soc_next = self.soc + energy_in / cap
            # Recompute realized power from clipped energy.
            realized = -(energy_in / (dt * batt.efficiency)) if dt > 0 else 0.0
            return realized, float(np.clip(soc_next, batt.soc_hard_min,
                                           batt.soc_hard_max))

        if p_batt_req > 0.0:  # Discharging
            p = min(p_batt_req, batt.max_discharge_kw)
            # Energy drawn from cells to deliver p at the terminal.
            energy_out = (p * dt) / batt.efficiency
            available = max((self.soc - soc_floor) * cap, 0.0)
            energy_out = min(energy_out, available)
            soc_next = self.soc - energy_out / cap
            realized = (energy_out * batt.efficiency) / dt if dt > 0 else 0.0
            return realized, float(np.clip(soc_next, batt.soc_hard_min,
                                           batt.soc_hard_max))

        return 0.0, self.soc  # Idle

    def render(self) -> None:
        """Print a one-line human-readable summary of the current step."""
        obs = self._get_obs()
        print(f"t={self._t:02d} load={obs[0]:.2f} pv={obs[1]:.2f} "
              f"soc={obs[2]:.2f} tariff={obs[3]:.2f}")

    # ------------------------------------------------------------------ #
    # Discretization for tabular Q-learning
    # ------------------------------------------------------------------ #
    def discretize(self, obs: np.ndarray) -> int:
        """Map a continuous observation to a discrete state index in [0, 27).

        Bins:
            Net demand (Load - PV): 0=Surplus, 1=Balanced, 2=Deficit
            SoC:                    0=Low(<0.25), 1=Mid, 2=High(>0.75)
            Tariff:                 0=Off-Peak, 1=Normal, 2=Peak

        Args:
            obs: Continuous observation vector of shape (6,).

        Returns:
            Discrete state index in ``[0, num_states)``.
        """
        return discretize_observation(obs, self.cfg)


def discretize_observation(obs: np.ndarray, config: Optional[Config] = None
                           ) -> int:
    """Standalone discretization function (also used by wrappers/baselines).

    Args:
        obs: Continuous observation [load, pv, soc, tariff, sin_t, cos_t].
        config: Project configuration (defaults to global singleton).

    Returns:
        Discrete state index in ``[0, 27)`` with no possibility of IndexError.
    """
    cfg = config if config is not None else get_config()
    disc = cfg.disc

    p_load = float(obs[0])
    p_pv = float(obs[1])
    soc = float(obs[2])
    tariff = float(obs[3])

    # --- Net demand bin ---
    net = p_load - p_pv
    if net < disc.net_demand_low:
        net_bin = 0  # Surplus (PV exceeds load)
    elif net <= disc.net_demand_high:
        net_bin = 1  # Balanced
    else:
        net_bin = 2  # Deficit
    net_bin = int(np.clip(net_bin, 0, disc.net_demand_bins - 1))

    # --- SoC bin ---
    if soc < disc.soc_low:
        soc_bin = 0  # Low
    elif soc <= disc.soc_high:
        soc_bin = 1  # Mid
    else:
        soc_bin = 2  # High
    soc_bin = int(np.clip(soc_bin, 0, disc.soc_bins - 1))

    # --- Tariff bin (map rate back to Off-Peak/Normal/Peak) ---
    if math.isclose(tariff, cfg.tariff.off_peak_rate, abs_tol=1e-6):
        tariff_bin = 0
    elif math.isclose(tariff, cfg.tariff.peak_rate, abs_tol=1e-6):
        tariff_bin = 2
    else:
        tariff_bin = 1
    tariff_bin = int(np.clip(tariff_bin, 0, disc.tariff_bins - 1))

    state = (net_bin * disc.soc_bins * disc.tariff_bins
             + soc_bin * disc.tariff_bins
             + tariff_bin)
    return int(np.clip(state, 0, disc.num_states - 1))


if gym is not None:
    class DiscretizeObservation(gym.ObservationWrapper):
        """Gymnasium ObservationWrapper returning discrete state indices."""

        def __init__(self, env: SmartGridEnv) -> None:
            super().__init__(env)
            self.cfg = env.cfg
            self.observation_space = spaces.Discrete(self.cfg.disc.num_states)

        def observation(self, observation: np.ndarray) -> int:
            """Discretize a continuous observation."""
            return discretize_observation(observation, self.cfg)
