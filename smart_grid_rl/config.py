"""Central configuration for the Smart Grid RL project.

All physical constants, tariff structures, environment parameters and
training hyperparameters are defined here as immutable dataclasses so that
every module shares a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BatteryConfig:
    """Physical specification of the home battery storage system."""

    capacity_kwh: float = 10.0          # Total usable energy capacity (kWh)
    max_charge_kw: float = 3.0          # Maximum charge power (kW)
    max_discharge_kw: float = 3.0       # Maximum discharge power (kW)
    efficiency: float = 0.95            # Round-trip charge/discharge efficiency
    soc_safe_min: float = 0.20          # Lower safe State-of-Charge bound
    soc_safe_max: float = 0.80          # Upper safe State-of-Charge bound
    soc_init: float = 0.50              # Initial State-of-Charge at reset
    soc_hard_min: float = 0.0           # Absolute minimum SoC (physical limit)
    soc_hard_max: float = 1.0           # Absolute maximum SoC (physical limit)


@dataclass(frozen=True)
class TariffConfig:
    """Time-of-Day (ToD) Indian tariff structure in Rupees per kWh."""

    off_peak_rate: float = 4.00         # 22:00 - 06:00
    normal_rate: float = 7.50           # 06:00 - 18:00
    peak_rate: float = 12.00            # 18:00 - 22:00
    feed_in_rate: float = 4.50          # Flat export (sell) tariff

    def buy_rate(self, hour: float) -> float:
        """Return the applicable purchase tariff (Rs/kWh) for a given hour.

        Args:
            hour: Hour of day in the range [0, 24).

        Returns:
            The buy tariff in Rs/kWh for the tariff window containing ``hour``.
        """
        h = hour % 24.0
        if 22.0 <= h or h < 6.0:
            return self.off_peak_rate
        if 18.0 <= h < 22.0:
            return self.peak_rate
        return self.normal_rate

    def tariff_bin(self, hour: float) -> int:
        """Map an hour of day to a discrete tariff bin.

        Returns:
            0 = Off-Peak, 1 = Normal, 2 = Peak.
        """
        h = hour % 24.0
        if 22.0 <= h or h < 6.0:
            return 0
        if 18.0 <= h < 22.0:
            return 2
        return 1


@dataclass(frozen=True)
class EnvConfig:
    """Environment / simulation parameters."""

    dt_hours: float = 0.5               # Time step length (hours) -> 30 min
    steps_per_day: int = 48             # 48 half-hour steps per day
    # Reward weighting coefficients
    alpha_degradation: float = 0.02     # Weight on battery degradation penalty
    beta_soc_violation: float = 40.0    # Weight on SoC boundary violation penalty
    # Discrete battery actions (kW). Negative = charge, Positive = discharge.
    battery_actions_kw: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)


@dataclass(frozen=True)
class TrainConfig:
    """Tabular Q-Learning training hyperparameters."""

    alpha: float = 0.1                  # Learning rate
    gamma: float = 0.99                 # Discount factor
    epsilon_start: float = 1.0          # Initial exploration rate
    epsilon_end: float = 0.05           # Final exploration rate
    epsilon_decay_episodes: int = 500   # Episodes over which epsilon decays
    num_episodes: int = 500             # Total training episodes
    moving_average_window: int = 20     # Window for moving-average logging
    milestone_episodes: Tuple[int, ...] = (50, 250, 500)
    seed: int = 42                      # Global RNG seed for reproducibility


@dataclass(frozen=True)
class DiscretizationConfig:
    """Discretization scheme for tabular Q-learning (27 total states)."""

    net_demand_bins: int = 3            # Deficit / Balanced / Surplus
    soc_bins: int = 3                   # Low / Mid / High
    tariff_bins: int = 3                # Off-Peak / Normal / Peak
    # Net demand thresholds (kW) separating Surplus | Balanced | Deficit
    net_demand_low: float = -0.1        # net < low            -> Surplus
    net_demand_high: float = 0.1        # low <= net <= high   -> Balanced
    # SoC thresholds separating Low | Mid | High, aligned to the safe band
    # edges so the agent can learn to stop discharging at the floor (0.20)
    # and stop charging at the ceiling (0.80).
    soc_low: float = 0.20
    soc_high: float = 0.80

    @property
    def num_states(self) -> int:
        """Total number of discrete states."""
        return self.net_demand_bins * self.soc_bins * self.tariff_bins


@dataclass(frozen=True)
class Config:
    """Aggregate configuration container."""

    battery: BatteryConfig = field(default_factory=BatteryConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    disc: DiscretizationConfig = field(default_factory=DiscretizationConfig)

    @property
    def num_actions(self) -> int:
        """Number of discrete battery actions."""
        return len(self.env.battery_actions_kw)


# Convenient module-level singleton used across the project.
CONFIG = Config()


def get_config() -> Config:
    """Return the shared immutable configuration instance."""
    return CONFIG
