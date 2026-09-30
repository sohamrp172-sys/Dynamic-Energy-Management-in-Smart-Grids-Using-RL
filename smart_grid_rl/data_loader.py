"""Data loading utilities for the Smart Grid RL project.

Loads the Ausgrid "Solar Home Electricity" dataset when available, otherwise
synthesizes a realistic stochastic residential load / solar profile so that the
project runs out-of-the-box on any machine.

Ausgrid format (one row per Customer/date/Consumption Category):
    Customer, Generator Capacity, Postcode, Consumption Category, date,
    0:30, 1:00, ... , 0:00, Row Quality

Consumption categories:
    GC = General Consumption   -> treated as household Load
    GG = Gross Generation      -> treated as PV generation
    CL = Controlled Load       -> ignored for this project
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

# 48 half-hour interval column labels as they appear in the Ausgrid CSV.
_INTERVAL_COLUMNS = [
    "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00", "4:30",
    "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00",
    "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00",
    "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00",
    "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00",
    "21:30", "22:00", "22:30", "23:00", "23:30", "0:00",
]

# 30-minute kWh -> average kW conversion factor (energy over 0.5 h -> power).
_KWH_TO_KW = 2.0

STEPS_PER_DAY = 48
DEFAULT_DAYS = 30


def _find_header_row(csv_path: str) -> int:
    """Locate the header row index in the Ausgrid CSV.

    The real file begins with a free-text disclaimer line before the true
    header (``Customer,Generator Capacity,...``). This scans the first few
    lines to find it robustly.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh):
            if line.startswith("Customer,"):
                return idx
            if idx > 10:
                break
    return 0


def _load_ausgrid(csv_path: str, customer_id: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a single customer's Load (GC) and PV (GG) series from Ausgrid data.

    Args:
        csv_path: Path to the Ausgrid CSV file.
        customer_id: Customer number to extract (default 1).

    Returns:
        Tuple ``(load_kw, pv_kw)`` as 1-D numpy arrays of equal length, one
        sample per 30-minute interval, expressed in kW.
    """
    header_row = _find_header_row(csv_path)
    df = pd.read_csv(csv_path, skiprows=header_row, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["Customer"].astype(str).str.strip() == str(customer_id)]

    interval_cols = [c for c in _INTERVAL_COLUMNS if c in df.columns]
    if not interval_cols:
        raise ValueError("No recognizable interval columns found in CSV.")

    def _series_for(category: str) -> np.ndarray:
        rows = df[df["Consumption Category"].astype(str).str.strip() == category]
        if rows.empty:
            return np.array([], dtype=np.float64)
        vals = rows[interval_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        flat = vals.to_numpy(dtype=np.float64).reshape(-1)
        return np.clip(flat, 0.0, None) * _KWH_TO_KW

    load_kw = _series_for("GC")
    pv_kw = _series_for("GG")

    n = min(len(load_kw), len(pv_kw))
    if n == 0:
        raise ValueError("Customer data could not be parsed (empty GC/GG).")

    # Align to whole days.
    n = (n // STEPS_PER_DAY) * STEPS_PER_DAY
    return load_kw[:n], pv_kw[:n]


def _generate_synthetic(days: int = DEFAULT_DAYS, seed: int = 42
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic stochastic residential load / PV dataset.

    Produces a morning peak, an evening 19:00-21:00 peak, and a bell-shaped
    solar curve with cloud-cover noise.

    Args:
        days: Number of days to synthesize.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple ``(load_kw, pv_kw)`` of length ``days * 48`` in kW.
    """
    rng = np.random.default_rng(seed)
    steps = days * STEPS_PER_DAY
    hours = (np.arange(steps) % STEPS_PER_DAY) * 0.5  # hour of day per step

    # --- Household load: base + morning peak + evening peak + noise ---
    base_load = 0.35 + 0.15 * np.sin((hours - 3.0) / 24.0 * 2 * np.pi)
    morning_peak = 1.2 * np.exp(-0.5 * ((hours - 7.5) / 1.1) ** 2)
    evening_peak = 2.0 * np.exp(-0.5 * ((hours - 20.0) / 1.3) ** 2)
    day_factor = rng.normal(1.0, 0.08, size=days).repeat(STEPS_PER_DAY)
    noise = rng.normal(0.0, 0.06, size=steps)
    load_kw = (base_load + morning_peak + evening_peak) * day_factor + noise
    load_kw = np.clip(load_kw, 0.05, None)

    # --- Solar PV: bell curve centered at noon with cloud noise ---
    daylight = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)
    clear_sky = 3.2 * daylight ** 1.3
    cloud = np.clip(rng.normal(1.0, 0.18, size=steps), 0.2, 1.25)
    pv_kw = clear_sky * cloud
    pv_kw[daylight <= 0.0] = 0.0
    pv_kw = np.clip(pv_kw, 0.0, None)

    return load_kw.astype(np.float64), pv_kw.astype(np.float64)


def load_or_generate_data(csv_path: str = "ausgrid_data.csv",
                          days: int = DEFAULT_DAYS,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Ausgrid dataset if present, otherwise generate synthetic data.

    Args:
        csv_path: Path to a candidate Ausgrid CSV file.
        days: Days to synthesize when falling back to generated data.
        seed: RNG seed for the synthetic generator.

    Returns:
        Tuple ``(load_kw, pv_kw)`` of equal-length 1-D arrays in kW.
    """
    candidates = [
        csv_path,
        "2012-2013 Solar home electricity data v2.csv",
        os.path.join("..", "2012-2013 Solar home electricity data v2.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                load_kw, pv_kw = _load_ausgrid(path)
                if load_kw.size >= STEPS_PER_DAY:
                    print(f"[data_loader] Loaded Ausgrid data from '{path}' "
                          f"({load_kw.size} samples, "
                          f"{load_kw.size // STEPS_PER_DAY} days).")
                    return load_kw, pv_kw
            except Exception as exc:  # pragma: no cover - robustness guard
                print(f"[data_loader] Failed to parse '{path}' ({exc}); "
                      f"falling back to synthetic data.")

    print(f"[data_loader] No dataset found. Generating {days} days of "
          f"synthetic residential data.")
    return _generate_synthetic(days=days, seed=seed)


def split_train_test(load_kw: np.ndarray, pv_kw: np.ndarray,
                     test_days: int = 1) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Split time series into training and (held-out) test portions.

    Args:
        load_kw: Full load series in kW.
        pv_kw: Full PV series in kW.
        test_days: Number of trailing days reserved for evaluation.

    Returns:
        ``(train_load, train_pv, test_load, test_pv)``.
    """
    test_steps = test_days * STEPS_PER_DAY
    if load_kw.size <= test_steps:
        # Not enough data to split; reuse the same day for both.
        return load_kw, pv_kw, load_kw, pv_kw
    return (load_kw[:-test_steps], pv_kw[:-test_steps],
            load_kw[-test_steps:], pv_kw[-test_steps:])


if __name__ == "__main__":
    load, pv = load_or_generate_data()
    print(f"Load: mean={load.mean():.3f} kW  max={load.max():.3f} kW")
    print(f"PV:   mean={pv.mean():.3f} kW  max={pv.max():.3f} kW")
