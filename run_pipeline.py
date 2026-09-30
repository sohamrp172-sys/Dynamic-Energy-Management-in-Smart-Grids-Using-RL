"""Master pipeline: data prep -> training -> evaluation -> reporting.

Run the entire project end-to-end with a single command:

    python run_pipeline.py
"""

from __future__ import annotations

import os
import time

from config import get_config
from data_loader import load_or_generate_data
from evaluate import evaluate
from train import train


def main() -> None:
    """Execute the full smart-grid RL pipeline."""
    cfg = get_config()
    artifact_dir = "artifacts"
    figures_dir = "figures"
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" Dynamic Microgrid Energy Management & Battery Dispatch (RL) ")
    print("=" * 70)

    # --- 1. Data preparation ---
    print("\n[1/3] Preparing dataset ...")
    load_kw, pv_kw = load_or_generate_data()
    print(f"      Load  : mean={load_kw.mean():.3f} kW  "
          f"peak={load_kw.max():.3f} kW")
    print(f"      Solar : mean={pv_kw.mean():.3f} kW  "
          f"peak={pv_kw.max():.3f} kW")
    print(f"      Days  : {load_kw.size // cfg.env.steps_per_day}")

    # --- 2. Training ---
    print(f"\n[2/3] Training Q-learning agent "
          f"({cfg.train.num_episodes} episodes) ...")
    t0 = time.time()
    _, history = train(config=cfg, artifact_dir=artifact_dir, verbose=True)
    print(f"      Training time: {time.time() - t0:.1f} s")

    # --- 3. Evaluation ---
    print("\n[3/3] Evaluating strategies and generating figures ...")
    evaluate(history=history, config=cfg, artifact_dir=artifact_dir,
             figures_dir=figures_dir)

    print("Figures written to:")
    for fig in ("figure_1_convergence.png", "figure_2_power_dispatch.png",
                "figure_3_soc_tariff_arbitrage.png"):
        print(f"      - {os.path.join(figures_dir, fig)}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
