"""
simulation/engine.py
--------------------
Vectorized NumPy simulation engine for the Game Economy Simulator.

Design goals
~~~~~~~~~~~~
* Zero Python loops over players or days for the hot path.
* Supports 10 000 players × 90 days in well under 1 second.
* Returns a clean DataFrame of per-day aggregate metrics plus the final
  per-player balance array for downstream visualisation.

Algorithm (fully vectorised)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Pre-generate a (players × days) income noise matrix in one NumPy call.
2. Pre-compute a boolean (players × days) weapon-purchase mask using modulo
   on a broadcast range array — no looping.
3. Walk through days in a single Python for-loop over *days* (max 90
   iterations of O(P) NumPy ops) rather than O(P×D) pure-Python ops.
   Each iteration is a vectorised slice, keeping total allocations tiny.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config.settings import EconomyConfig, SimulationConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    sim_config: SimulationConfig,
    eco_config: EconomyConfig,
) -> dict:
    """
    Run the game economy simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Player count, duration, starting balance, RNG seed.
    eco_config : EconomyConfig
        Daily income sources and spending sinks.

    Returns
    -------
    dict with keys:
        "metrics"         – pd.DataFrame of per-day aggregated statistics.
        "final_balances"  – np.ndarray of shape (num_players,), final gold.
        "elapsed_ms"      – float, wall-clock time in milliseconds.
    """
    t0 = time.perf_counter()

    rng = np.random.default_rng(sim_config.random_seed)

    P = sim_config.num_players
    D = sim_config.days

    # ------------------------------------------------------------------
    # 1.  Pre-build income matrix  [P × D]
    #     Each player/day cell = base daily income * (1 + noise)
    # ------------------------------------------------------------------
    base_income = eco_config.daily_income  # scalar
    noise = rng.normal(
        loc=0.0,
        scale=eco_config.daily_income * sim_config.noise_std_pct,
        size=(P, D),
    )
    # income is always non-negative even with large noise
    income_matrix = np.maximum(base_income + noise, 0.0)  # [P × D]

    # ------------------------------------------------------------------
    # 2.  Pre-build daily potion cost vector  [D]  (same for all players)
    #     and weapon purchase mask  [D]  (True on weapon-purchase days)
    # ------------------------------------------------------------------
    daily_potion = eco_config.daily_potion_cost  # scalar
    day_indices = np.arange(1, D + 1)             # days 1 … D (1-indexed)
    weapon_days = (day_indices % eco_config.weapon_interval_days == 0)  # [D] bool

    # ------------------------------------------------------------------
    # 3.  Simulate day-by-day, operating on all P players at once
    # ------------------------------------------------------------------
    balances = np.full(P, sim_config.starting_gold, dtype=np.float64)

    # Accumulators for per-day metrics
    day_records: list[dict] = []

    # Cumulative totals
    cumulative_earned: float = 0.0
    cumulative_spent: float = 0.0

    for d in range(D):
        # --- Income ---
        earned_today = income_matrix[:, d]          # [P]

        # --- Potion sink (always paid, balance floored at 0 afterward) ---
        spent_today = np.full(P, daily_potion, dtype=np.float64)  # [P]

        # --- Weapon sink (conditional on day parity AND affordability) ---
        if weapon_days[d]:
            can_afford = balances + earned_today - daily_potion >= eco_config.weapon_cost
            spent_today += can_afford * eco_config.weapon_cost

        # --- Update balances, floor at 0 ---
        balances = np.maximum(balances + earned_today - spent_today, 0.0)

        # --- Day aggregates ---
        total_earned_day = float(earned_today.sum())
        total_spent_day = float(spent_today.sum())
        cumulative_earned += total_earned_day
        cumulative_spent += total_spent_day

        day_records.append(
            {
                "day": int(day_indices[d]),
                "avg_gold": float(balances.mean()),
                "median_gold": float(np.median(balances)),
                "min_gold": float(balances.min()),
                "max_gold": float(balances.max()),
                "total_earned_day": total_earned_day,
                "total_spent_day": total_spent_day,
                "cumulative_earned": cumulative_earned,
                "cumulative_spent": cumulative_spent,
                "net_flow_day": total_earned_day - total_spent_day,
            }
        )

    metrics_df = pd.DataFrame(day_records)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    return {
        "metrics": metrics_df,
        "final_balances": balances.copy(),
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Derived statistics helpers
# ---------------------------------------------------------------------------

def gini_coefficient(balances: np.ndarray) -> float:
    """
    Compute the Gini coefficient for an array of wealth values.

    Returns a value in [0, 1] where 0 = perfect equality, 1 = maximum
    inequality.  Uses the fast sorted-array formula.

    Parameters
    ----------
    balances : np.ndarray, shape (N,)
        Non-negative player gold balances.
    """
    if balances.size == 0:
        return 0.0

    arr = np.sort(balances.astype(np.float64))
    n = arr.size
    total = arr.sum()

    if total == 0.0:
        return 0.0

    # G = (2 * Σ i*x_i) / (n * Σ x_i) - (n+1)/n
    index = np.arange(1, n + 1)
    return float((2.0 * (index * arr).sum()) / (n * total) - (n + 1.0) / n)


def income_spending_ratio(metrics_df: pd.DataFrame) -> float:
    """Return the cumulative income / cumulative spending ratio."""
    total_earned = metrics_df["cumulative_earned"].iloc[-1]
    total_spent = metrics_df["cumulative_spent"].iloc[-1]
    if total_spent == 0:
        return float("inf")
    return float(total_earned / total_spent)
