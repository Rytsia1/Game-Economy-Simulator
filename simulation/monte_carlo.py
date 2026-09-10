"""
simulation/monte_carlo.py
-------------------------
Monte Carlo Risk Analysis Engine for the Game Economy Simulator (Step 4 / v0.4).

Evaluates macro stability and regime resilience across M iterations:
1. Macro Parameter Jittering (Stochastic macro shocks per run).
2. Quantile Aggregation across all simulated days (p5, p25, p50, p75, p95).
3. Risk Probability Quantification:
   - Inflation Probability: P(Flow Ratio >= 1.5)
   - Poverty Trap Probability: P(Casual Failure Rate >= 40%)
   - System Stability Index: Fraction of runs passing both safety gates.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.diagnostics import run_diagnostics, DiagnosticReport
from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation


# ---------------------------------------------------------------------------
# Result Data Model
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Structured outcome of a Monte Carlo risk analysis execution.

    Attributes
    ----------
    percentiles_df   : pd.DataFrame with columns:
                       ['day', 'p5', 'p25', 'p50', 'p75', 'p95', 'mean']
    prob_inflation   : P(Flow Ratio >= inflation_threshold).
    prob_poverty     : P(Casual Poverty Rate >= poverty_threshold).
    stability_score  : Percentage of runs meeting both stability conditions.
    run_summaries    : pd.DataFrame with per-run records:
                       ['run_id', 'quest_reward', 'potion_cost', 'flow_ratio',
                        'gini', 'median_wealth', 'casual_fail_rate', 'is_stable']
    num_runs         : Total iterations M completed.
    days             : Simulation duration in days.
    elapsed_ms       : Total execution wall-clock time in milliseconds.
    volatility       : Macro parameter jittering standard deviation fraction.
    inflation_thresh : Flow ratio threshold used for inflation definition.
    poverty_thresh   : Casual failure rate threshold used for poverty trap.
    """
    percentiles_df:   pd.DataFrame
    prob_inflation:   float
    prob_poverty:     float
    stability_score:  float
    run_summaries:    pd.DataFrame
    num_runs:         int
    days:             int
    elapsed_ms:       float
    volatility:       float
    inflation_thresh: float = 1.50
    poverty_thresh:   float = 0.40


# ---------------------------------------------------------------------------
# Monte Carlo Runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    base_sim_config: SimulationConfig,
    base_eco_config: EconomyConfig,
    num_runs: int = 100,
    volatility: float = 0.15,
    inflation_threshold: float = 1.50,
    poverty_threshold: float = 0.40,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResult:
    """
    Execute M Monte Carlo iterations with macro parameter jittering,
    collect empirical daily wealth quantiles, and quantify economic risk probabilities.

    Parameters
    ----------
    base_sim_config     : SimulationConfig
        Base simulation settings (player count, days, starting gold).
    base_eco_config     : EconomyConfig
        Base economy settings (rewards, costs, archetypes).
    num_runs            : int
        Number of Monte Carlo iterations M (default 100).
    volatility          : float
        Macro jittering standard deviation as a fraction of baseline parameter values
        (e.g., 0.15 = 15% std-dev).
    inflation_threshold : float
        Flow ratio above which an economy is classified as inflationary (default 1.50).
    poverty_threshold   : float
        Casual failure rate above which an economy is deemed a poverty trap (default 0.40).
    random_seed         : int, optional
        RNG seed for macro parameter jittering reproducibility.
    progress_callback   : Callable[[int, int], None], optional
        Optional hook invoked as ``progress_callback(current_run, total_runs)``.

    Returns
    -------
    MonteCarloResult
    """
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")

    t_start = time.perf_counter()
    rng = np.random.default_rng(random_seed)

    days = base_sim_config.days
    mu_quest = float(base_eco_config.quest_reward)
    mu_potion = float(base_eco_config.potion_cost)

    # Standard deviations for macro parameters
    sigma_quest = volatility * mu_quest
    sigma_potion = (volatility * 0.67) * mu_potion  # potion cost has slightly lower macro variance

    daily_medians = np.zeros((num_runs, days), dtype=np.float64)
    run_rows: List[dict] = []

    for m in range(num_runs):
        # 1. Jitter macro parameters (clamped to positive sensible minimums)
        jittered_quest = max(1.0, float(rng.normal(mu_quest, sigma_quest)))
        jittered_potion = max(1.0, float(rng.normal(mu_potion, sigma_potion)))

        # Build per-run config with only the two jittered parameters changed.
        # dataclasses.replace is more efficient than deepcopy here because the
        # archetypes list is read-only during simulation and need not be cloned.
        run_eco = dataclasses.replace(
            base_eco_config,
            quest_reward=jittered_quest,
            potion_cost=jittered_potion,
        )

        # Run independent stochastic simulation
        run_sim = dataclasses.replace(
            base_sim_config,
            random_seed=int(rng.integers(0, 2_147_483_647)) if base_sim_config.random_seed is not None else None,
        )
        sim_res = run_simulation(run_sim, run_eco)

        # 2. Extract daily median trajectory
        metrics_df = sim_res["metrics"]
        daily_medians[m, :] = metrics_df["median_gold"].values

        # 3. Diagnostic checks
        diag: DiagnosticReport = run_diagnostics(sim_res, run_eco, run_sim)
        flow_ratio = diag.flow_ratio
        gini = diag.gini
        casual_fail_rate = diag.casual_fail_rate
        final_median = float(np.median(sim_res["final_balances"]))

        # Condition checks
        is_inflated = flow_ratio >= inflation_threshold
        is_poverty = casual_fail_rate >= poverty_threshold
        is_stable = (not is_inflated) and (not is_poverty)

        run_rows.append({
            "run_id":           m + 1,
            "quest_reward":     round(jittered_quest, 2),
            "potion_cost":      round(jittered_potion, 2),
            "flow_ratio":       round(flow_ratio, 3),
            "gini":             round(gini, 4),
            "median_wealth":    round(final_median, 1),
            "casual_fail_rate": round(casual_fail_rate, 4),
            "is_inflated":      is_inflated,
            "is_poverty":       is_poverty,
            "is_stable":        is_stable,
        })

        if progress_callback is not None:
            progress_callback(m + 1, num_runs)

    run_summaries = pd.DataFrame(run_rows)

    # 4. Compute empirical quantiles across runs for each day
    # daily_medians shape is (M, days)
    p5  = np.percentile(daily_medians, 5, axis=0)
    p25 = np.percentile(daily_medians, 25, axis=0)
    p50 = np.percentile(daily_medians, 50, axis=0)
    p75 = np.percentile(daily_medians, 75, axis=0)
    p95 = np.percentile(daily_medians, 95, axis=0)
    p_mean = np.mean(daily_medians, axis=0)

    days_axis = np.arange(1, days + 1, dtype=int)
    percentiles_df = pd.DataFrame({
        "day":   days_axis,
        "p5":    np.round(p5, 1),
        "p25":   np.round(p25, 1),
        "p50":   np.round(p50, 1),
        "p75":   np.round(p75, 1),
        "p95":   np.round(p95, 1),
        "mean":  np.round(p_mean, 1),
    })

    # 5. Risk probability quantification
    prob_inflation  = float(np.mean(run_summaries["is_inflated"]))
    prob_poverty    = float(np.mean(run_summaries["is_poverty"]))
    stability_score = float(np.mean(run_summaries["is_stable"]))

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return MonteCarloResult(
        percentiles_df   = percentiles_df,
        prob_inflation   = prob_inflation,
        prob_poverty     = prob_poverty,
        stability_score  = stability_score,
        run_summaries    = run_summaries,
        num_runs         = num_runs,
        days             = days,
        elapsed_ms       = elapsed_ms,
        volatility       = volatility,
        inflation_thresh = inflation_threshold,
        poverty_thresh   = poverty_threshold,
    )
