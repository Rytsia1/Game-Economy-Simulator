"""
analytics/optimizer.py
-----------------------
Parameter Auto-Tuner — Binary Search solver for pacing targets.

v0.3: Given a design goal like "the median Casual player should afford
Tier-1 gear by exactly Day D*", the tuner finds the optimal value of a
selected scalar parameter (quest_reward or enemy_reward) via binary search
on monotonic objective function evaluations.

Design constraints
~~~~~~~~~~~~~~~~~~
* Each probe uses N=1 000 players to stay fast (~15–25 ms per eval).
* Maximum 20 binary-search iterations → total budget ≈ 400–500 ms.
* After convergence the best parameter is validated once on full N.
* Convergence tolerance: |objective| ≤ 1 day OR parameter range ≤ 1 gold.

Usage
~~~~~
    from analytics.optimizer import tune_parameter, TunerResult
    result = tune_parameter(
        eco_config       = eco,
        sim_config       = sim,
        param_name       = "quest_reward",
        search_lo        = 10.0,
        search_hi        = 300.0,
        target_archetype = "Casual",
        target_tier      = 1,
        target_day       = 20,
    )
    print(result.best_value, result.achieved_day, result.iterations)
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import run_simulation


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class TunerProbe:
    """Record of a single binary-search probe evaluation."""
    iteration:    int
    param_value:  float
    achieved_day: Optional[int]    # None if milestone was never reached
    objective:    float            # achieved_day - target_day  (signed error)


@dataclass
class TunerResult:
    """
    Full output of the Auto-Tuner.

    Attributes
    ----------
    param_name          : The parameter that was tuned.
    target_archetype    : Archetype whose milestone day was targeted.
    target_tier         : 1 or 2 — which gear tier was targeted.
    target_day          : The designer-specified milestone day D*.
    best_value          : Optimal parameter value found.
    achieved_day        : Milestone day produced by best_value (validation run).
    residual            : |achieved_day - target_day| in days.
    converged           : True if residual ≤ 1 day at termination.
    iterations          : Total binary-search iterations executed.
    probes              : List of all TunerProbe records (for transparency).
    elapsed_ms          : Total wall-clock time in milliseconds.
    validation_elapsed_ms: Time for the final full-N validation run (ms).
    """
    param_name:           str
    target_archetype:     str
    target_tier:          int
    target_day:           int
    best_value:           float
    achieved_day:         Optional[int]
    residual:             float
    converged:            bool
    iterations:           int
    probes:               List[TunerProbe]
    elapsed_ms:           float
    validation_elapsed_ms: float


# ---------------------------------------------------------------------------
# Supported tunable parameters
# ---------------------------------------------------------------------------

TUNABLE_PARAMS = Literal["quest_reward", "enemy_reward", "potion_cost", "potions_per_day"]

_TUNABLE_SET = {"quest_reward", "enemy_reward", "potion_cost", "potions_per_day"}


# ---------------------------------------------------------------------------
# Internal: single simulation probe
# ---------------------------------------------------------------------------

def _probe(
    eco_config: EconomyConfig,
    sim_config: SimulationConfig,
    param_name: str,
    param_value: float,
    target_archetype: str,
    target_tier: int,
) -> Optional[int]:
    """
    Run one simulation with ``param_name`` overridden to ``param_value``.
    Returns the day where ≥50% of ``target_archetype`` can afford the
    requested gear tier, or None if the milestone was never reached.

    Parameters
    ----------
    eco_config       : Base economy config (will NOT be mutated).
    sim_config       : Base sim config (will NOT be mutated).
    param_name       : One of TUNABLE_PARAMS.
    param_value      : The candidate value to test.
    target_archetype : Name of the archetype to track (e.g. "Casual").
    target_tier      : 1 or 2.
    """
    # Build a modified EconomyConfig without mutating the original.
    # dataclasses.replace is safe here: archetypes is a list reference which
    # is shared (read-only during simulation), and the changed field is a scalar.
    probe_eco = dataclasses.replace(eco_config, **{param_name: float(param_value)})

    results = run_simulation(sim_config, probe_eco)
    tier_key = f"tier_{target_tier}"
    return results["time_to_afford"].get(target_archetype, {}).get(tier_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_parameter(
    eco_config:        EconomyConfig,
    sim_config:        SimulationConfig,
    param_name:        str,
    search_lo:         float,
    search_hi:         float,
    target_archetype:  str = "Casual",
    target_tier:       int = 1,
    target_day:        int = 20,
    probe_players:     int = 1_000,
    max_iterations:    int = 20,
    convergence_days:  float = 1.0,
) -> TunerResult:
    """
    Binary-search-based 1-D parameter optimizer.

    The search minimizes ``|median_day_to_afford(archetype, tier) - target_day|``
    over the scalar parameter ``param_name`` ∈ [search_lo, search_hi].

    The objective is assumed **monotonically decreasing** in ``param_name``
    (higher reward → earlier milestone day).  For cost parameters
    (potion_cost) the relationship is inverted automatically.

    Parameters
    ----------
    eco_config        : Base economy configuration.
    sim_config        : Full simulation config — player count used only for
                        the final validation run.  Search runs use probe_players.
    param_name        : Parameter to tune (see TUNABLE_PARAMS).
    search_lo         : Lower bound of search range.
    search_hi         : Upper bound of search range.
    target_archetype  : Archetype to target (default "Casual").
    target_tier       : Gear tier milestone (1 or 2).
    target_day        : Desired day D* for 50% of archetype to afford gear.
    probe_players     : Number of players per search probe (default 1 000).
    max_iterations    : Maximum binary-search steps.
    convergence_days  : Residual threshold (days) for early stop.

    Returns
    -------
    TunerResult
    """
    if param_name not in _TUNABLE_SET:
        raise ValueError(
            f"param_name must be one of {sorted(_TUNABLE_SET)}, got {param_name!r}"
        )
    if target_tier not in (1, 2):
        raise ValueError(f"target_tier must be 1 or 2, got {target_tier}")
    if search_lo >= search_hi:
        raise ValueError(f"search_lo ({search_lo}) must be < search_hi ({search_hi})")

    t_total = time.perf_counter()

    # Probe config: smaller player count for speed
    probe_sim = dataclasses.replace(sim_config, num_players=probe_players)

    # Determine monotonicity direction:
    # Income params (quest_reward, enemy_reward): higher value → smaller day (milestone earlier).
    # Cost params (potion_cost, potions_per_day): higher value → larger day (milestone later).
    cost_params = {"potion_cost", "potions_per_day"}
    is_cost = param_name in cost_params

    probes: list[TunerProbe] = []
    lo, hi = float(search_lo), float(search_hi)
    best_value  = (lo + hi) / 2
    best_day: Optional[int] = None
    best_obj  = float("inf")

    for i in range(max_iterations):
        mid = (lo + hi) / 2.0
        achieved = _probe(eco_config, probe_sim, param_name, mid, target_archetype, target_tier)

        # Treat unreachable milestone as a very large day number
        day_val = achieved if achieved is not None else (sim_config.days + 100)
        obj = float(day_val - target_day)      # signed error: + means too late

        probes.append(TunerProbe(
            iteration=i + 1,
            param_value=mid,
            achieved_day=achieved,
            objective=obj,
        ))

        abs_obj = abs(obj)
        if abs_obj < abs(best_obj):
            best_obj   = obj
            best_value = mid
            best_day   = achieved

        # Binary search step
        if not is_cost:
            # Income param: higher → smaller day
            # obj > 0 (too late) → need more income → increase mid → move lo up
            if obj > 0:
                lo = mid
            else:
                hi = mid
        else:
            # Cost param: higher → larger day
            # obj > 0 (too late) → costs too high → decrease mid → move hi down
            if obj > 0:
                hi = mid
            else:
                lo = mid

        # Early convergence
        if abs_obj <= convergence_days or (hi - lo) <= 1.0:
            break

    elapsed_ms = (time.perf_counter() - t_total) * 1_000

    # --- Final validation on full player count ---
    t_val = time.perf_counter()
    val_achieved = _probe(eco_config, sim_config, param_name, best_value, target_archetype, target_tier)
    val_elapsed_ms = (time.perf_counter() - t_val) * 1_000

    residual = abs((val_achieved or (sim_config.days + 100)) - target_day)

    return TunerResult(
        param_name=param_name,
        target_archetype=target_archetype,
        target_tier=target_tier,
        target_day=target_day,
        best_value=best_value,
        achieved_day=val_achieved,
        residual=residual,
        converged=residual <= convergence_days,
        iterations=len(probes),
        probes=probes,
        elapsed_ms=elapsed_ms + val_elapsed_ms,
        validation_elapsed_ms=val_elapsed_ms,
    )
