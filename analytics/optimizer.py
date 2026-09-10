"""
analytics/optimizer.py
-----------------------
Parameter Auto-Tuner — Binary Search heuristic for pacing targets.

v0.3: Given a design goal like "the median Casual player should afford
Tier-1 gear by approximately Day D*", the tuner searches for a value of a
selected scalar parameter (quest_reward or enemy_reward) via binary search
on an objective function evaluated by stochastic simulation.

The search assumes a monotonic relationship between the parameter and the
milestone day, which holds in expectation for the modelled economy but is
NOT guaranteed for individual stochastic runs.  Results should be treated
as an approximate heuristic estimate, not a mathematically exact solution.

Design constraints
~~~~~~~~~~~~~~~~~~
* Each probe uses N=1 000 players to stay fast (~15–25 ms per eval).
* Maximum 20 binary-search iterations → total budget ~400–500 ms.
* After convergence the best parameter is validated once on full N.
* Convergence tolerance: |objective| <= 1 day OR parameter range <= 1 gold.

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
    print(result.best_value, result.achieved_day, result.converged)
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
    achieved_day: Optional[int]    # None if milestone was never reached within sim duration
    objective:    float            # achieved_day - target_day  (signed error)


@dataclass
class TunerResult:
    """
    Full output of the Auto-Tuner.

    Because each evaluation is a stochastic simulation, the returned
    best_value is an approximate heuristic estimate — not a mathematically
    exact optimum.  Re-running the tuner with the same seed will produce
    the same result; changing the seed may shift the recommendation by a
    few days due to stochastic variance.

    Attributes
    ----------
    param_name            : The parameter that was tuned.
    target_archetype      : Archetype whose milestone day was targeted.
    target_tier           : 1 or 2 — which gear tier was targeted.
    target_day            : The designer-specified milestone day D*.
    best_value            : Approximate parameter value found by binary search.
    achieved_day          : Milestone day produced by best_value (validation run).
                            None if the milestone was never reached within the
                            simulation duration.
    residual              : |achieved_day - target_day| in days, or
                            sim_config.days + 100 when achieved_day is None
                            (target unreachable at best_value).
    converged             : True if residual <= convergence_days at termination.
    target_unreachable    : True if the milestone was never reached within the
                            simulation duration during the validation run.
                            When True, converged is always False.
    iterations            : Total binary-search iterations executed.
    probes                : List of all TunerProbe records (for transparency).
    elapsed_ms            : Total wall-clock time in milliseconds.
    validation_elapsed_ms : Time for the final full-N validation run (ms).
    """
    param_name:            str
    target_archetype:      str
    target_tier:           int
    target_day:            int
    best_value:            float
    achieved_day:          Optional[int]
    residual:              float
    converged:             bool
    target_unreachable:    bool
    iterations:            int
    probes:                List[TunerProbe]
    elapsed_ms:            float
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
    Returns the day where >= 50% of ``target_archetype`` can afford the
    requested gear tier, or None if the milestone was never reached within
    the simulation duration.

    Parameters
    ----------
    eco_config       : Base economy config (will NOT be mutated).
    sim_config       : Base sim config (will NOT be mutated).
    param_name       : One of TUNABLE_PARAMS.
    param_value      : The candidate value to test.
    target_archetype : Name of the archetype to track (e.g. "Casual").
                       Assumed to be valid; callers are responsible for
                       validating the archetype name before calling this.
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
    Binary-search-based 1-D parameter heuristic tuner.

    Searches for a value of ``param_name`` in [search_lo, search_hi] such
    that the median milestone day for ``target_archetype`` is approximately
    equal to ``target_day``.

    The objective is assumed **monotonically decreasing** in ``param_name``
    for income parameters (higher reward -> earlier milestone day) and
    **monotonically increasing** for cost parameters (potion_cost,
    potions_per_day: higher cost -> later milestone day).  This monotonic
    assumption holds in expectation, but individual stochastic runs may
    exhibit noise.  The returned best_value is an approximate estimate.

    If the target milestone is unreachable within the simulation duration
    for the entire search range, the result will have ``target_unreachable``
    set to True and ``converged`` set to False.

    Parameters
    ----------
    eco_config        : Base economy configuration.
    sim_config        : Full simulation config — player count used only for
                        the final validation run.  Search runs use probe_players.
    param_name        : Parameter to tune (see TUNABLE_PARAMS).
    search_lo         : Lower bound of search range.  Must be > 0.
    search_hi         : Upper bound of search range.  Must be > search_lo.
    target_archetype  : Archetype to target (default "Casual").  Must match
                        one of the archetype names in eco_config.archetypes.
    target_tier       : Gear tier milestone (1 or 2).
    target_day        : Desired day D* for 50% of archetype to afford gear.
                        Must be >= 1.
    probe_players     : Number of players per search probe (default 1 000).
    max_iterations    : Maximum binary-search steps.
    convergence_days  : Residual threshold (days) for early stop.

    Returns
    -------
    TunerResult

    Raises
    ------
    ValueError
        If param_name is not a supported tunable parameter.
        If target_tier is not 1 or 2.
        If search_lo >= search_hi.
        If search_lo <= 0.
        If target_day < 1.
        If target_archetype does not match any archetype in eco_config.
    """
    if param_name not in _TUNABLE_SET:
        raise ValueError(
            f"param_name must be one of {sorted(_TUNABLE_SET)}, got {param_name!r}"
        )
    if target_tier not in (1, 2):
        raise ValueError(f"target_tier must be 1 or 2, got {target_tier}")
    if search_lo >= search_hi:
        raise ValueError(f"search_lo ({search_lo}) must be < search_hi ({search_hi})")
    if search_lo <= 0:
        raise ValueError(f"search_lo must be > 0, got {search_lo}")
    if target_day < 1:
        raise ValueError(f"target_day must be >= 1, got {target_day}")

    # Validate archetype name against the configured roster up front so that
    # a typo produces a clear error rather than silently treating every probe
    # as unreachable (which would happen because _probe returns None for an
    # unknown key).
    known_archetypes = {a.name for a in eco_config.archetypes}
    if target_archetype not in known_archetypes:
        raise ValueError(
            f"target_archetype {target_archetype!r} not found in eco_config.archetypes. "
            f"Known archetypes: {sorted(known_archetypes)}"
        )

    t_total = time.perf_counter()

    # Probe config: smaller player count for speed
    probe_sim = dataclasses.replace(sim_config, num_players=probe_players)

    # Determine monotonicity direction:
    # Income params (quest_reward, enemy_reward): higher value -> smaller day (milestone earlier).
    # Cost params (potion_cost, potions_per_day): higher value -> larger day (milestone later).
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

        # Treat an unreachable milestone as a very large day number so the
        # binary search continues to push the parameter toward higher income /
        # lower cost.  The explicit None check avoids any truthiness pitfall
        # (e.g. achieved == 0 would be falsy but is not the same as missing).
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
            # Income param: higher -> smaller day
            # obj > 0 (too late) -> need more income -> increase mid -> move lo up
            if obj > 0:
                lo = mid
            else:
                hi = mid
        else:
            # Cost param: higher -> larger day
            # obj > 0 (too late) -> costs too high -> decrease mid -> move hi down
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

    # Explicit None check: val_achieved == 0 is impossible in practice (Day 0
    # does not exist), but using `or` for a None-fallback would be incorrect
    # style and could mask future bugs.
    target_unreachable = val_achieved is None
    val_day_or_fallback = val_achieved if val_achieved is not None else (sim_config.days + 100)
    residual = abs(val_day_or_fallback - target_day)

    return TunerResult(
        param_name=param_name,
        target_archetype=target_archetype,
        target_tier=target_tier,
        target_day=target_day,
        best_value=best_value,
        achieved_day=val_achieved,
        residual=residual,
        converged=(not target_unreachable) and (residual <= convergence_days),
        target_unreachable=target_unreachable,
        iterations=len(probes),
        probes=probes,
        elapsed_ms=elapsed_ms + val_elapsed_ms,
        validation_elapsed_ms=val_elapsed_ms,
    )
