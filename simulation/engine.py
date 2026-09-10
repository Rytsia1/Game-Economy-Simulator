"""
simulation/engine.py
--------------------
Vectorized NumPy simulation engine for the Game Economy Simulator.

v0.2 additions
~~~~~~~~~~~~~~
* Player archetype assignment (Casual / Grinder / Collector / Optimizer).
* Stochastic daily activity via Poisson quest draws and clipped Normal
  enemy-drop variance — all vectorised across (num_players,) per day.
* Per-archetype median wealth trajectories tracked in the metrics DataFrame.
* "Time-to-Afford" computation: the first day where ≥50% of each archetype
  can afford Tier 1 and Tier 2 gear milestones.
* Backward-compatible v0.1 Gaussian noise fallback when stochastic_mode=False.

Design goals (unchanged from v0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Zero Python loops over individual players.
* 10 000 players × 90 days in well under 1 second.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import ArchetypeProfile, EconomyConfig, SimulationConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_archetypes(
    num_players: int,
    archetypes: List[ArchetypeProfile],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign each player an archetype index in [0, len(archetypes)-1] by
    sampling from the configured population shares.

    Returns
    -------
    np.ndarray of int, shape (num_players,)
    """
    shares = np.array([a.population_share for a in archetypes], dtype=np.float64)
    # Normalise in case shares do not sum exactly to 1.0
    shares /= shares.sum()
    return rng.choice(len(archetypes), size=num_players, p=shares)


def _build_archetype_multiplier_arrays(
    archetype_ids: np.ndarray,
    archetypes: List[ArchetypeProfile],
    field: str,
) -> np.ndarray:
    """
    Build a float array of shape (num_players,) by looking up ``field``
    on each player's assigned ArchetypeProfile.

    Example: field="quest_mult" → [0.6, 1.2, 0.6, …]
    """
    lookup = np.array([getattr(a, field) for a in archetypes], dtype=np.float64)
    return lookup[archetype_ids]   # fancy indexing → (P,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    sim_config: SimulationConfig,
    eco_config: EconomyConfig,
) -> dict:
    """
    Run the game economy simulation with optional stochastic archetype behavior.

    Parameters
    ----------
    sim_config : SimulationConfig
        Player count, duration, starting balance, RNG seed, stochastic_mode.
    eco_config : EconomyConfig
        Daily income sources, spending sinks, gear milestones, and archetype list.

    Returns
    -------
    dict with keys:
        "metrics"           – pd.DataFrame, per-day aggregate statistics plus
                              per-archetype median columns.
        "archetype_metrics" – pd.DataFrame, per-day per-archetype median gold
                              (long format: columns day | archetype | median_gold).
        "final_balances"    – np.ndarray (P,), final gold per player.
        "archetype_ids"     – np.ndarray (P,), archetype index per player.
        "archetypes"        – List[ArchetypeProfile] used in this run.
        "time_to_afford"    – dict mapping archetype name →
                              {"tier_1": int|None, "tier_2": int|None}
        "elapsed_ms"        – float, wall-clock time in milliseconds.
    """
    t0 = time.perf_counter()

    archetypes = eco_config.archetypes
    rng = np.random.default_rng(sim_config.random_seed)

    P: int = sim_config.num_players
    D: int = sim_config.days
    A: int = len(archetypes)

    # ------------------------------------------------------------------
    # 1.  Assign archetypes → per-player multiplier arrays  [P]
    # ------------------------------------------------------------------
    archetype_ids = _assign_archetypes(P, archetypes, rng)

    quest_mults   = _build_archetype_multiplier_arrays(archetype_ids, archetypes, "quest_mult")
    enemy_mults   = _build_archetype_multiplier_arrays(archetype_ids, archetypes, "enemy_mult")
    potion_mults  = _build_archetype_multiplier_arrays(archetype_ids, archetypes, "potion_mult")
    buffer_mults  = _build_archetype_multiplier_arrays(archetype_ids, archetypes, "saving_buffer_mult")

    # Per-player base rates (archetype-scaled)
    base_quests_per_player  = eco_config.quests_per_day  * quest_mults   # [P]
    base_enemies_per_player = eco_config.enemies_per_day * enemy_mults   # [P]
    base_potions_per_player = eco_config.potions_per_day * potion_mults  # [P]

    # ------------------------------------------------------------------
    # 2.  Pre-build stochastic activity matrices  [P × D]
    #     All random draws happen in one block here — no per-day RNG calls.
    # ------------------------------------------------------------------
    if sim_config.stochastic_mode:
        # --- Quest completions: Poisson(λ = archetype_quests) ---
        # Broadcasting: rng.poisson accepts a (P, 1) lam array and broadcasts
        # it to (P, D) draws in one call — no temporary ones-matrix needed.
        quests_matrix = rng.poisson(
            lam=base_quests_per_player[:, None],
            size=(P, D),
        ).astype(np.float64)   # [P × D]

        # --- Enemy counts: Poisson(λ = archetype_enemies) ---
        enemies_matrix = rng.poisson(
            lam=base_enemies_per_player[:, None],
            size=(P, D),
        ).astype(np.float64)   # [P × D]

        # --- Per-enemy gold drops: clipped Normal(μ=enemy_reward, σ=25%μ) ---
        #     We need the total gold per player per day = enemies * avg_drop.
        #     Vectorise by drawing a full (P, D) matrix of per-day average drops.
        mu_drop = eco_config.enemy_reward
        sigma_drop = eco_config.enemy_reward_std_pct * mu_drop
        drop_matrix = rng.normal(
            loc=mu_drop, scale=sigma_drop, size=(P, D)
        ).clip(min=1.0)   # floor at 1 gold per enemy (never negative)   # [P × D]

        # --- Enemy income: enemy_count × per-enemy drop ---
        enemy_income_matrix = enemies_matrix * drop_matrix   # [P × D]

        # --- Quest income: flat reward × count (no per-quest drop variance) ---
        quest_income_matrix = quests_matrix * eco_config.quest_reward  # [P × D]

        # --- Total daily income ---
        income_matrix = quest_income_matrix + enemy_income_matrix   # [P × D]

        # --- Daily potion spend: Poisson draws capped at 0 min ---
        potions_matrix = rng.poisson(
            lam=base_potions_per_player[:, None],
            size=(P, D),
        ).astype(np.float64)   # [P × D]
        potion_spend_matrix = potions_matrix * eco_config.potion_cost  # [P × D]

    else:
        # --- v0.1 backward-compat: Gaussian noise on aggregate income ---
        base_income_per_player = (
            eco_config.quest_reward * base_quests_per_player
            + eco_config.enemy_reward * base_enemies_per_player
        )  # [P]
        noise = rng.normal(
            loc=0.0,
            scale=base_income_per_player[:, np.newaxis] * sim_config.noise_std_pct,
            size=(P, D),
        )
        income_matrix = np.maximum(
            base_income_per_player[:, np.newaxis] + noise, 0.0
        )  # [P × D]

        # (P,) * scalar → (P,); broadcast to (P, D) without a full allocation.
        daily_potion_spend = base_potions_per_player * eco_config.potion_cost  # [P]
        potion_spend_matrix = np.broadcast_to(
            daily_potion_spend[:, None], (P, D)
        )  # [P × D]  — read-only view, no copy

    # ------------------------------------------------------------------
    # 3.  Weapon purchase threshold per player  [P]
    #     Player buys when balance >= weapon_cost * saving_buffer_mult
    # ------------------------------------------------------------------
    weapon_threshold = eco_config.weapon_cost * buffer_mults   # [P]

    day_indices = np.arange(1, D + 1)
    weapon_days = (day_indices % eco_config.weapon_interval_days == 0)  # [D] bool

    # ------------------------------------------------------------------
    # 4.  Day-by-day simulation  (P players vectorised per iteration)
    # ------------------------------------------------------------------
    balances = np.full(P, sim_config.starting_gold, dtype=np.float64)

    # Mask tracking which players have hit each milestone for the first time
    _t1 = eco_config.tier_1_weapon_cost
    _t2 = eco_config.tier_2_weapon_cost
    tier1_hit_day = np.full(P, -1, dtype=np.int32)   # -1 = not yet reached
    tier2_hit_day = np.full(P, -1, dtype=np.int32)

    # Per-archetype median trajectories: stored as (D, A) array
    archetype_median_matrix = np.zeros((D, A), dtype=np.float64)

    day_records: List[dict] = []
    cumulative_earned: float = 0.0
    cumulative_spent: float  = 0.0

    for d in range(D):
        earned_today = income_matrix[:, d]            # [P]
        spent_today  = potion_spend_matrix[:, d].copy()   # [P]  (copy!)

        # Weapon purchase: day parity + individual balance threshold
        if weapon_days[d]:
            projected = balances + earned_today - spent_today
            can_afford = projected >= weapon_threshold   # [P] bool
            spent_today += can_afford.astype(np.float64) * eco_config.weapon_cost

        # Update balances, floor at 0
        balances = np.maximum(balances + earned_today - spent_today, 0.0)

        # Milestone tracking — record first day balance reaches threshold
        newly_t1 = (tier1_hit_day == -1) & (balances >= _t1)
        tier1_hit_day[newly_t1] = int(day_indices[d])

        newly_t2 = (tier2_hit_day == -1) & (balances >= _t2)
        tier2_hit_day[newly_t2] = int(day_indices[d])

        # Per-archetype medians
        for a_idx in range(A):
            mask = archetype_ids == a_idx
            archetype_median_matrix[d, a_idx] = (
                float(np.median(balances[mask])) if mask.any() else 0.0
            )

        total_earned_day = float(earned_today.sum())
        total_spent_day  = float(spent_today.sum())
        cumulative_earned += total_earned_day
        cumulative_spent  += total_spent_day

        row: dict = {
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
        # Add per-archetype median columns  (e.g. "median_Casual")
        for a_idx, arch in enumerate(archetypes):
            row[f"median_{arch.name}"] = archetype_median_matrix[d, a_idx]

        day_records.append(row)

    metrics_df = pd.DataFrame(day_records)

    # ------------------------------------------------------------------
    # 5.  Build long-format archetype metrics DataFrame
    # ------------------------------------------------------------------
    archetype_rows: List[dict] = []
    for a_idx, arch in enumerate(archetypes):
        for d in range(D):
            archetype_rows.append({
                "day": int(day_indices[d]),
                "archetype": arch.name,
                "median_gold": archetype_median_matrix[d, a_idx],
            })
    archetype_metrics_df = pd.DataFrame(archetype_rows)

    # ------------------------------------------------------------------
    # 6.  Time-to-Afford  (day where ≥50% of archetype crosses milestone)
    # ------------------------------------------------------------------
    time_to_afford: Dict[str, dict] = {}
    for a_idx, arch in enumerate(archetypes):
        mask = archetype_ids == a_idx
        t1_days = tier1_hit_day[mask]
        t2_days = tier2_hit_day[mask]

        def _median_day(hit_days: np.ndarray) -> Optional[int]:
            achieved = hit_days[hit_days >= 0]
            if achieved.size == 0:
                return None
            # Median day is the earliest day ≥50% have crossed
            # Equivalent: sort, pick the value at the 50th-percentile index
            n_arch = hit_days.size
            sorted_days = np.sort(achieved)
            # Find first day where cumulative count ≥ 50% of archetype pop
            cumcount = np.arange(1, achieved.size + 1)
            above_50 = cumcount / n_arch >= 0.5
            if not above_50.any():
                return None
            return int(sorted_days[above_50.argmax()])

        time_to_afford[arch.name] = {
            "tier_1": _median_day(t1_days),
            "tier_2": _median_day(t2_days),
        }

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    return {
        "metrics": metrics_df,
        "archetype_metrics": archetype_metrics_df,
        "final_balances": balances.copy(),
        "archetype_ids": archetype_ids.copy(),
        "archetypes": archetypes,
        "time_to_afford": time_to_afford,
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Derived statistics helpers  (unchanged from v0.1)
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

    index = np.arange(1, n + 1)
    return float((2.0 * (index * arr).sum()) / (n * total) - (n + 1.0) / n)


def income_spending_ratio(metrics_df: pd.DataFrame) -> float:
    """Return the cumulative income / cumulative spending ratio."""
    total_earned = metrics_df["cumulative_earned"].iloc[-1]
    total_spent  = metrics_df["cumulative_spent"].iloc[-1]
    if total_spent == 0:
        return float("inf")
    return float(total_earned / total_spent)
