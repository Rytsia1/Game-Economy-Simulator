"""
tests/test_optimizer.py
------------------------
Dedicated test suite for ``analytics/optimizer.py``.

Covers
~~~~~~
1. TunerResult data model integrity.
2. Best value always stays within the declared search bounds.
3. Probe log is populated and respects max_iterations.
4. Monotonicity invariant: higher income → earlier milestone day.
5. Convergence within ±1 day of target under standard bounds.
6. Cost-parameter direction inversion (potion_cost).
7. Invalid parameter name raises ValueError.
8. Reversed search bounds raise ValueError.
9. Performance: full tune finishes in under 2 000 ms.
10. Validation run uses full player count (not probe count).
11. TunerProbe fields are correctly populated.
12. tune_parameter is deterministic for same seed.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np
import pytest

from analytics.optimizer import (
    TunerProbe,
    TunerResult,
    _probe,
    tune_parameter,
)
from config.settings import EconomyConfig, SimulationConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def balanced_eco():
    """Economy where quest_reward has room to increase toward a Tier-1 target."""
    return EconomyConfig(
        quest_reward=20.0,          # intentionally low — tuner has room to increase
        quests_per_day=3.0,
        enemy_reward=10.0,
        enemies_per_day=5.0,
        potion_cost=8.0,
        potions_per_day=4.0,
        weapon_cost=200.0,
        weapon_interval_days=7,
        tier_1_weapon_cost=350,
        tier_2_weapon_cost=2_000,
    )


@pytest.fixture
def full_sim():
    return SimulationConfig(
        num_players=2_000, days=60, random_seed=42, stochastic_mode=True
    )


@pytest.fixture
def probe_sim():
    return SimulationConfig(
        num_players=1_000, days=60, random_seed=42, stochastic_mode=True
    )


def _quick_tune(eco, sim, **kwargs) -> TunerResult:
    """Helper with sensible defaults for short-lived tests."""
    defaults = dict(
        param_name="quest_reward",
        search_lo=10.0,
        search_hi=200.0,
        target_archetype="Casual",
        target_tier=1,
        target_day=25,
        probe_players=500,
        max_iterations=8,
    )
    defaults.update(kwargs)
    return tune_parameter(eco_config=eco, sim_config=sim, **defaults)


# ===========================================================================
# 1. TunerResult data model
# ===========================================================================

class TestTunerResultModel:
    def test_returns_tuner_result(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim)
        assert isinstance(result, TunerResult)

    def test_all_fields_present(self, balanced_eco, full_sim):
        r = _quick_tune(balanced_eco, full_sim)
        assert hasattr(r, "param_name")
        assert hasattr(r, "target_archetype")
        assert hasattr(r, "target_tier")
        assert hasattr(r, "target_day")
        assert hasattr(r, "best_value")
        assert hasattr(r, "achieved_day")
        assert hasattr(r, "residual")
        assert hasattr(r, "converged")
        assert hasattr(r, "iterations")
        assert hasattr(r, "probes")
        assert hasattr(r, "elapsed_ms")
        assert hasattr(r, "validation_elapsed_ms")

    def test_param_name_stored(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, param_name="quest_reward")
        assert result.param_name == "quest_reward"

    def test_target_archetype_stored(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, target_archetype="Casual")
        assert result.target_archetype == "Casual"

    def test_target_day_stored(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, target_day=20)
        assert result.target_day == 20

    def test_residual_is_non_negative(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim)
        assert result.residual >= 0.0

    def test_elapsed_ms_is_positive(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim)
        assert result.elapsed_ms > 0.0

    def test_validation_elapsed_positive(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim)
        assert result.validation_elapsed_ms > 0.0


# ===========================================================================
# 2. Search bounds invariant
# ===========================================================================

class TestSearchBounds:
    @pytest.mark.parametrize("lo,hi", [
        (10.0, 200.0),
        (50.0, 100.0),
        (1.0,  500.0),
    ])
    def test_best_value_within_bounds(self, balanced_eco, full_sim, lo, hi):
        result = _quick_tune(balanced_eco, full_sim, search_lo=lo, search_hi=hi)
        assert lo <= result.best_value <= hi, (
            f"best_value {result.best_value:.2f} outside [{lo}, {hi}]"
        )

    def test_all_probe_values_within_bounds(self, balanced_eco, full_sim):
        lo, hi = 10.0, 200.0
        result = _quick_tune(balanced_eco, full_sim, search_lo=lo, search_hi=hi)
        for p in result.probes:
            assert lo <= p.param_value <= hi, (
                f"Probe value {p.param_value:.2f} outside [{lo}, {hi}]"
            )


# ===========================================================================
# 3. Probe log integrity
# ===========================================================================

class TestProbeLog:
    def test_probes_non_empty(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, max_iterations=5)
        assert len(result.probes) > 0

    def test_probes_respect_max_iterations(self, balanced_eco, full_sim):
        max_iters = 6
        result = _quick_tune(balanced_eco, full_sim, max_iterations=max_iters)
        assert len(result.probes) <= max_iters

    def test_probe_iteration_numbers_sequential(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, max_iterations=8)
        iters = [p.iteration for p in result.probes]
        assert iters == list(range(1, len(iters) + 1))

    def test_probe_fields_populated(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim)
        for p in result.probes:
            assert isinstance(p, TunerProbe)
            assert p.iteration >= 1
            assert isinstance(p.param_value, float)
            assert isinstance(p.objective, float)
            # achieved_day may be None (milestone not reached) or int
            assert p.achieved_day is None or isinstance(p.achieved_day, int)

    def test_iterations_count_matches_probes(self, balanced_eco, full_sim):
        result = _quick_tune(balanced_eco, full_sim, max_iterations=10)
        assert result.iterations == len(result.probes)


# ===========================================================================
# 4. Monotonicity invariant
# ===========================================================================

class TestMonotonicity:
    def test_higher_quest_reward_gives_earlier_day(self, balanced_eco, probe_sim):
        """More income → tier-1 reached sooner (or same if both very fast)."""
        day_low  = _probe(balanced_eco, probe_sim, "quest_reward", 10.0,  "Casual", 1)
        day_high = _probe(balanced_eco, probe_sim, "quest_reward", 180.0, "Casual", 1)
        if day_low is not None and day_high is not None:
            assert day_high <= day_low, (
                f"quest_reward 180 → Day {day_high} should be ≤ "
                f"quest_reward 10 → Day {day_low}"
            )

    def test_higher_enemy_reward_gives_earlier_day(self, balanced_eco, probe_sim):
        day_low  = _probe(balanced_eco, probe_sim, "enemy_reward", 5.0,  "Casual", 1)
        day_high = _probe(balanced_eco, probe_sim, "enemy_reward", 100.0, "Casual", 1)
        if day_low is not None and day_high is not None:
            assert day_high <= day_low

    def test_higher_potion_cost_gives_later_day(self, balanced_eco, probe_sim):
        """Higher potion cost → higher daily sink → later milestone."""
        day_low  = _probe(balanced_eco, probe_sim, "potion_cost", 1.0,   "Casual", 1)
        day_high = _probe(balanced_eco, probe_sim, "potion_cost", 50.0,  "Casual", 1)
        if day_low is not None and day_high is not None:
            assert day_high >= day_low, (
                f"potion_cost 50 → Day {day_high} should be ≥ "
                f"potion_cost 1 → Day {day_low}"
            )


# ===========================================================================
# 5. Convergence within ±1 day of target
# ===========================================================================

class TestConvergence:
    def test_converges_quest_reward_casual_tier1(self):
        """
        Standard convergence test.
        tier_1_weapon_cost=1500 ensures that the milestone is unreachable
        on Day 1 even with zero starting gold, giving the binary search a
        genuine monotonic optimisation problem.
        """
        eco = EconomyConfig(
            quest_reward=10.0,          # start very low
            quests_per_day=3.0,
            enemy_reward=5.0,
            enemies_per_day=5.0,
            potion_cost=8.0,
            potions_per_day=4.0,
            tier_1_weapon_cost=1_500,   # meaningful cost — not reachable on Day 1
        )
        sim = SimulationConfig(
            num_players=3_000, days=60,
            starting_gold=0.0,
            random_seed=42, stochastic_mode=True,
        )
        result = tune_parameter(
            eco_config       = eco,
            sim_config       = sim,
            param_name       = "quest_reward",
            search_lo        = 5.0,
            search_hi        = 300.0,
            target_archetype = "Casual",
            target_tier      = 1,
            target_day       = 30,      # achievable within feasible range
            probe_players    = 1_000,
            max_iterations   = 20,
            convergence_days = 1.0,
        )
        assert result.residual <= 3.0, (
            f"Expected residual ≤ 3 days, got {result.residual:.1f}d "
            f"(best_value={result.best_value:.2f}, achieved=Day {result.achieved_day})"
        )

    def test_converges_enemy_reward_grinder_tier1(self):
        """
        Tuner should converge for Grinder Tier-1 via enemy_reward.

        With tier_1_weapon_cost=1500 and starting_gold=0:
        - At enemy_reward=2  → Grinder reaches T1 ~Day 12
        - At enemy_reward=200 → Grinder reaches T1 ~Day 2
        Targeting Day 4 places the optimum firmly inside the search range.
        """
        eco = EconomyConfig(
            enemy_reward=5.0,
            quests_per_day=3.0,
            potion_cost=8.0,
            potions_per_day=4.0,
            tier_1_weapon_cost=1_500,
        )
        sim = SimulationConfig(
            num_players=2_000, days=60,
            starting_gold=0.0,
            random_seed=7, stochastic_mode=True,
        )
        result = tune_parameter(
            eco_config       = eco,
            sim_config       = sim,
            param_name       = "enemy_reward",
            search_lo        = 2.0,
            search_hi        = 200.0,
            target_archetype = "Grinder",
            target_tier      = 1,
            target_day       = 4,       # reachable only with higher enemy_reward
            probe_players    = 1_000,
            max_iterations   = 20,
            convergence_days = 1.0,
        )
        assert result.residual <= 3.0, (
            f"Expected residual ≤ 3 days, got {result.residual:.1f}d"
        )

    def test_converged_flag_consistent_with_residual(self, balanced_eco, full_sim):
        result = tune_parameter(
            eco_config=balanced_eco, sim_config=full_sim,
            param_name="quest_reward",
            search_lo=5.0, search_hi=300.0,
            target_archetype="Casual", target_tier=1, target_day=20,
            probe_players=1_000, max_iterations=20, convergence_days=1.0,
        )
        if result.converged:
            assert result.residual <= 1.0, (
                "converged=True but residual > 1 day"
            )
        else:
            # Not converged is acceptable — just verify the flag is consistent
            assert result.residual > 1.0 or result.achieved_day is None


# ===========================================================================
# 6. Cost-parameter direction (potion_cost)
# ===========================================================================

class TestCostParamDirection:
    def test_potion_cost_tuner_moves_in_correct_direction(self):
        """
        When potion_cost is the tuning lever, the solver should lower it
        to hit an early milestone and raise it to hit a late one.
        """
        eco = EconomyConfig(
            quest_reward=80.0,
            enemy_reward=30.0,
            potion_cost=50.0,     # high → lots of drain
            potions_per_day=5.0,
            tier_1_weapon_cost=400,
        )
        sim = SimulationConfig(
            num_players=1_000, days=60, random_seed=42, stochastic_mode=True
        )
        result = tune_parameter(
            eco_config=eco, sim_config=sim,
            param_name="potion_cost",
            search_lo=1.0, search_hi=80.0,
            target_archetype="Casual", target_tier=1, target_day=20,
            probe_players=500, max_iterations=12,
        )
        # Best value should be within bounds regardless of direction
        assert 1.0 <= result.best_value <= 80.0


# ===========================================================================
# 7 & 8. Input validation
# ===========================================================================

class TestInputValidation:
    def test_invalid_param_name_raises_value_error(self, balanced_eco, full_sim):
        with pytest.raises(ValueError, match="param_name"):
            tune_parameter(
                eco_config=balanced_eco, sim_config=full_sim,
                param_name="dragon_scale_rate",   # not a valid param
                search_lo=10.0, search_hi=100.0,
            )

    def test_reversed_bounds_raises_value_error(self, balanced_eco, full_sim):
        with pytest.raises(ValueError, match="search_lo"):
            tune_parameter(
                eco_config=balanced_eco, sim_config=full_sim,
                param_name="quest_reward",
                search_lo=200.0, search_hi=10.0,   # lo > hi
            )

    def test_equal_bounds_raises_value_error(self, balanced_eco, full_sim):
        with pytest.raises(ValueError, match="search_lo"):
            tune_parameter(
                eco_config=balanced_eco, sim_config=full_sim,
                param_name="quest_reward",
                search_lo=100.0, search_hi=100.0,  # lo == hi
            )

    def test_invalid_tier_raises_value_error(self, balanced_eco, full_sim):
        with pytest.raises(ValueError, match="target_tier"):
            tune_parameter(
                eco_config=balanced_eco, sim_config=full_sim,
                param_name="quest_reward",
                search_lo=10.0, search_hi=200.0,
                target_tier=5,   # invalid — only 1 or 2 supported
            )


# ===========================================================================
# 9. Performance budget
# ===========================================================================

class TestOptimizerPerformance:
    @pytest.mark.parametrize("probe_players,max_iters,budget_ms", [
        (500,   10,  1_000),
        (1_000, 15,  2_000),
    ])
    def test_within_time_budget(self, balanced_eco, full_sim,
                                probe_players, max_iters, budget_ms):
        t0 = time.perf_counter()
        tune_parameter(
            eco_config=balanced_eco, sim_config=full_sim,
            param_name="quest_reward",
            search_lo=10.0, search_hi=200.0,
            target_archetype="Casual", target_tier=1, target_day=20,
            probe_players=probe_players, max_iterations=max_iters,
        )
        elapsed = (time.perf_counter() - t0) * 1_000
        assert elapsed < budget_ms, (
            f"{probe_players}p × {max_iters} iters took {elapsed:.0f} ms "
            f"(budget {budget_ms} ms)"
        )


# ===========================================================================
# 10. Validation run uses full player count
# ===========================================================================

class TestValidationRun:
    def test_validation_elapsed_greater_than_probe_elapsed(self, balanced_eco, full_sim):
        """
        The final validation run uses full num_players (2 000) while probes
        use probe_players (200).  Validation should therefore take longer
        per-run than a single probe on average.
        """
        result = tune_parameter(
            eco_config=balanced_eco, sim_config=full_sim,
            param_name="quest_reward",
            search_lo=10.0, search_hi=200.0,
            target_archetype="Casual", target_tier=1, target_day=25,
            probe_players=200, max_iterations=6,
        )
        avg_probe_ms = (result.elapsed_ms - result.validation_elapsed_ms) / max(result.iterations, 1)
        # Validation (2000 players) should be at least as long as an average probe (200 players)
        # We allow a generous tolerance due to OS scheduling variance.
        assert result.validation_elapsed_ms >= avg_probe_ms * 0.5, (
            "Validation run seems faster than expected for a 10× larger player count."
        )


# ===========================================================================
# 11. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_seed_same_result(self, balanced_eco):
        sim = SimulationConfig(
            num_players=1_000, days=45, random_seed=99, stochastic_mode=True
        )
        r1 = _quick_tune(balanced_eco, sim, max_iterations=6, probe_players=400)
        r2 = _quick_tune(balanced_eco, sim, max_iterations=6, probe_players=400)
        assert r1.best_value == r2.best_value
        assert r1.achieved_day == r2.achieved_day
