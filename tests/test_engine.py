"""
tests/test_engine.py
---------------------
Pytest suite for simulation/engine.py

Verifications
~~~~~~~~~~~~~
1. Gold never drops below zero for any player on any day.
2. Cumulative earned/spent are mathematically consistent at every checkpoint.
3. Edge cases: zero sources, zero sinks, weapon-interval = 1, single player.
4. Gini coefficient is in [0, 1].
5. Income/spending ratio is correct when spending is zero.
6. Performance: 10 000 players × 90 days runs in < 1 000 ms.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_configs():
    """Standard configuration used as the baseline for most tests."""
    eco = EconomyConfig()
    sim = SimulationConfig(num_players=500, days=30, random_seed=0)
    return sim, eco


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(sim_override: dict | None = None, eco_override: dict | None = None) -> dict:
    """Build configs with optional field overrides and run."""
    import dataclasses

    eco = EconomyConfig(**(eco_override or {}))
    sim = SimulationConfig(**({"num_players": 200, "days": 10, "random_seed": 1} | (sim_override or {})))
    return run_simulation(sim, eco)


# ---------------------------------------------------------------------------
# 1. Gold floors at zero
# ---------------------------------------------------------------------------

class TestGoldFloor:
    def test_final_balances_non_negative(self, default_configs):
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        assert np.all(results["final_balances"] >= 0.0), (
            "Some players have negative final balances."
        )

    def test_min_gold_non_negative_all_days(self, default_configs):
        """Per-day minimum gold must never be negative."""
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        assert (results["metrics"]["min_gold"] >= 0.0).all(), (
            "min_gold metric dipped below zero on at least one day."
        )

    def test_aggressive_sinks_still_non_negative(self):
        """Very high sinks should drain to zero, not go negative."""
        results = _run(
            sim_override={"num_players": 1000, "days": 30, "starting_gold": 10.0},
            eco_override={
                "potion_cost": 500.0,   # enormous daily potion cost
                "potions_per_day": 10.0,
                "weapon_cost": 1_000.0,
                "weapon_interval_days": 1,
                "quest_reward": 1.0,    # tiny income
                "quests_per_day": 1.0,
                "enemy_reward": 0.0,
                "enemies_per_day": 0.0,
            },
        )
        assert np.all(results["final_balances"] >= 0.0)

    def test_zero_income_floor(self):
        """With zero income and large sinks, balances must floor at 0."""
        results = _run(
            eco_override={
                "quest_reward": 0.0, "quests_per_day": 0.0,
                "enemy_reward": 0.0, "enemies_per_day": 0.0,
                "potion_cost": 100.0, "potions_per_day": 5.0,
            },
        )
        assert np.all(results["final_balances"] >= 0.0)


# ---------------------------------------------------------------------------
# 2. Cumulative math consistency
# ---------------------------------------------------------------------------

class TestCumulativeMath:
    def test_cumulative_earned_non_decreasing(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        diffs = df["cumulative_earned"].diff().dropna()
        assert (diffs >= 0).all(), "cumulative_earned decreased — logic error."

    def test_cumulative_spent_non_decreasing(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        diffs = df["cumulative_spent"].diff().dropna()
        assert (diffs >= 0).all(), "cumulative_spent decreased — logic error."

    def test_daily_sums_match_cumulative(self, default_configs):
        """Sum of daily earned == final cumulative_earned."""
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert abs(
            df["total_earned_day"].sum() - df["cumulative_earned"].iloc[-1]
        ) < 1e-6, "Sum of daily earned does not equal final cumulative value."

    def test_net_flow_matches_difference(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        computed_net = df["total_earned_day"] - df["total_spent_day"]
        np.testing.assert_allclose(
            df["net_flow_day"].values,
            computed_net.values,
            atol=1e-6,
            err_msg="net_flow_day does not equal earned - spent.",
        )

    def test_day_count(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert len(df) == sim.days, f"Expected {sim.days} rows, got {len(df)}."

    def test_day_index_is_sequential(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        np.testing.assert_array_equal(
            df["day"].values,
            np.arange(1, sim.days + 1),
            err_msg="Day index is not 1…N.",
        )


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_player(self):
        results = _run(sim_override={"num_players": 1, "days": 7})
        assert results["final_balances"].shape == (1,)
        assert results["final_balances"][0] >= 0.0

    def test_zero_sinks(self):
        """With no spending, final balance must be >= starting gold."""
        results = _run(
            sim_override={"starting_gold": 200.0, "days": 20},
            eco_override={
                "potion_cost": 0.0, "potions_per_day": 0.0,
                "weapon_cost": 0.0, "weapon_interval_days": 7,
            },
        )
        # All players earn and spend nothing — balances must only grow
        assert np.all(results["final_balances"] >= 200.0)

    def test_weapon_interval_one(self):
        """Weapon purchased every day — should not go negative."""
        results = _run(
            eco_override={"weapon_interval_days": 1, "weapon_cost": 50.0},
        )
        assert np.all(results["final_balances"] >= 0.0)

    def test_zero_noise(self):
        """With zero noise, all players should have identical balances."""
        results = _run(
            sim_override={
                "num_players": 100,
                "days": 10,
                "noise_std_pct": 0.0,
                "starting_gold": 500.0,
            },
        )
        balances = results["final_balances"]
        # All balances should be equal (or very close) with zero noise
        assert np.allclose(balances, balances[0], atol=1e-6), (
            "With zero noise, all player balances should be identical."
        )

    def test_reproducible_with_seed(self):
        """Same seed must produce identical results."""
        r1 = _run(sim_override={"random_seed": 99})
        r2 = _run(sim_override={"random_seed": 99})
        np.testing.assert_array_equal(r1["final_balances"], r2["final_balances"])

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different results."""
        r1 = _run(sim_override={"random_seed": 1, "num_players": 500})
        r2 = _run(sim_override={"random_seed": 2, "num_players": 500})
        assert not np.array_equal(r1["final_balances"], r2["final_balances"])


# ---------------------------------------------------------------------------
# 4. Statistical helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_gini_perfect_equality(self):
        arr = np.ones(1000) * 500.0
        g = gini_coefficient(arr)
        assert abs(g) < 1e-6, f"Gini of equal distribution should be ~0, got {g}"

    def test_gini_maximum_inequality(self):
        arr = np.zeros(999)
        arr = np.append(arr, 1_000_000.0)
        g = gini_coefficient(arr)
        assert g > 0.99, f"Gini of maximum inequality should be ~1, got {g}"

    def test_gini_in_range(self, default_configs):
        sim, eco = default_configs
        balances = run_simulation(sim, eco)["final_balances"]
        g = gini_coefficient(balances)
        assert 0.0 <= g <= 1.0, f"Gini coefficient {g} out of [0, 1] range."

    def test_gini_empty_array(self):
        assert gini_coefficient(np.array([])) == 0.0

    def test_gini_all_zeros(self):
        assert gini_coefficient(np.zeros(100)) == 0.0

    def test_income_spending_ratio_zero_income(self):
        """If spending is zero the ratio should be infinite."""
        results = _run(
            eco_override={
                "potion_cost": 0.0, "potions_per_day": 0.0,
                "weapon_cost": 0.0, "weapon_interval_days": 7,
            },
        )
        r = income_spending_ratio(results["metrics"])
        assert r == float("inf") or r > 1.0, (
            "With zero sinks, ISR should be ∞ or very large."
        )

    def test_income_spending_ratio_balanced(self):
        """A crafted config where income ≈ spend should give ratio ≈ 1."""
        # Daily income = 50*1 + 10*1 = 60
        # Daily potions = 60*1 = 60  (exact balance)
        results = _run(
            sim_override={"num_players": 2000, "days": 30, "noise_std_pct": 0.0},
            eco_override={
                "quest_reward": 50.0, "quests_per_day": 1.0,
                "enemy_reward": 10.0, "enemies_per_day": 1.0,
                "potion_cost": 60.0,  "potions_per_day": 1.0,
                "weapon_cost": 0.0,   "weapon_interval_days": 7,
            },
        )
        r = income_spending_ratio(results["metrics"])
        assert abs(r - 1.0) < 0.01, (
            f"Expected ISR ≈ 1.0 for balanced economy, got {r:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    @pytest.mark.parametrize(
        "num_players,days,budget_ms",
        [
            (1_000,  30,  500),   # typical small run
            (5_000,  60,  800),   # medium run
            (10_000, 90, 1_000),  # specification maximum
        ],
    )
    def test_performance_under_budget(self, num_players, days, budget_ms):
        sim = SimulationConfig(
            num_players=num_players,
            days=days,
            random_seed=42,
        )
        eco = EconomyConfig()
        t0 = time.perf_counter()
        run_simulation(sim, eco)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        assert elapsed_ms < budget_ms, (
            f"{num_players} players × {days} days took {elapsed_ms:.1f} ms "
            f"(budget: {budget_ms} ms)."
        )
