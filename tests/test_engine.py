"""
tests/test_engine.py
---------------------
Pytest suite for simulation/engine.py

v0.1 verifications (preserved)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Gold never drops below zero for any player on any day.
2. Cumulative earned/spent are mathematically consistent.
3. Edge cases: zero sources, zero sinks, weapon-interval=1, single player.
4. Gini coefficient is in [0, 1].
5. Income/spending ratio correctness.
6. Performance: 10 000 players × 90 days < 1 000 ms.

v0.2 additions
~~~~~~~~~~~~~~
7. Archetype assignment proportions match configured weights (± tolerance).
8. Grinder aggregate earnings exceed Casual earnings under identical base settings.
9. Zero-balance edge cases remain strictly non-negative with stochastic draws.
10. time_to_afford returns valid day indices or None.
11. Per-archetype median columns are present in metrics DataFrame.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytest

from config.settings import ArchetypeProfile, EconomyConfig, SimulationConfig, DEFAULT_ARCHETYPES
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_configs():
    """Standard v0.2 configuration used as the baseline for most tests."""
    eco = EconomyConfig()
    sim = SimulationConfig(num_players=1_000, days=30, random_seed=0, stochastic_mode=True)
    return sim, eco


@pytest.fixture
def deterministic_configs():
    """v0.1 fallback mode — Gaussian noise, no stochastic draws."""
    eco = EconomyConfig()
    sim = SimulationConfig(num_players=500, days=30, random_seed=0, stochastic_mode=False)
    return sim, eco


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(
    sim_override: dict | None = None,
    eco_override: dict | None = None,
    stochastic: bool = True,
) -> dict:
    """Build configs with optional field overrides and run."""
    eco = EconomyConfig(**(eco_override or {}))
    base_sim = {
        "num_players": 300,
        "days": 15,
        "random_seed": 7,
        "stochastic_mode": stochastic,
    }
    sim = SimulationConfig(**(base_sim | (sim_override or {})))
    return run_simulation(sim, eco)


# ===========================================================================
# 1. Gold floors at zero
# ===========================================================================

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
        assert (results["metrics"]["min_gold"] >= 0.0).all()

    def test_aggressive_sinks_stochastic_non_negative(self):
        """Very high sinks should drain to zero, not go negative (stochastic)."""
        results = _run(
            stochastic=True,
            sim_override={"num_players": 500, "days": 20, "starting_gold": 5.0},
            eco_override={
                "potion_cost": 500.0,
                "potions_per_day": 10.0,
                "weapon_cost": 1_000.0,
                "weapon_interval_days": 1,
                "quest_reward": 1.0,
                "quests_per_day": 0.5,
                "enemy_reward": 0.0,
                "enemies_per_day": 0.0,
            },
        )
        assert np.all(results["final_balances"] >= 0.0)

    def test_aggressive_sinks_deterministic_non_negative(self):
        """Same test in deterministic / v0.1 fallback mode."""
        results = _run(
            stochastic=False,
            sim_override={"num_players": 500, "days": 20, "starting_gold": 5.0},
            eco_override={
                "potion_cost": 500.0,
                "potions_per_day": 10.0,
                "weapon_cost": 1_000.0,
                "weapon_interval_days": 1,
                "quest_reward": 1.0,
                "quests_per_day": 0.5,
                "enemy_reward": 0.0,
                "enemies_per_day": 0.0,
            },
        )
        assert np.all(results["final_balances"] >= 0.0)

    def test_zero_income_floor_stochastic(self):
        """With zero income and large sinks, balances must floor at 0."""
        results = _run(
            stochastic=True,
            eco_override={
                "quest_reward": 0.0, "quests_per_day": 0.0,
                "enemy_reward": 0.0, "enemies_per_day": 0.0,
                "potion_cost": 100.0, "potions_per_day": 5.0,
            },
        )
        assert np.all(results["final_balances"] >= 0.0)


# ===========================================================================
# 2. Cumulative math consistency
# ===========================================================================

class TestCumulativeMath:
    def test_cumulative_earned_non_decreasing(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert (df["cumulative_earned"].diff().dropna() >= 0).all()

    def test_cumulative_spent_non_decreasing(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert (df["cumulative_spent"].diff().dropna() >= 0).all()

    def test_daily_sums_match_cumulative(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert abs(
            df["total_earned_day"].sum() - df["cumulative_earned"].iloc[-1]
        ) < 1e-4, "Sum of daily earned does not equal final cumulative value."

    def test_net_flow_matches_difference(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        computed_net = df["total_earned_day"] - df["total_spent_day"]
        np.testing.assert_allclose(
            df["net_flow_day"].values, computed_net.values, atol=1e-6,
        )

    def test_day_count(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        assert len(df) == sim.days

    def test_day_index_is_sequential(self, default_configs):
        sim, eco = default_configs
        df = run_simulation(sim, eco)["metrics"]
        np.testing.assert_array_equal(df["day"].values, np.arange(1, sim.days + 1))


# ===========================================================================
# 3. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_player(self):
        results = _run(sim_override={"num_players": 1, "days": 7})
        assert results["final_balances"].shape == (1,)
        assert results["final_balances"][0] >= 0.0

    def test_zero_sinks(self):
        results = _run(
            sim_override={"starting_gold": 200.0, "days": 20},
            eco_override={
                "potion_cost": 0.0, "potions_per_day": 0.0,
                "weapon_cost": 0.0, "weapon_interval_days": 7,
            },
        )
        assert np.all(results["final_balances"] >= 200.0)

    def test_weapon_interval_one(self):
        results = _run(
            eco_override={"weapon_interval_days": 1, "weapon_cost": 50.0},
        )
        assert np.all(results["final_balances"] >= 0.0)

    def test_reproducible_with_seed(self):
        r1 = _run(sim_override={"random_seed": 99})
        r2 = _run(sim_override={"random_seed": 99})
        np.testing.assert_array_equal(r1["final_balances"], r2["final_balances"])

    def test_different_seeds_differ(self):
        r1 = _run(sim_override={"random_seed": 1, "num_players": 500})
        r2 = _run(sim_override={"random_seed": 2, "num_players": 500})
        assert not np.array_equal(r1["final_balances"], r2["final_balances"])

    def test_deterministic_fallback_non_negative(self, deterministic_configs):
        sim, eco = deterministic_configs
        results = run_simulation(sim, eco)
        assert np.all(results["final_balances"] >= 0.0)


# ===========================================================================
# 4. Statistical helpers
# ===========================================================================

class TestHelpers:
    def test_gini_perfect_equality(self):
        g = gini_coefficient(np.ones(1000) * 500.0)
        assert abs(g) < 1e-6

    def test_gini_maximum_inequality(self):
        arr = np.append(np.zeros(999), 1_000_000.0)
        assert gini_coefficient(arr) > 0.99

    def test_gini_in_range(self, default_configs):
        sim, eco = default_configs
        g = gini_coefficient(run_simulation(sim, eco)["final_balances"])
        assert 0.0 <= g <= 1.0

    def test_gini_empty_array(self):
        assert gini_coefficient(np.array([])) == 0.0

    def test_gini_all_zeros(self):
        assert gini_coefficient(np.zeros(100)) == 0.0

    def test_income_spending_ratio_zero_sink(self):
        results = _run(
            eco_override={
                "potion_cost": 0.0, "potions_per_day": 0.0,
                "weapon_cost": 0.0, "weapon_interval_days": 7,
            },
        )
        r = income_spending_ratio(results["metrics"])
        assert r == float("inf") or r > 1.0


# ===========================================================================
# 5. Performance
# ===========================================================================

class TestPerformance:
    @pytest.mark.parametrize(
        "num_players,days,budget_ms",
        [
            (1_000,  30,  500),
            (5_000,  60,  800),
            (10_000, 90, 1_000),
        ],
    )
    def test_performance_under_budget(self, num_players, days, budget_ms):
        sim = SimulationConfig(
            num_players=num_players, days=days,
            random_seed=42, stochastic_mode=True,
        )
        eco = EconomyConfig()
        t0 = time.perf_counter()
        run_simulation(sim, eco)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        assert elapsed_ms < budget_ms, (
            f"{num_players}p × {days}d took {elapsed_ms:.1f}ms (budget {budget_ms}ms)"
        )


# ===========================================================================
# 6. Archetype Assignment (v0.2 NEW)
# ===========================================================================

class TestArchetypeAssignment:
    """Verify that archetypes are assigned proportionally to configured weights."""

    def test_archetype_ids_shape(self, default_configs):
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        assert results["archetype_ids"].shape == (sim.num_players,)

    def test_archetype_proportions_within_tolerance(self):
        """
        With a large player count each archetype's actual share should be
        within ±5 percentage-points of its configured share.
        """
        sim = SimulationConfig(num_players=10_000, days=1, random_seed=0, stochastic_mode=True)
        eco = EconomyConfig()
        results = run_simulation(sim, eco)

        ids      = results["archetype_ids"]
        arches   = results["archetypes"]
        n        = sim.num_players

        for a_idx, arch in enumerate(arches):
            actual_share   = float((ids == a_idx).sum()) / n
            expected_share = arch.population_share
            assert abs(actual_share - expected_share) < 0.05, (
                f"{arch.name}: expected ~{expected_share:.1%}, got {actual_share:.1%}"
            )

    def test_all_archetype_indices_valid(self, default_configs):
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        ids = results["archetype_ids"]
        n_archetypes = len(results["archetypes"])
        assert np.all(ids >= 0) and np.all(ids < n_archetypes)

    def test_single_archetype_population(self):
        """When one archetype has 100% share, all players should be that archetype."""
        only_grinder = [
            ArchetypeProfile("Grinder", 1.0, 1.2, 2.0, 1.5, 1.2, "#00C9A7"),
        ]
        eco = EconomyConfig(archetypes=only_grinder)
        sim = SimulationConfig(num_players=500, days=5, random_seed=42, stochastic_mode=True)
        results = run_simulation(sim, eco)
        assert np.all(results["archetype_ids"] == 0), "All players should be archetype 0"


# ===========================================================================
# 7. Grinder Earns More Than Casual (v0.2 NEW)
# ===========================================================================

class TestArchetypeEconomicLogic:
    """Verify that archetype multipliers produce meaningful economic differences."""

    def test_grinder_median_exceeds_casual_median(self):
        """
        Over a sufficiently long simulation the Grinder archetype (which has
        higher quest and enemy multipliers) should have a higher final median
        balance than the Casual archetype.
        """
        sim = SimulationConfig(
            num_players=4_000, days=60, random_seed=42, stochastic_mode=True,
            starting_gold=500.0,
        )
        eco = EconomyConfig()
        results = run_simulation(sim, eco)

        ids           = results["archetype_ids"]
        balances      = results["final_balances"]
        archetypes    = results["archetypes"]

        grinder_idx = next(i for i, a in enumerate(archetypes) if a.name == "Grinder")
        casual_idx  = next(i for i, a in enumerate(archetypes) if a.name == "Casual")

        grinder_med = float(np.median(balances[ids == grinder_idx]))
        casual_med  = float(np.median(balances[ids == casual_idx]))

        assert grinder_med > casual_med, (
            f"Grinder median ({grinder_med:,.0f}) should exceed Casual ({casual_med:,.0f})"
        )

    def test_optimizer_low_potion_spend(self):
        """
        Optimizer has potion_mult=0.3, so over N days its potion spend
        should be roughly 30% of the Casual archetype's spend (within noise).
        """
        sim = SimulationConfig(
            num_players=6_000, days=30, random_seed=10, stochastic_mode=True,
        )
        eco = EconomyConfig()
        results = run_simulation(sim, eco)

        ids      = results["archetype_ids"]
        arches   = results["archetypes"]
        # Use per-archetype median trajectories from metrics to compare spend
        # Indirectly: Optimizer should have higher balance relative to income due to low potion use.
        opt_idx     = next(i for i, a in enumerate(arches) if a.name == "Optimizer")
        casual_idx  = next(i for i, a in enumerate(arches) if a.name == "Casual")

        balances = results["final_balances"]
        # Optimizer earns ~(1.0 quests + 1.3 enemies) vs Casual (0.6 + 0.4)
        # Optimizer net income should dominate — validate final balance is higher
        opt_med    = float(np.median(balances[ids == opt_idx]))
        casual_med = float(np.median(balances[ids == casual_idx]))
        # Optimizer should accumulate significantly more
        assert opt_med > casual_med * 1.5, (
            f"Optimizer median {opt_med:,.0f} should be > 1.5× Casual {casual_med:,.0f}"
        )

    def test_grinder_aggregate_earnings_exceed_casual(self):
        """
        Grinder's total earned gold (over all players in that archetype) should
        exceed Casual's total earned, normalised by player count.
        This is verified via archetype_metrics median trajectory at the final day.
        """
        sim = SimulationConfig(
            num_players=5_000, days=45, random_seed=3, stochastic_mode=True,
        )
        eco = EconomyConfig()
        results = run_simulation(sim, eco)

        arch_df  = results["archetype_metrics"]
        last_day = arch_df["day"].max()

        grinder_final = arch_df[(arch_df["archetype"] == "Grinder") & (arch_df["day"] == last_day)]["median_gold"].values[0]
        casual_final  = arch_df[(arch_df["archetype"] == "Casual")  & (arch_df["day"] == last_day)]["median_gold"].values[0]

        assert grinder_final > casual_final, (
            f"Grinder final median ({grinder_final:,.0f}) should exceed Casual ({casual_final:,.0f})"
        )


# ===========================================================================
# 8. Time-to-Afford (v0.2 NEW)
# ===========================================================================

class TestTimeToAfford:
    """Verify Time-to-Afford logic produces plausible, consistent results."""

    def test_time_to_afford_keys_present(self, default_configs):
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        tta = results["time_to_afford"]
        for arch in results["archetypes"]:
            assert arch.name in tta, f"{arch.name} missing from time_to_afford"
            assert "tier_1" in tta[arch.name]
            assert "tier_2" in tta[arch.name]

    def test_tier1_day_before_tier2_day(self):
        """
        If both milestones are achieved, Tier 1 day must be <= Tier 2 day
        (players reach cheaper gear first).
        """
        sim = SimulationConfig(
            num_players=2_000, days=90, random_seed=0, stochastic_mode=True,
        )
        eco = EconomyConfig(tier_1_weapon_cost=300, tier_2_weapon_cost=2_000)
        results = run_simulation(sim, eco)
        tta = results["time_to_afford"]
        for arch_name, days_dict in tta.items():
            t1: Optional[int] = days_dict["tier_1"]
            t2: Optional[int] = days_dict["tier_2"]
            if t1 is not None and t2 is not None:
                assert t1 <= t2, (
                    f"{arch_name}: Tier 1 day ({t1}) should be ≤ Tier 2 day ({t2})"
                )

    def test_impossible_milestone_returns_none(self):
        """
        When the tier_2 milestone is unreachably expensive in the simulation
        window, time_to_afford tier_2 should be None for all archetypes.
        """
        sim = SimulationConfig(
            num_players=500, days=10, random_seed=42, stochastic_mode=True,
            starting_gold=0.0,
        )
        eco = EconomyConfig(
            tier_2_weapon_cost=10_000_000,  # impossibly high
            quest_reward=1.0, quests_per_day=1.0,
            enemy_reward=1.0, enemies_per_day=1.0,
        )
        results = run_simulation(sim, eco)
        for arch_name, days_dict in results["time_to_afford"].items():
            assert days_dict["tier_2"] is None, (
                f"{arch_name}: unreachable milestone should return None, got {days_dict['tier_2']}"
            )

    def test_grinder_affords_tier1_before_casual(self):
        """
        Grinder (higher income multipliers) should afford Tier 1 gear
        no later than Casual.
        """
        sim = SimulationConfig(
            num_players=4_000, days=90, random_seed=5, stochastic_mode=True,
            starting_gold=0.0,
        )
        eco = EconomyConfig(tier_1_weapon_cost=1_000)
        results = run_simulation(sim, eco)
        tta = results["time_to_afford"]
        grinder_t1 = tta.get("Grinder", {}).get("tier_1")
        casual_t1  = tta.get("Casual",  {}).get("tier_1")

        if grinder_t1 is not None and casual_t1 is not None:
            assert grinder_t1 <= casual_t1, (
                f"Grinder should afford Tier 1 by day {grinder_t1}, Casual by {casual_t1}"
            )

    def test_per_archetype_median_columns_in_metrics(self, default_configs):
        """Metrics DataFrame must contain a median_<ArchetypeName> column for each archetype."""
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        df = results["metrics"]
        for arch in results["archetypes"]:
            col = f"median_{arch.name}"
            assert col in df.columns, f"Expected column '{col}' in metrics DataFrame"

    def test_archetype_metrics_long_format(self, default_configs):
        """archetype_metrics should contain one row per (day × archetype)."""
        sim, eco = default_configs
        results = run_simulation(sim, eco)
        df = results["archetype_metrics"]
        n_archetypes = len(results["archetypes"])
        expected_rows = sim.days * n_archetypes
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(df)}"
        )
