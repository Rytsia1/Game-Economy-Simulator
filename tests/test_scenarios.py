"""
tests/test_scenarios.py
-----------------------
Unit and property tests for Scenario Stress-Testing Matrix (Step 4 / v0.4).
"""

from __future__ import annotations

import copy
import pytest
import numpy as np
import pandas as pd

from config.scenarios import (
    get_scenario_presets,
    run_scenario_matrix,
    SCENARIO_COLORS,
    SCENARIO_DESCRIPTIONS,
)
from config.settings import EconomyConfig, SimulationConfig


class TestScenarioPresets:
    """Test preset definitions and parameter mutations."""

    def test_presets_keys(self):
        presets = get_scenario_presets()
        expected_keys = {
            "Normal (Baseline)",
            "Gold Rush (Hyper-Inflation)",
            "Economic Crisis (Depression)",
            "Hardcore Shift (Demographic)",
        }
        assert set(presets.keys()) == expected_keys

    def test_presets_do_not_mutate_base_config(self):
        base_eco = EconomyConfig(quest_reward=50.0, enemy_reward=20.0, potion_cost=10.0)
        base_sim = SimulationConfig(num_players=100, days=15)

        base_eco_copy = copy.deepcopy(base_eco)
        base_sim_copy = copy.deepcopy(base_sim)

        presets = get_scenario_presets(base_sim, base_eco)

        # Baseline configs should be completely unchanged
        assert base_eco == base_eco_copy
        assert base_sim == base_sim_copy

    def test_gold_rush_regime_parameters(self):
        base_eco = EconomyConfig(quest_reward=50.0, enemy_reward=20.0)
        presets = get_scenario_presets(base_eco_config=base_eco)

        _, eco_goldrush = presets["Gold Rush (Hyper-Inflation)"]
        assert eco_goldrush.quest_reward == pytest.approx(100.0)
        assert eco_goldrush.enemy_reward == pytest.approx(36.0)
        # Sinks unchanged
        assert eco_goldrush.potion_cost == base_eco.potion_cost
        assert eco_goldrush.weapon_cost == base_eco.weapon_cost

    def test_economic_crisis_regime_parameters(self):
        base_eco = EconomyConfig(
            quest_reward=50.0,
            enemy_reward=20.0,
            potion_cost=10.0,
            weapon_cost=300.0,
            tier_1_weapon_cost=500,
            tier_2_weapon_cost=2500,
            potions_per_day=5.0,
        )
        presets = get_scenario_presets(base_eco_config=base_eco)

        _, eco_crisis = presets["Economic Crisis (Depression)"]
        assert eco_crisis.quest_reward == pytest.approx(35.0)
        assert eco_crisis.enemy_reward == pytest.approx(14.0)
        assert eco_crisis.potion_cost == pytest.approx(12.5)
        assert eco_crisis.weapon_cost == pytest.approx(375.0)
        assert eco_crisis.tier_1_weapon_cost == int(500 * 1.25)
        assert eco_crisis.tier_2_weapon_cost == int(2500 * 1.25)
        assert eco_crisis.potions_per_day == pytest.approx(6.0)

    def test_hardcore_shift_demographics(self):
        presets = get_scenario_presets()
        _, eco_hardcore = presets["Hardcore Shift (Demographic)"]

        shares = {a.name: a.population_share for a in eco_hardcore.archetypes}
        assert shares["Grinder"] == pytest.approx(0.50)
        assert shares["Optimizer"] == pytest.approx(0.30)
        assert shares["Casual"] == pytest.approx(0.10)
        assert shares["Collector"] == pytest.approx(0.10)
        assert sum(shares.values()) == pytest.approx(1.0)


class TestRunScenarioMatrix:
    """Test execution of multi-scenario simulation runner."""

    @pytest.fixture
    def fast_configs(self):
        sim = SimulationConfig(num_players=200, days=15, random_seed=42)
        eco = EconomyConfig()
        return sim, eco

    def test_matrix_structure_and_columns(self, fast_configs):
        sim, eco = fast_configs
        matrix = run_scenario_matrix(base_sim_config=sim, base_eco_config=eco)

        assert "summary_df" in matrix
        assert "progression_df" in matrix
        assert "results" in matrix
        assert "diagnostics" in matrix

        sum_df = matrix["summary_df"]
        assert len(sum_df) == 4
        required_cols = {
            "Scenario", "Avg Gold", "Median Gold", "Gini",
            "Flow Ratio", "Casual T1 Afford Day", "Health Status"
        }
        assert required_cols.issubset(sum_df.columns)

        prog_df = matrix["progression_df"]
        assert len(prog_df) == 4 * sim.days
        assert set(prog_df.columns) == {"day", "scenario", "median_gold", "avg_gold"}

    def test_relative_wealth_dynamics_across_scenarios(self, fast_configs):
        sim, eco = fast_configs
        matrix = run_scenario_matrix(base_sim_config=sim, base_eco_config=eco)
        sum_df = matrix["summary_df"].set_index("Scenario")

        # Gold Rush should have substantially higher average gold than Baseline
        normal_avg = sum_df.loc["Normal (Baseline)", "Avg Gold"]
        goldrush_avg = sum_df.loc["Gold Rush (Hyper-Inflation)", "Avg Gold"]
        crisis_avg = sum_df.loc["Economic Crisis (Depression)", "Avg Gold"]

        assert goldrush_avg > normal_avg
        assert crisis_avg < normal_avg

        # Gold Rush flow ratio should be higher than normal
        normal_flow = sum_df.loc["Normal (Baseline)", "Flow Ratio"]
        goldrush_flow = sum_df.loc["Gold Rush (Hyper-Inflation)", "Flow Ratio"]
        assert goldrush_flow > normal_flow

    def test_diagnostics_produced_for_all_scenarios(self, fast_configs):
        sim, eco = fast_configs
        matrix = run_scenario_matrix(base_sim_config=sim, base_eco_config=eco)
        diags = matrix["diagnostics"]

        assert len(diags) == 4
        for sc_name, d_report in diags.items():
            assert d_report.overall_health.value in {"OK", "INFO", "WARNING", "CRITICAL"}
            assert len(d_report.alerts) > 0
            assert 0.0 <= d_report.gini <= 1.0
