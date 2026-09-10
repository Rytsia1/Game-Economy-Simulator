"""
tests/test_integration.py
-------------------------
Integration and end-to-end pipeline tests for the Game Economy Simulator.

These tests exercise realistic multi-module workflows across:
  Configuration -> Simulation Engine -> Diagnostic Engine -> Parameter Optimizer -> Scenario Matrix -> Monte Carlo

Invariants verified:
  1. Full pipeline: Configuration -> Simulation -> Metrics -> Diagnostics.
  2. Full pipeline: Configuration -> Simulation -> Optimizer/Tuner -> Validation feedback loop.
  3. Multi-regime pipeline: Base Configuration -> Scenario Matrix -> Multi-Scenario Diagnostics.
  4. Stochastic risk pipeline: Configuration -> Monte Carlo Batching -> Percentile Fan & Risk Metrics.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest

from analytics.diagnostics import (
    DiagnosticAlert,
    DiagnosticReport,
    Severity,
    run_diagnostics,
)
from analytics.optimizer import (
    TunerProbe,
    TunerResult,
    tune_parameter,
)
from config.scenarios import run_scenario_matrix
from config.settings import (
    ArchetypeProfile,
    EconomyConfig,
    SimulationConfig,
    DEFAULT_ARCHETYPES,
)
from simulation.engine import (
    gini_coefficient,
    income_spending_ratio,
    run_simulation,
)
from simulation.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo,
)


# ===========================================================================
# 1. Full Simulation to Diagnostics Pipeline
# ===========================================================================

class TestSimulationToDiagnosticsPipeline:
    """
    End-to-end integration:
      EconomyConfig + SimulationConfig
      -> run_simulation()
      -> metrics DataFrame & archetype breakdowns
      -> run_diagnostics()
      -> DiagnosticReport with flow ratio, Gini, and alert prescriptions.
    """

    def test_simulation_to_diagnostics_integration(self):
        # 1. Configuration
        num_players = 300
        days = 20
        starting_gold = 100.0

        sim_cfg = SimulationConfig(
            num_players=num_players,
            days=days,
            starting_gold=starting_gold,
            random_seed=42,
            stochastic_mode=True,
        )
        eco_cfg = EconomyConfig(
            quest_reward=50.0,
            quests_per_day=3.0,
            enemy_reward=20.0,
            enemies_per_day=4.0,
            potion_cost=10.0,
            potions_per_day=4.0,
            weapon_cost=200.0,
            weapon_interval_days=7,
            tier_1_weapon_cost=300,
            tier_2_weapon_cost=1500,
        )

        # 2. Execute simulation
        sim_results = run_simulation(sim_cfg, eco_cfg)

        # Invariant checks on simulation outputs
        assert isinstance(sim_results, dict)
        required_keys = {
            "metrics",
            "archetype_metrics",
            "final_balances",
            "archetype_ids",
            "archetypes",
            "time_to_afford",
            "elapsed_ms",
        }
        assert required_keys.issubset(sim_results.keys())

        final_balances = sim_results["final_balances"]
        archetype_ids = sim_results["archetype_ids"]
        df_metrics = sim_results["metrics"]
        df_arch = sim_results["archetype_metrics"]
        time_to_afford = sim_results["time_to_afford"]

        assert len(final_balances) == num_players
        assert len(archetype_ids) == num_players
        assert np.all(final_balances >= 0.0), "Balances must never be negative"
        assert len(df_metrics) == days
        assert (df_metrics["min_gold"] >= 0.0).all()

        # Archetype metrics structure
        assert {"day", "archetype", "median_gold"}.issubset(df_arch.columns)
        assert len(df_arch) == days * len(eco_cfg.archetypes)

        # Time-to-afford structure for all archetypes
        for arch in eco_cfg.archetypes:
            assert arch.name in time_to_afford
            tta = time_to_afford[arch.name]
            assert "tier_1" in tta and "tier_2" in tta
            if tta["tier_1"] is not None:
                assert 1 <= tta["tier_1"] <= days
            if tta["tier_2"] is not None:
                assert 1 <= tta["tier_2"] <= days
                if tta["tier_1"] is not None:
                    assert tta["tier_1"] <= tta["tier_2"], "Tier 1 must precede Tier 2"

        # 3. Feed directly into Diagnostics Engine
        target_day = 12
        diag_report = run_diagnostics(
            sim_results=sim_results,
            eco_config=eco_cfg,
            sim_config=sim_cfg,
            casual_target_day=target_day,
        )

        # Invariant checks on diagnostic outputs
        assert isinstance(diag_report, DiagnosticReport)
        assert diag_report.casual_target_day == target_day
        assert diag_report.flow_ratio > 0.0
        assert 0.0 <= diag_report.gini <= 1.0
        assert 0.0 <= diag_report.casual_fail_rate <= 1.0
        assert isinstance(diag_report.overall_health, Severity)
        assert isinstance(diag_report.summary, str) and len(diag_report.summary) > 0

        # Mathematical consistency between simulation metrics and diagnostic report
        expected_flow = income_spending_ratio(df_metrics)
        assert diag_report.flow_ratio == pytest.approx(expected_flow, rel=1e-4)

        expected_gini = gini_coefficient(final_balances)
        assert diag_report.gini == pytest.approx(expected_gini, rel=1e-4)

        # Alerts integrity
        assert isinstance(diag_report.alerts, list)
        for alert in diag_report.alerts:
            assert isinstance(alert, DiagnosticAlert)
            assert isinstance(alert.severity, Severity)
            assert len(alert.title) > 0
            assert len(alert.recommendation) > 0

        # Overall health must reflect the worst severity among alerts
        severities_in_alerts = [a.severity for a in diag_report.alerts]
        if Severity.CRITICAL in severities_in_alerts:
            assert diag_report.overall_health == Severity.CRITICAL
        elif Severity.WARNING in severities_in_alerts:
            assert diag_report.overall_health in (Severity.CRITICAL, Severity.WARNING)


# ===========================================================================
# 2. Configuration to Optimizer/Tuner Integration Pipeline
# ===========================================================================

class TestSimulationToOptimizerPipeline:
    """
    End-to-end integration:
      Baseline Economy -> tune_parameter()
      -> binary search with probe simulations
      -> TunerResult with convergence metrics
      -> re-simulation with tuned parameter confirms validated affordability.
    """

    def test_optimizer_tuning_and_validation_loop(self):
        # Baseline economy where quest_reward is low, so Casuals afford Tier-1 late
        sim_cfg = SimulationConfig(
            num_players=300,
            days=25,
            starting_gold=50.0,
            random_seed=42,
            stochastic_mode=True,
        )
        eco_cfg = EconomyConfig(
            quest_reward=20.0,
            quests_per_day=3.0,
            enemy_reward=10.0,
            enemies_per_day=4.0,
            potion_cost=8.0,
            potions_per_day=3.0,
            weapon_cost=200.0,
            weapon_interval_days=7,
            tier_1_weapon_cost=350,
            tier_2_weapon_cost=2000,
        )

        target_day = 10
        search_lo = 10.0
        search_hi = 150.0

        # 1. Execute parameter tuning
        tuner_result = tune_parameter(
            eco_config=eco_cfg,
            sim_config=sim_cfg,
            param_name="quest_reward",
            search_lo=search_lo,
            search_hi=search_hi,
            target_archetype="Casual",
            target_tier=1,
            target_day=target_day,
            probe_players=150,
            max_iterations=10,
            convergence_days=1.5,
        )

        # Invariant checks on TunerResult
        assert isinstance(tuner_result, TunerResult)
        assert tuner_result.param_name == "quest_reward"
        assert tuner_result.target_archetype == "Casual"
        assert tuner_result.target_tier == 1
        assert tuner_result.target_day == target_day
        assert search_lo <= tuner_result.best_value <= search_hi
        assert tuner_result.iterations == len(tuner_result.probes)
        assert tuner_result.elapsed_ms > 0.0
        assert tuner_result.validation_elapsed_ms > 0.0
        assert tuner_result.residual >= 0.0
        assert isinstance(tuner_result.target_unreachable, bool)
        # unreachable target must never be marked converged
        if tuner_result.target_unreachable:
            assert not tuner_result.converged

        # Probe history structure
        for probe in tuner_result.probes:
            assert isinstance(probe, TunerProbe)
            assert 1 <= probe.iteration <= tuner_result.iterations
            assert search_lo <= probe.param_value <= search_hi

        # 2. Closed-loop verification: re-run simulation with the recommended value
        tuned_eco = dataclasses.replace(eco_cfg, quest_reward=tuner_result.best_value)
        re_sim_results = run_simulation(sim_cfg, tuned_eco)

        assert np.all(re_sim_results["final_balances"] >= 0.0)
        re_achieved_day = re_sim_results["time_to_afford"]["Casual"]["tier_1"]

        # The re-simulated milestone must match the validated achieved_day reported by the tuner
        assert re_achieved_day == tuner_result.achieved_day

        # Residual calculation must be exact: |achieved_day - target_day|
        if tuner_result.achieved_day is not None:
            expected_residual = abs(tuner_result.achieved_day - target_day)
            assert tuner_result.residual == pytest.approx(expected_residual)


# ===========================================================================
# 3. Multi-Scenario Stress-Testing Pipeline
# ===========================================================================

class TestScenarioStressPipeline:
    """
    End-to-end integration:
      Base Configuration -> run_scenario_matrix()
      -> 4 distinct economic regimes simulated
      -> Comparative summary dataframe & diagnostics across regimes.
    """

    def test_scenario_matrix_end_to_end(self):
        sim_cfg = SimulationConfig(num_players=150, days=15, random_seed=42)
        eco_cfg = EconomyConfig(tier_1_weapon_cost=300, tier_2_weapon_cost=1500)

        matrix_results = run_scenario_matrix(base_sim_config=sim_cfg, base_eco_config=eco_cfg)

        assert isinstance(matrix_results, dict)
        assert {"summary_df", "progression_df", "diagnostics", "results"}.issubset(matrix_results.keys())

        summary_df = matrix_results["summary_df"]
        progression_df = matrix_results["progression_df"]
        diagnostics_dict = matrix_results["diagnostics"]
        results_dict = matrix_results["results"]

        expected_scenarios = {
            "Normal (Baseline)",
            "Gold Rush (Hyper-Inflation)",
            "Economic Crisis (Depression)",
            "Hardcore Shift (Demographic)",
        }

        assert set(summary_df["Scenario"]) == expected_scenarios
        assert set(diagnostics_dict.keys()) == expected_scenarios
        assert set(results_dict.keys()) == expected_scenarios

        # Verify each regime produced valid simulation data and diagnostic reports
        for sc_name in expected_scenarios:
            diag = diagnostics_dict[sc_name]
            assert isinstance(diag, DiagnosticReport)
            assert diag.flow_ratio > 0.0
            assert 0.0 <= diag.gini <= 1.0

            res = results_dict[sc_name]
            assert np.all(res["final_balances"] >= 0.0)

        # Progression dataframe is long-format with day records for each scenario
        assert {"day", "scenario", "median_gold", "avg_gold"}.issubset(progression_df.columns)
        assert len(progression_df) == sim_cfg.days * len(expected_scenarios)
        assert set(progression_df["scenario"].unique()) == expected_scenarios

        # Economic invariant: Gold Rush flow ratio must exceed Normal baseline flow ratio
        gold_rush_flow = summary_df.loc[summary_df["Scenario"] == "Gold Rush (Hyper-Inflation)", "Flow Ratio"].values[0]
        normal_flow = summary_df.loc[summary_df["Scenario"] == "Normal (Baseline)", "Flow Ratio"].values[0]
        assert gold_rush_flow > normal_flow, "Gold Rush must have higher flow ratio than baseline"


# ===========================================================================
# 4. Stochastic Risk Pipeline (Monte Carlo)
# ===========================================================================

class TestMonteCarloRiskPipeline:
    """
    End-to-end integration:
      Base Configuration -> run_monte_carlo()
      -> parameter jittering across M iterations
      -> multi-quantile confidence fan chart percentiles
      -> empirical risk probabilities & stability index.
    """

    def test_monte_carlo_risk_quantification_pipeline(self):
        sim_cfg = SimulationConfig(num_players=120, days=15, random_seed=42)
        eco_cfg = EconomyConfig(quest_reward=50.0, potion_cost=10.0)
        num_runs = 15

        mc_result = run_monte_carlo(
            base_sim_config=sim_cfg,
            base_eco_config=eco_cfg,
            num_runs=num_runs,
            volatility=0.15,
            random_seed=42,
        )

        assert isinstance(mc_result, MonteCarloResult)
        assert mc_result.num_runs == num_runs
        assert mc_result.days == sim_cfg.days
        assert mc_result.elapsed_ms > 0.0

        # Run summaries dataframe
        assert len(mc_result.run_summaries) == num_runs
        assert {"run_id", "flow_ratio", "gini", "median_wealth", "casual_fail_rate"}.issubset(
            mc_result.run_summaries.columns
        )

        # Percentiles dataframe and quantile ordering invariant
        pdf = mc_result.percentiles_df
        assert len(pdf) == sim_cfg.days
        for _, row in pdf.iterrows():
            assert row["p5"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"]

        # Risk probabilities bounds
        assert 0.0 <= mc_result.prob_inflation <= 1.0
        assert 0.0 <= mc_result.prob_poverty <= 1.0
        assert 0.0 <= mc_result.stability_score <= 1.0

        # Stability score calculation invariant:
        # stability_score is fraction of runs where flow_ratio < infl_thresh and casual_fail < pov_thresh
        df = mc_result.run_summaries
        passed = (df["flow_ratio"] < mc_result.inflation_thresh) & (
            df["casual_fail_rate"] < mc_result.poverty_thresh
        )
        expected_stability = float(passed.mean())
        assert mc_result.stability_score == pytest.approx(expected_stability, rel=1e-4)
