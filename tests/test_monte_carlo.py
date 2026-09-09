"""
tests/test_monte_carlo.py
-------------------------
Unit and property tests for Monte Carlo Risk Analysis Engine (Step 4 / v0.4).
"""

from __future__ import annotations

import time
import pytest
import numpy as np
import pandas as pd

from config.settings import EconomyConfig, SimulationConfig
from simulation.monte_carlo import run_monte_carlo, MonteCarloResult


class TestMonteCarloQuantiles:
    """Test empirical quantile invariants and monotonicity."""

    @pytest.fixture
    def mc_result(self):
        sim = SimulationConfig(num_players=200, days=20, random_seed=42)
        eco = EconomyConfig()
        return run_monte_carlo(sim, eco, num_runs=25, volatility=0.15, random_seed=42)

    def test_quantile_ordering_monotonicity(self, mc_result):
        """
        CRITICAL INVARIANT:
        Across all simulated days t, empirical quantiles must strictly satisfy:
        p5 <= p25 <= p50 <= p75 <= p95
        """
        pdf = mc_result.percentiles_df
        assert len(pdf) == mc_result.days

        for _, row in pdf.iterrows():
            day = row["day"]
            p5, p25, p50, p75, p95 = row["p5"], row["p25"], row["p50"], row["p75"], row["p95"]
            assert p5 <= p25, f"Day {day}: p5 ({p5}) > p25 ({p25})"
            assert p25 <= p50, f"Day {day}: p25 ({p25}) > p50 ({p50})"
            assert p50 <= p75, f"Day {day}: p50 ({p50}) > p75 ({p75})"
            assert p75 <= p95, f"Day {day}: p75 ({p75}) > p95 ({p95})"

    def test_percentiles_dataframe_schema(self, mc_result):
        pdf = mc_result.percentiles_df
        expected_cols = {"day", "p5", "p25", "p50", "p75", "p95", "mean"}
        assert set(pdf.columns) == expected_cols
        assert (pdf["day"] == np.arange(1, mc_result.days + 1)).all()


class TestMonteCarloRiskQuantification:
    """Test risk probabilities and stability bounds."""

    @pytest.fixture
    def mc_result(self):
        sim = SimulationConfig(num_players=200, days=15, random_seed=123)
        eco = EconomyConfig()
        return run_monte_carlo(sim, eco, num_runs=20, volatility=0.20, random_seed=123)

    def test_risk_probabilities_bounded(self, mc_result):
        assert 0.0 <= mc_result.prob_inflation <= 1.0
        assert 0.0 <= mc_result.prob_poverty <= 1.0
        assert 0.0 <= mc_result.stability_score <= 1.0

    def test_stability_score_matches_summaries(self, mc_result):
        summaries = mc_result.run_summaries
        assert len(summaries) == mc_result.num_runs

        expected_prob_inf = float(np.mean(summaries["is_inflated"]))
        expected_prob_pov = float(np.mean(summaries["is_poverty"]))
        expected_stab = float(np.mean(summaries["is_stable"]))

        assert mc_result.prob_inflation == pytest.approx(expected_prob_inf)
        assert mc_result.prob_poverty == pytest.approx(expected_prob_pov)
        assert mc_result.stability_score == pytest.approx(expected_stab)

    def test_runs_summary_has_required_columns(self, mc_result):
        summaries = mc_result.run_summaries
        expected_cols = {
            "run_id", "quest_reward", "potion_cost", "flow_ratio",
            "gini", "median_wealth", "casual_fail_rate",
            "is_inflated", "is_poverty", "is_stable"
        }
        assert expected_cols.issubset(summaries.columns)


class TestMonteCarloParameterJittering:
    """Test macro parameter shock distributions and edge conditions."""

    def test_macro_jittering_generates_variance(self):
        sim = SimulationConfig(num_players=100, days=10)
        eco = EconomyConfig(quest_reward=50.0, potion_cost=10.0)

        res = run_monte_carlo(sim, eco, num_runs=30, volatility=0.20, random_seed=999)
        summaries = res.run_summaries

        # There should be empirical variance in parameter values across runs
        quest_std = float(summaries["quest_reward"].std())
        potion_std = float(summaries["potion_cost"].std())

        assert quest_std > 1.0, f"Expected quest reward std > 1.0, got {quest_std}"
        assert potion_std > 0.2, f"Expected potion cost std > 0.2, got {potion_std}"

        # Values should all be positive
        assert (summaries["quest_reward"] >= 1.0).all()
        assert (summaries["potion_cost"] >= 1.0).all()

    def test_progress_callback_invoked(self):
        sim = SimulationConfig(num_players=100, days=10)
        eco = EconomyConfig()

        calls = []
        def _cb(curr, total):
            calls.append((curr, total))

        num_runs = 10
        run_monte_carlo(sim, eco, num_runs=num_runs, progress_callback=_cb)

        assert len(calls) == num_runs
        assert calls[-1] == (num_runs, num_runs)
        for i, (curr, total) in enumerate(calls, 1):
            assert curr == i
            assert total == num_runs

    def test_invalid_num_runs_raises(self):
        sim = SimulationConfig()
        eco = EconomyConfig()
        with pytest.raises(ValueError, match="at least 1"):
            run_monte_carlo(sim, eco, num_runs=0)


class TestMonteCarloPerformance:
    """Verify performance requirement: fast execution across multiple iterations."""

    def test_multi_run_speed(self):
        sim = SimulationConfig(num_players=500, days=15, random_seed=42)
        eco = EconomyConfig()

        t0 = time.perf_counter()
        res = run_monte_carlo(sim, eco, num_runs=30, volatility=0.15, random_seed=42)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        # 30 runs with 500 players over 15 days should finish well under 1.5 seconds
        assert elapsed < 1.5, f"Expected 30 runs in < 1.5s, took {elapsed:.2f}s"
        assert res.elapsed_ms > 0
