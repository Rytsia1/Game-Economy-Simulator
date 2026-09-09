"""
tests/test_diagnostics.py
--------------------------
Dedicated test suite for ``analytics/diagnostics.py``.

Covers
~~~~~~
1. DiagnosticReport data model integrity (fields, types, ordering).
2. Gini coefficient computation against known mathematical distributions.
3. Flow-ratio threshold boundaries → correct Severity level.
4. Gini threshold → WARNING boundary.
5. Casual fail-rate threshold boundaries.
6. overall_health equals the worst alert in the report.
7. Poverty-trap and critical-inflation end-to-end smoke tests.
8. Prescriptive recommendations are non-empty strings.
9. run_diagnostics returns a DiagnosticReport for extreme inputs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from analytics.diagnostics import (
    DiagnosticAlert,
    DiagnosticReport,
    Severity,
    _check_macro_flow,
    _check_progression_bottleneck,
    _check_wealth_disparity,
    _FLOW_BALANCED_LOW,
    _FLOW_CRITICAL_INFLATION,
    _FLOW_MODERATE_INFLATION,
    _GINI_WARNING_THRESHOLD,
    _POVERTY_FAIL_RATE_WARN,
    run_diagnostics,
)
from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import gini_coefficient, run_simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def standard_run():
    """Cached simulation results for the default economy (used across tests)."""
    eco = EconomyConfig()
    sim = SimulationConfig(num_players=1_500, days=45, random_seed=0, stochastic_mode=True)
    results = run_simulation(sim, eco)
    return results, eco, sim


@pytest.fixture(scope="module")
def standard_report(standard_run):
    results, eco, sim = standard_run
    return run_diagnostics(results, eco, sim, casual_target_day=20)


# ===========================================================================
# 1. DiagnosticReport structure
# ===========================================================================

class TestDiagnosticReportStructure:
    def test_returns_diagnostic_report(self, standard_report):
        assert isinstance(standard_report, DiagnosticReport)

    def test_required_fields_exist(self, standard_report):
        r = standard_report
        assert hasattr(r, "flow_ratio")
        assert hasattr(r, "gini")
        assert hasattr(r, "casual_fail_rate")
        assert hasattr(r, "alerts")
        assert hasattr(r, "overall_health")
        assert hasattr(r, "summary")
        assert hasattr(r, "casual_target_day")

    def test_field_types(self, standard_report):
        r = standard_report
        assert isinstance(r.flow_ratio, float)
        assert isinstance(r.gini, float)
        assert isinstance(r.casual_fail_rate, float)
        assert isinstance(r.alerts, list)
        assert isinstance(r.overall_health, Severity)
        assert isinstance(r.summary, str)
        assert isinstance(r.casual_target_day, int)

    def test_flow_ratio_positive(self, standard_report):
        assert standard_report.flow_ratio >= 0.0

    def test_gini_in_unit_interval(self, standard_report):
        assert 0.0 <= standard_report.gini <= 1.0

    def test_casual_fail_rate_in_unit_interval(self, standard_report):
        assert 0.0 <= standard_report.casual_fail_rate <= 1.0

    def test_alerts_sorted_by_severity(self, standard_report):
        """Alerts must be ordered CRITICAL → WARNING → INFO → OK."""
        rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.OK: 3}
        ranks = [rank[a.severity] for a in standard_report.alerts]
        assert ranks == sorted(ranks), "Alerts are not sorted from most to least severe."

    def test_alert_fields_populated(self, standard_report):
        for a in standard_report.alerts:
            assert isinstance(a, DiagnosticAlert)
            assert a.criterion, "Alert criterion must be non-empty."
            assert a.title, "Alert title must be non-empty."
            assert a.detail, "Alert detail must be non-empty."
            assert a.recommendation, "Alert recommendation must be non-empty."
            assert isinstance(a.metric_value, float)
            assert a.metric_label, "Alert metric_label must be non-empty."


# ===========================================================================
# 2. Gini coefficient — known mathematical distributions
# ===========================================================================

class TestGiniCoefficient:
    """
    The Gini coefficient has well-known closed-form values for
    certain distributions.  We test the NumPy-vectorized implementation
    (exposed via simulation.engine.gini_coefficient) against them.
    """

    def test_perfect_equality_gives_zero(self):
        """All players have the same wealth → Gini = 0."""
        arr = np.full(1_000, 500.0)
        assert gini_coefficient(arr) == pytest.approx(0.0, abs=1e-9)

    def test_single_winner_gives_near_one(self):
        """One player owns all wealth → Gini ≈ (N-1)/N → near 1 for large N."""
        n = 10_000
        arr = np.zeros(n)
        arr[0] = 1_000_000.0
        g = gini_coefficient(arr)
        expected = (n - 1) / n       # exact value for point-mass Lorenz curve
        assert g == pytest.approx(expected, rel=1e-4)

    def test_two_equal_halves(self):
        """
        N players: half have 0, half have 1.
        Lorenz curve: first 50% own 0, last 50% own 100%.
        Gini = 0.5 exactly.
        """
        n = 10_000
        arr = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        g = gini_coefficient(arr)
        assert g == pytest.approx(0.5, abs=1e-3)

    def test_gini_increases_with_inequality(self):
        """
        Three distributions of increasing inequality should yield
        monotonically increasing Gini values.
        """
        rng = np.random.default_rng(42)
        low_ineq  = rng.uniform(900, 1_100, 5_000)          # narrow band
        mid_ineq  = rng.exponential(scale=1_000, size=5_000) # moderate tail
        high_ineq = np.append(np.zeros(4_800), rng.exponential(5_000, 200))  # extreme

        g_low  = gini_coefficient(low_ineq)
        g_mid  = gini_coefficient(mid_ineq)
        g_high = gini_coefficient(high_ineq)

        assert g_low < g_mid < g_high, (
            f"Expected g_low({g_low:.3f}) < g_mid({g_mid:.3f}) < g_high({g_high:.3f})"
        )

    def test_uniform_distribution_gini(self):
        """
        For a Uniform(0, b) distribution the theoretical Gini = 1/3 ≈ 0.333.
        With N=100 000 the empirical estimate should be within ±0.005.
        """
        rng = np.random.default_rng(7)
        arr = rng.uniform(0, 1_000, 100_000)
        g = gini_coefficient(arr)
        assert g == pytest.approx(1 / 3, abs=0.005)

    def test_empty_array_returns_zero(self):
        assert gini_coefficient(np.array([])) == 0.0

    def test_all_zeros_returns_zero(self):
        assert gini_coefficient(np.zeros(500)) == 0.0

    def test_single_element_returns_zero(self):
        assert gini_coefficient(np.array([42.0])) == 0.0


# ===========================================================================
# 3. Flow-ratio threshold boundaries
# ===========================================================================

class TestFlowRatioSeverity:
    @pytest.mark.parametrize("ratio, expected", [
        # Below poverty threshold
        (0.0,                           Severity.CRITICAL),
        (_FLOW_BALANCED_LOW - 0.01,     Severity.CRITICAL),
        # Balanced zone (lower boundary inclusive)
        (_FLOW_BALANCED_LOW,            Severity.OK),
        (1.1,                           Severity.OK),
        (_FLOW_MODERATE_INFLATION - 0.01, Severity.OK),
        # Moderate inflation (lower boundary inclusive)
        (_FLOW_MODERATE_INFLATION,      Severity.WARNING),
        (1.5,                           Severity.WARNING),
        (_FLOW_CRITICAL_INFLATION - 0.01, Severity.WARNING),
        # Critical inflation (lower boundary inclusive)
        (_FLOW_CRITICAL_INFLATION,      Severity.CRITICAL),
        (3.0,                           Severity.CRITICAL),
    ])
    def test_boundary(self, ratio, expected):
        alert = _check_macro_flow(ratio)
        assert alert.severity == expected, (
            f"Flow ratio {ratio:.2f} → expected {expected}, got {alert.severity}"
        )

    def test_poverty_trap_criterion_tag(self):
        alert = _check_macro_flow(0.5)
        assert alert.criterion == "MACRO_FLOW"

    def test_critical_inflation_contains_recommendation(self):
        alert = _check_macro_flow(2.5)
        assert len(alert.recommendation) > 20, "Recommendation should be descriptive."


# ===========================================================================
# 4. Gini warning threshold
# ===========================================================================

class TestGiniWarningSeverity:
    def test_below_threshold_is_ok(self):
        alert = _check_wealth_disparity(_GINI_WARNING_THRESHOLD - 0.01)
        assert alert.severity == Severity.OK

    def test_at_threshold_is_warning(self):
        alert = _check_wealth_disparity(_GINI_WARNING_THRESHOLD)
        assert alert.severity == Severity.WARNING

    def test_above_threshold_is_warning(self):
        alert = _check_wealth_disparity(0.80)
        assert alert.severity == Severity.WARNING

    def test_warning_has_recommendation(self):
        alert = _check_wealth_disparity(0.70)
        assert alert.recommendation


# ===========================================================================
# 5. Casual fail-rate boundaries
# ===========================================================================

class TestCasualFailRate:
    def test_low_fail_rate_is_ok(self):
        alert = _check_progression_bottleneck(0.10, target_day=20, tier_1_cost=500)
        assert alert.severity == Severity.OK

    def test_medium_fail_rate_is_info(self):
        alert = _check_progression_bottleneck(0.20, target_day=20, tier_1_cost=500)
        assert alert.severity == Severity.INFO

    def test_high_fail_rate_is_warning(self):
        alert = _check_progression_bottleneck(
            _POVERTY_FAIL_RATE_WARN + 0.01, target_day=20, tier_1_cost=500
        )
        assert alert.severity == Severity.WARNING

    def test_recommendation_mentions_tier1_cost(self):
        alert = _check_progression_bottleneck(0.60, target_day=20, tier_1_cost=500)
        assert "500" in alert.recommendation or "425" in alert.recommendation


# ===========================================================================
# 6. overall_health == worst alert severity
# ===========================================================================

class TestOverallHealth:
    def test_overall_equals_worst_severity(self, standard_report):
        rank = {Severity.CRITICAL: 0, Severity.WARNING: 1,
                Severity.INFO: 2, Severity.OK: 3}
        worst_rank = min(rank[a.severity] for a in standard_report.alerts)
        expected   = [s for s, r in rank.items() if r == worst_rank][0]
        assert standard_report.overall_health == expected

    def test_critical_economy_has_critical_health(self):
        eco = EconomyConfig(
            quest_reward=200.0, quests_per_day=10.0,
            enemy_reward=100.0, enemies_per_day=10.0,
            potion_cost=0.0, potions_per_day=0.0,
            weapon_cost=0.0, weapon_interval_days=7,
        )
        sim = SimulationConfig(
            num_players=500, days=20, random_seed=0, stochastic_mode=False
        )
        results = run_simulation(sim, eco)
        report  = run_diagnostics(results, eco, sim)
        assert report.overall_health == Severity.CRITICAL

    def test_poverty_trap_has_critical_health(self):
        eco = EconomyConfig(
            quest_reward=1.0, quests_per_day=1.0,
            enemy_reward=1.0, enemies_per_day=1.0,
            potion_cost=500.0, potions_per_day=10.0,
        )
        sim = SimulationConfig(
            num_players=500, days=20, random_seed=0, stochastic_mode=False
        )
        results = run_simulation(sim, eco)
        report  = run_diagnostics(results, eco, sim)
        assert report.overall_health == Severity.CRITICAL


# ===========================================================================
# 7. Prescriptive recommendation quality
# ===========================================================================

class TestRecommendationQuality:
    @pytest.mark.parametrize("flow_ratio", [0.4, 0.95, 1.5, 2.5])
    def test_recommendation_is_non_empty(self, flow_ratio):
        alert = _check_macro_flow(flow_ratio)
        assert len(alert.recommendation.strip()) > 0

    def test_all_alerts_in_report_have_recommendations(self, standard_report):
        for alert in standard_report.alerts:
            assert len(alert.recommendation.strip()) > 0, (
                f"Alert [{alert.criterion}] has an empty recommendation."
            )

    def test_summary_non_empty(self, standard_report):
        assert len(standard_report.summary.strip()) > 0


# ===========================================================================
# 8. run_diagnostics robustness with extreme inputs
# ===========================================================================

class TestDiagnosticsRobustness:
    def test_single_player_does_not_crash(self):
        eco = EconomyConfig()
        sim = SimulationConfig(num_players=1, days=10, random_seed=0, stochastic_mode=True)
        results = run_simulation(sim, eco)
        report  = run_diagnostics(results, eco, sim)
        assert isinstance(report, DiagnosticReport)

    def test_zero_sink_economy(self):
        eco = EconomyConfig(
            potion_cost=0.0, potions_per_day=0.0,
            weapon_cost=0.0, weapon_interval_days=7,
        )
        sim = SimulationConfig(num_players=200, days=10, random_seed=0, stochastic_mode=False)
        results = run_simulation(sim, eco)
        report  = run_diagnostics(results, eco, sim)
        assert isinstance(report, DiagnosticReport)
        # Zero sinks → very high flow ratio → critical inflation
        assert report.flow_ratio >= _FLOW_CRITICAL_INFLATION

    def test_target_day_beyond_simulation(self):
        """casual_target_day > simulation days should not crash."""
        eco = EconomyConfig()
        sim = SimulationConfig(num_players=200, days=10, random_seed=0, stochastic_mode=False)
        results = run_simulation(sim, eco)
        # target_day=50 > days=10 — should degrade gracefully
        report = run_diagnostics(results, eco, sim, casual_target_day=50)
        assert isinstance(report, DiagnosticReport)
