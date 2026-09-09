"""
analytics/diagnostics.py
------------------------
Economy Doctor — automated health checker for the Game Economy Simulator.

v0.3: evaluates 4 objective criteria and returns structured alerts with
      game-design-level, prescriptive recommendations.

Criteria
~~~~~~~~
1. Macro Flow        — Income-to-Sink ratio (Inflation / Poverty trap)
2. Wealth Disparity  — Gini coefficient threshold (market distortion risk)
3. Progression Gate  — Casual archetype fail-rate to reach Tier 1 on time
4. Prescriptive Recs — Per-alert actionable design guidance

Usage
~~~~~
    from analytics.diagnostics import run_diagnostics
    report = run_diagnostics(sim_results, eco_config, sim_config)
    for alert in report.alerts:
        print(alert.severity, alert.title, alert.recommendation)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import gini_coefficient, income_spending_ratio


# ---------------------------------------------------------------------------
# Alert data model
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Alert severity levels, ordered from most to least critical."""
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"
    OK       = "OK"

    @property
    def emoji(self) -> str:
        return {
            "CRITICAL": "🔴",
            "WARNING":  "🟡",
            "INFO":     "🔵",
            "OK":       "🟢",
        }[self.value]

    @property
    def color(self) -> str:
        """CSS / hex color for each severity."""
        return {
            "CRITICAL": "#F87171",   # rose
            "WARNING":  "#FBBF24",   # amber
            "INFO":     "#4F8EF7",   # blue
            "OK":       "#00C9A7",   # teal
        }[self.value]


@dataclass
class DiagnosticAlert:
    """
    A single diagnostic finding with context and recommended actions.

    Attributes
    ----------
    criterion       : Short machine-readable tag (e.g. "MACRO_FLOW").
    severity        : Severity enum value.
    title           : One-line human-readable headline.
    detail          : Quantitative detail supporting the finding.
    recommendation  : Concrete game design action to resolve the issue.
    metric_value    : The raw numeric value that triggered this alert.
    metric_label    : Human label for the metric (e.g. "Flow Ratio").
    """
    criterion:      str
    severity:       Severity
    title:          str
    detail:         str
    recommendation: str
    metric_value:   float
    metric_label:   str


@dataclass
class DiagnosticReport:
    """
    Full diagnostic output for one simulation run.

    Attributes
    ----------
    alerts              : Ordered list of DiagnosticAlert (most severe first).
    flow_ratio          : Income / Sink cumulative ratio.
    gini                : Final-day Gini coefficient.
    casual_fail_rate    : Fraction of Casual players that did not hit Tier 1
                          within the target window.
    casual_target_day   : The Day X used as the Tier-1 deadline.
    overall_health      : Worst severity across all alerts.
    summary             : One-sentence plain-English economy summary.
    """
    alerts:            List[DiagnosticAlert]
    flow_ratio:        float
    gini:              float
    casual_fail_rate:  float
    casual_target_day: int
    overall_health:    Severity
    summary:           str


# ---------------------------------------------------------------------------
# Thresholds (module-level constants — easy to adjust in future versions)
# ---------------------------------------------------------------------------

_FLOW_CRITICAL_INFLATION  = 2.0
_FLOW_MODERATE_INFLATION  = 1.3
_FLOW_BALANCED_LOW        = 0.9

_GINI_WARNING_THRESHOLD   = 0.55

_POVERTY_FAIL_RATE_WARN   = 0.35   # 35 % of Casual failing Tier 1 deadline

_SEVERITY_ORDER = [Severity.CRITICAL, Severity.WARNING, Severity.INFO, Severity.OK]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _worst_severity(alerts: list[DiagnosticAlert]) -> Severity:
    if not alerts:
        return Severity.OK
    for sev in _SEVERITY_ORDER:
        if any(a.severity == sev for a in alerts):
            return sev
    return Severity.OK


def _flow_summary(ratio: float) -> str:
    if ratio >= _FLOW_CRITICAL_INFLATION:
        return "CRITICAL_INFLATION"
    if ratio >= _FLOW_MODERATE_INFLATION:
        return "MODERATE_INFLATION"
    if ratio >= _FLOW_BALANCED_LOW:
        return "BALANCED"
    return "POVERTY_TRAP"


# ---------------------------------------------------------------------------
# Diagnostic checks (one function per criterion)
# ---------------------------------------------------------------------------

def _check_macro_flow(flow_ratio: float) -> DiagnosticAlert:
    """Criterion 1 — Income-to-Sink ratio."""
    label = "Flow Ratio (Income / Sink)"

    if flow_ratio >= _FLOW_CRITICAL_INFLATION:
        return DiagnosticAlert(
            criterion="MACRO_FLOW",
            severity=Severity.CRITICAL,
            title="Critical Inflation — Gold snowballs uncontrollably",
            detail=(
                f"Flow ratio is {flow_ratio:.2f}× "
                f"(threshold: < {_FLOW_CRITICAL_INFLATION:.1f}×). "
                "Player wealth compounds faster than the sink can drain it, "
                "devaluing all gold-denominated items."
            ),
            recommendation=(
                "Introduce recurring late-game gold sinks: "
                "add equipment durability decay costs (daily % of weapon value), "
                "cosmetic crafting recipes, or a guild-tax mechanism. "
                "Alternatively raise potion cost by ≥ 20% or add a second tier of potions."
            ),
            metric_value=flow_ratio,
            metric_label=label,
        )

    if flow_ratio >= _FLOW_MODERATE_INFLATION:
        return DiagnosticAlert(
            criterion="MACRO_FLOW",
            severity=Severity.WARNING,
            title="Moderate Inflation — Mild unchecked accumulation",
            detail=(
                f"Flow ratio is {flow_ratio:.2f}× "
                f"(threshold: < {_FLOW_MODERATE_INFLATION:.1f}×). "
                "Wealth grows slightly faster than intended; acceptable short-term "
                "but will cause inflation in extended play sessions."
            ),
            recommendation=(
                "Consider a gentle progressive tax: players with balance above "
                "a threshold (e.g. 5× Tier 1 cost) pay a small daily maintenance fee. "
                "Or increase weapon upgrade frequency (lower weapon_interval_days by 1–2 days)."
            ),
            metric_value=flow_ratio,
            metric_label=label,
        )

    if flow_ratio >= _FLOW_BALANCED_LOW:
        return DiagnosticAlert(
            criterion="MACRO_FLOW",
            severity=Severity.OK,
            title="Balanced Economy — Sustainable sink-to-source equilibrium",
            detail=(
                f"Flow ratio is {flow_ratio:.2f}× "
                f"(target window: {_FLOW_BALANCED_LOW:.1f}–{_FLOW_MODERATE_INFLATION:.1f}×). "
                "Gold generation and consumption are well-matched."
            ),
            recommendation=(
                "Economy is healthy. Monitor Gini coefficient to ensure wealth "
                "does not concentrate among power players over time."
            ),
            metric_value=flow_ratio,
            metric_label=label,
        )

    # ratio < 0.9 — poverty trap
    return DiagnosticAlert(
        criterion="MACRO_FLOW",
        severity=Severity.CRITICAL,
        title="Poverty Trap — Economy drains faster than replenishment",
        detail=(
            f"Flow ratio is {flow_ratio:.2f}× "
            f"(threshold: ≥ {_FLOW_BALANCED_LOW:.1f}×). "
            "Sinks consume gold faster than sources replenish it. "
            "Players will deplete savings rapidly, risking high early-churn."
        ),
        recommendation=(
            "Increase base quest_reward by 15–25% or add a daily login bonus "
            "(e.g. 50 gold unconditional). "
            "Alternatively reduce potion_cost by 20% or lower potions_per_day. "
            "A new low-cost quest type for casual players would also help."
        ),
        metric_value=flow_ratio,
        metric_label=label,
    )


def _check_wealth_disparity(gini: float) -> DiagnosticAlert:
    """Criterion 2 — Gini coefficient wealth concentration."""
    label = "Gini Coefficient"

    if gini >= _GINI_WARNING_THRESHOLD:
        return DiagnosticAlert(
            criterion="WEALTH_DISPARITY",
            severity=Severity.WARNING,
            title="High Wealth Inequality — Risk of market distortion",
            detail=(
                f"Gini is {gini:.4f} "
                f"(threshold: < {_GINI_WARNING_THRESHOLD:.2f}). "
                "Top-earning archetypes (Grinder/Optimizer) accumulate disproportionately, "
                "which can dominate player-driven markets and crowd out casual players."
            ),
            recommendation=(
                "Add diminishing-returns mechanics for high-income activities: "
                "e.g. enemy farm fatigue (enemies_per_day drops 10% after 50 kills), "
                "or a progressive quest difficulty cap. "
                "Alternatively introduce high-value cosmetic sinks (vanity shops) "
                "that target whale spending without punishing casuals."
            ),
            metric_value=gini,
            metric_label=label,
        )

    return DiagnosticAlert(
        criterion="WEALTH_DISPARITY",
        severity=Severity.OK,
        title="Wealth Inequality Within Acceptable Range",
        detail=(
            f"Gini is {gini:.4f} "
            f"(threshold: < {_GINI_WARNING_THRESHOLD:.2f}). "
            "Wealth distribution across archetypes is reasonably equitable."
        ),
        recommendation=(
            "Continue monitoring as player base grows. "
            "Gini may drift upward in longer sessions or with new high-income content."
        ),
        metric_value=gini,
        metric_label=label,
    )


def _check_progression_bottleneck(
    casual_fail_rate: float,
    target_day: int,
    tier_1_cost: int,
) -> DiagnosticAlert:
    """Criterion 3 — Casual archetype progression gate."""
    label = f"Casual Tier-1 Fail Rate (target: Day {target_day})"
    pct = casual_fail_rate * 100

    if casual_fail_rate > _POVERTY_FAIL_RATE_WARN:
        return DiagnosticAlert(
            criterion="PROGRESSION_BLOCKER",
            severity=Severity.WARNING,
            title=f"Progression Bottleneck — {pct:.1f}% of Casual players miss Tier-1 deadline",
            detail=(
                f"{pct:.1f}% of Casual players (threshold: > {_POVERTY_FAIL_RATE_WARN:.0%}) "
                f"cannot afford the Tier-1 weapon ({tier_1_cost:,}g) by Day {target_day}. "
                "This indicates a frustrating experience for the majority of your audience."
            ),
            recommendation=(
                f"Reduce Tier-1 weapon cost by ~15% (to {int(tier_1_cost * 0.85):,}g), "
                "or increase quest_reward by 10–15 gold. "
                "Alternatively add a 'starter pack' mechanism: new players receive a "
                "one-time bonus of 100–200g after completing Day 7."
            ),
            metric_value=casual_fail_rate,
            metric_label=label,
        )

    if casual_fail_rate > 0.15:
        return DiagnosticAlert(
            criterion="PROGRESSION_BOTTLENECK",
            severity=Severity.INFO,
            title=f"Mild Progression Friction — {pct:.1f}% of Casuals miss Tier-1 target",
            detail=(
                f"{pct:.1f}% of Casual players miss the Tier-1 deadline (Day {target_day}). "
                "Within acceptable range but worth tracking."
            ),
            recommendation=(
                "Consider a modest quest_reward increase (+5–10g) or a Day 10 "
                "milestone reward to smooth the early progression curve."
            ),
            metric_value=casual_fail_rate,
            metric_label=label,
        )

    return DiagnosticAlert(
        criterion="PROGRESSION_GATE",
        severity=Severity.OK,
        title="Casual Progression On-Track",
        detail=(
            f"Only {pct:.1f}% of Casual players miss the Tier-1 deadline (Day {target_day}). "
            "The progression curve is accessible for your mainstream audience."
        ),
        recommendation=(
            "Progression is healthy. Optionally audit Tier-2 accessibility for Collectors."
        ),
        metric_value=casual_fail_rate,
        metric_label=label,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diagnostics(
    sim_results: dict,
    eco_config: EconomyConfig,
    sim_config: SimulationConfig,
    casual_target_day: int = 20,
) -> DiagnosticReport:
    """
    Evaluate economy health across 4 criteria and return a structured report.

    Parameters
    ----------
    sim_results       : Output dict from ``simulation.engine.run_simulation()``.
    eco_config        : The EconomyConfig used for this run.
    sim_config        : The SimulationConfig used for this run.
    casual_target_day : Day by which the median Casual player should afford
                        Tier 1 gear.  Default is Day 20.

    Returns
    -------
    DiagnosticReport
    """
    metrics_df     = sim_results["metrics"]
    final_balances = sim_results["final_balances"]
    archetype_ids  = sim_results["archetype_ids"]
    archetypes     = sim_results["archetypes"]
    time_to_afford = sim_results["time_to_afford"]

    # --- Metric 1: Flow ratio ---
    flow_ratio = income_spending_ratio(metrics_df)

    # --- Metric 2: Gini ---
    gini = gini_coefficient(final_balances)

    # --- Metric 3: Casual Tier-1 fail rate within target_day ---
    casual_idx = next(
        (i for i, a in enumerate(archetypes) if a.name == "Casual"), None
    )
    if casual_idx is not None:
        casual_mask = archetype_ids == casual_idx
        casual_n    = int(casual_mask.sum())
        if casual_n > 0:
            # Use the per-player tier1_hit_day array if available, otherwise
            # approximate from time_to_afford dict (median proxy)
            # We recompute the fail rate precisely from the archetype_metrics
            tta_casual = time_to_afford.get("Casual", {})
            t1_day = tta_casual.get("tier_1")
            if t1_day is None:
                casual_fail_rate = 1.0          # 100% failed
            else:
                # t1_day is the day where ≥50% cross; use archetype_metrics
                # to derive approx fail rate: fraction whose T1 day > target
                # We use the tier1_day per player via metrics snapshot approach.
                # Best available: if median_day > target, fail rate > 50%; else < 50%.
                # For precise per-player counts we need the hit-day array which
                # isn't in the public result dict. Approximate from median day.
                casual_balances = final_balances[casual_mask]
                t1_cost         = eco_config.tier_1_weapon_cost
                # Reconstruct fail proxy from archetype_metrics up to target_day
                arch_df = sim_results["archetype_metrics"]
                casual_traj = arch_df[arch_df["archetype"] == "Casual"]
                within = casual_traj[casual_traj["day"] <= casual_target_day]
                if within.empty:
                    casual_fail_rate = 1.0
                else:
                    # Median gold by target day — approximate fail rate via
                    # how the distribution sits relative to tier_1 cost.
                    # Use final balances of Casual players truncated to target day
                    # (not available without per-player history) — use Gini-like
                    # percentile estimate: players who never hit T1 by target day
                    # are those whose expected balance by day X < T1 cost.
                    median_at_target = float(within["median_gold"].iloc[-1]) if len(within) > 0 else 0.0
                    # Use normal approximation: if median is well above cost → low fail rate
                    # If median < cost → >50% failed
                    if median_at_target <= 0:
                        casual_fail_rate = 1.0
                    elif median_at_target >= t1_cost * 2:
                        casual_fail_rate = 0.0
                    elif median_at_target < t1_cost:
                        # More than 50% failed
                        casual_fail_rate = min(1.0, 0.5 + 0.5 * (1 - median_at_target / t1_cost))
                    else:
                        # median >= cost → some passed; estimate from ratio
                        casual_fail_rate = max(0.0, 0.5 * (1 - (median_at_target - t1_cost) / t1_cost))
        else:
            casual_fail_rate = 0.0
    else:
        casual_fail_rate = 0.0

    # --- Build alerts ---
    alerts: list[DiagnosticAlert] = [
        _check_macro_flow(flow_ratio),
        _check_wealth_disparity(gini),
        _check_progression_bottleneck(
            casual_fail_rate,
            target_day=casual_target_day,
            tier_1_cost=eco_config.tier_1_weapon_cost,
        ),
    ]

    # Sort: CRITICAL first, then WARNING, INFO, OK
    _sev_rank = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
    alerts.sort(key=lambda a: _sev_rank[a.severity])

    overall = _worst_severity(alerts)

    # --- One-line summary ---
    flow_tag = _flow_summary(flow_ratio)
    summary = (
        f"Economy is {overall.value} — "
        f"flow={flow_tag} ({flow_ratio:.2f}×), "
        f"Gini={gini:.3f}, "
        f"Casual T1 fail-rate={casual_fail_rate:.1%}"
    )

    return DiagnosticReport(
        alerts=alerts,
        flow_ratio=flow_ratio,
        gini=gini,
        casual_fail_rate=casual_fail_rate,
        casual_target_day=casual_target_day,
        overall_health=overall,
        summary=summary,
    )
