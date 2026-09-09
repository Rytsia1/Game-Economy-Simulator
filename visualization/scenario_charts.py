"""
visualization/scenario_charts.py
---------------------------------
Plotly chart builders for Scenario Stress-Testing and Monte Carlo Risk Analysis
(Step 4 / v0.4).

Charts:
1. plot_scenario_comparison    – Multi-line median wealth curves across 4 presets.
2. plot_scenario_metrics_bar   – Comparative grouped bar chart across regimes.
3. plot_monte_carlo_fan_chart  – Quantile fan chart with 50% and 90% confidence bands.
4. plot_risk_distribution      – Histogram of Flow Ratios with critical threshold lines.
5. plot_risk_scatter_or_cdf    – Quadrant scatter plot: Flow Ratio vs. Casual Poverty Rate.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.scenarios import SCENARIO_COLORS
from simulation.monte_carlo import MonteCarloResult

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------

_BG       = "#0F1117"
_SURFACE  = "#1A1D27"
_SURFACE2 = "#22263A"
_GRID     = "#2A2D3A"
_TEXT     = "#E0E4F0"
_TEXT_MUT = "#8892AA"

_ACCENT_BLUE   = "#4F8EF7"
_ACCENT_TEAL   = "#00C9A7"
_ACCENT_PURPLE = "#A78BFA"
_ACCENT_AMBER  = "#FBBF24"
_ACCENT_ROSE   = "#F87171"

_FONT_FAMILY = "Inter, system-ui, sans-serif"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_SURFACE,
    font=dict(family=_FONT_FAMILY, color=_TEXT, size=13),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(
        bgcolor="rgba(26,29,39,0.85)",
        bordercolor=_GRID,
        borderwidth=1,
        font=dict(size=11, color=_TEXT),
    ),
)

_AXIS_STYLE = dict(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    tickcolor=_TEXT,
    linecolor=_GRID,
)


def _base_layout(**overrides) -> dict:
    layout = dict(**_LAYOUT_BASE)
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# 1. Multi-Scenario Wealth Comparison Chart
# ---------------------------------------------------------------------------

def plot_scenario_comparison(progression_df: pd.DataFrame) -> go.Figure:
    """
    Multi-line chart comparing median wealth progression across all scenario presets.

    Parameters
    ----------
    progression_df : pd.DataFrame
        Must contain columns: ['day', 'scenario', 'median_gold']

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if progression_df.empty:
        fig.update_layout(**_base_layout(title="No scenario data available"))
        return fig

    scenarios = progression_df["scenario"].unique()

    for sc in scenarios:
        sub = progression_df[progression_df["scenario"] == sc].sort_values("day")
        color = SCENARIO_COLORS.get(sc, _ACCENT_BLUE)

        fig.add_trace(go.Scatter(
            x=sub["day"],
            y=sub["median_gold"],
            mode="lines",
            name=sc,
            line=dict(color=color, width=2.8),
            hovertemplate=(
                f"<b>{sc}</b><br>"
                "Day: %{x}<br>"
                "Median Gold: %{y:,.0f}g"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="🧪 Scenario Stress-Test: Wealth Trajectories", font=dict(size=18)),
            xaxis=dict(title="Simulation Day", **_AXIS_STYLE),
            yaxis=dict(title="Median Player Wealth (Gold)", **_AXIS_STYLE),
            height=440,
            hovermode="x unified",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Scenario Comparative Metrics Bar Chart
# ---------------------------------------------------------------------------

def plot_scenario_metrics_bar(summary_df: pd.DataFrame) -> go.Figure:
    """
    Comparative bar chart contrasting Flow Ratio, Gini, and Casual Poverty Rate.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Columns: ['Scenario', 'Flow Ratio', 'Gini', 'Casual Fail Rate %']
    """
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(title="No summary metrics available"))
        return fig

    scenarios = summary_df["Scenario"].tolist()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Macro Flow Ratio", "Gini Coefficient", "Casual T1 Fail Rate (%)"),
        shared_yaxes=False,
    )

    # 1. Flow Ratio
    fig.add_trace(go.Bar(
        x=scenarios,
        y=summary_df["Flow Ratio"],
        name="Flow Ratio",
        marker=dict(color=[SCENARIO_COLORS.get(s, _ACCENT_BLUE) for s in scenarios]),
        text=[f"{v:.2f}×" for v in summary_df["Flow Ratio"]],
        textposition="auto",
    ), row=1, col=1)

    # 2. Gini
    fig.add_trace(go.Bar(
        x=scenarios,
        y=summary_df["Gini"],
        name="Gini",
        marker=dict(color=[SCENARIO_COLORS.get(s, _ACCENT_PURPLE) for s in scenarios]),
        text=[f"{v:.3f}" for v in summary_df["Gini"]],
        textposition="auto",
    ), row=1, col=2)

    # 3. Casual Fail Rate %
    fig.add_trace(go.Bar(
        x=scenarios,
        y=summary_df["Casual Fail Rate %"],
        name="Casual Fail %",
        marker=dict(color=[SCENARIO_COLORS.get(s, _ACCENT_ROSE) for s in scenarios]),
        text=[f"{v:.1f}%" for v in summary_df["Casual Fail Rate %"]],
        textposition="auto",
    ), row=1, col=3)

    fig.update_layout(
        **_base_layout(
            title=dict(text="📊 Comparative Macroeconomic Metrics", font=dict(size=18)),
            height=320,
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=50),
        )
    )

    for i in range(1, 4):
        fig.update_xaxes(row=1, col=i, tickangle=-15, **_AXIS_STYLE)
        fig.update_yaxes(row=1, col=i, **_AXIS_STYLE)

    return fig


# ---------------------------------------------------------------------------
# 3. Monte Carlo Confidence Interval Fan Chart
# ---------------------------------------------------------------------------

def plot_monte_carlo_fan_chart(mc_result: MonteCarloResult) -> go.Figure:
    """
    Plotly fan chart showing median trajectory wrapped in 50% and 90% confidence bands.

    Parameters
    ----------
    mc_result : MonteCarloResult

    Returns
    -------
    go.Figure
    """
    pdf = mc_result.percentiles_df
    days = pdf["day"]

    fig = go.Figure()

    # --- 90% CI: p5 to p95 ---
    # Upper bound p95
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["p95"],
        mode="lines",
        line=dict(color="rgba(79, 142, 247, 0.0)", width=0),
        showlegend=False,
        name="p95",
        hoverinfo="skip",
    ))
    # Lower bound p5 with fill to p95
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["p5"],
        mode="lines",
        line=dict(color="rgba(79, 142, 247, 0.0)", width=0),
        fill="tonexty",
        fillcolor="rgba(79, 142, 247, 0.12)",
        name="90% Confidence Interval (p5 – p95)",
        hoverinfo="skip",
    ))

    # --- 50% CI: p25 to p75 ---
    # Upper bound p75
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["p75"],
        mode="lines",
        line=dict(color="rgba(0, 201, 167, 0.0)", width=0),
        showlegend=False,
        name="p75",
        hoverinfo="skip",
    ))
    # Lower bound p25 with fill to p75
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["p25"],
        mode="lines",
        line=dict(color="rgba(0, 201, 167, 0.0)", width=0),
        fill="tonexty",
        fillcolor="rgba(0, 201, 167, 0.22)",
        name="50% Confidence Interval (p25 – p75)",
        hoverinfo="skip",
    ))

    # --- Median trajectory: p50 ---
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["p50"],
        mode="lines",
        line=dict(color=_ACCENT_TEAL, width=3.2),
        name="Median Trajectory (p50)",
        hovertemplate=(
            "<b>Day %{x}</b><br>"
            "Median (p50): %{y:,.0f}g<br>"
            "<extra></extra>"
        ),
    ))

    # --- Mean trajectory (dashed) ---
    fig.add_trace(go.Scatter(
        x=days,
        y=pdf["mean"],
        mode="lines",
        line=dict(color=_ACCENT_BLUE, width=1.6, dash="dot"),
        name="Mean Trajectory",
        hovertemplate="Mean: %{y:,.0f}g<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"🎲 Monte Carlo Fan Chart (M={mc_result.num_runs} Runs, Volatility={mc_result.volatility:.0%})",
                font=dict(size=18),
            ),
            xaxis=dict(title="Simulation Day", **_AXIS_STYLE),
            yaxis=dict(title="Player Wealth (Gold)", **_AXIS_STYLE),
            height=460,
            hovermode="x unified",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Monte Carlo Risk Distribution (Histogram)
# ---------------------------------------------------------------------------

def plot_risk_distribution(mc_result: MonteCarloResult) -> go.Figure:
    """
    Histogram of Flow Ratios across all runs with critical threshold indicators.
    """
    ratios = mc_result.run_summaries["flow_ratio"]
    infl_thresh = mc_result.inflation_thresh

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ratios,
        nbinsx=max(12, min(40, mc_result.num_runs // 3)),
        marker=dict(
            color="rgba(79, 142, 247, 0.70)",
            line=dict(color=_ACCENT_BLUE, width=1),
        ),
        name="Flow Ratio Distribution",
        hovertemplate="Flow Ratio: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    # Inflation threshold vertical line
    fig.add_vline(
        x=infl_thresh,
        line_color=_ACCENT_ROSE,
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"Inflation Gate ({infl_thresh:.1f}×)",
        annotation_position="top right",
        annotation_font=dict(color=_ACCENT_ROSE, size=11),
    )

    # Balanced lower threshold line (0.9)
    fig.add_vline(
        x=0.9,
        line_color=_ACCENT_AMBER,
        line_width=2,
        line_dash="dot",
        annotation_text="Poverty Line (0.9×)",
        annotation_position="top left",
        annotation_font=dict(color=_ACCENT_AMBER, size=11),
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text="📈 Macro Flow Ratio Risk Distribution", font=dict(size=18)),
            xaxis=dict(title="Flow Ratio (Income / Sinks)", **_AXIS_STYLE),
            yaxis=dict(title="Number of Simulated Runs", **_AXIS_STYLE),
            height=340,
            showlegend=False,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Stability Quadrant Scatter Plot
# ---------------------------------------------------------------------------

def plot_risk_scatter_or_cdf(mc_result: MonteCarloResult) -> go.Figure:
    """
    Quadrant scatter plot comparing Flow Ratio vs Casual Poverty Rate per run.
    """
    df = mc_result.run_summaries.copy()
    infl_thresh = mc_result.inflation_thresh
    pov_thresh = mc_result.poverty_thresh * 100

    # Categorize runs
    colors = []
    labels = []
    for _, r in df.iterrows():
        inflated = r["flow_ratio"] >= infl_thresh
        poverty = (r["casual_fail_rate"] * 100) >= pov_thresh
        if not inflated and not poverty:
            colors.append(_ACCENT_TEAL)
            labels.append("Stable")
        elif inflated and not poverty:
            colors.append(_ACCENT_AMBER)
            labels.append("Inflation Risk")
        elif poverty and not inflated:
            colors.append(_ACCENT_ROSE)
            labels.append("Poverty Trap")
        else:
            colors.append("#E11D48")
            labels.append("Compounded Crisis")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["flow_ratio"],
        y=df["casual_fail_rate"] * 100,
        mode="markers",
        marker=dict(
            color=colors,
            size=9,
            line=dict(color="#ffffff", width=0.8),
            opacity=0.85,
        ),
        text=[f"Run #{r['run_id']}<br>Quest: {r['quest_reward']:.1f}g<br>Potion: {r['potion_cost']:.1f}g"
              for _, r in df.iterrows()],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Flow Ratio: %{x:.2f}×<br>"
            "Casual Fail Rate: %{y:.1f}%"
            "<extra></extra>"
        ),
        name="Simulated Runs",
    ))

    # Threshold divider lines
    fig.add_vline(x=infl_thresh, line_color=_GRID, line_width=1.5, line_dash="dash")
    fig.add_hline(y=pov_thresh, line_color=_GRID, line_width=1.5, line_dash="dash")

    fig.update_layout(
        **_base_layout(
            title=dict(text="🎯 System Stability Quadrants", font=dict(size=18)),
            xaxis=dict(title="Flow Ratio (Income / Sinks)", **_AXIS_STYLE),
            yaxis=dict(title="Casual Fail Rate (%)", **_AXIS_STYLE),
            height=340,
            showlegend=False,
        )
    )
    return fig
