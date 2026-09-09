"""
visualization/charts.py
-----------------------
Plotly chart builders for the Game Economy Simulator dashboard.
v0.2 additions: archetype trajectory, violin distribution, milestone bar chart.

Each function returns a fully-configured ``plotly.graph_objects.Figure``
ready to be rendered by Streamlit's ``st.plotly_chart``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import ArchetypeProfile

# ---------------------------------------------------------------------------
# Shared design tokens
# ---------------------------------------------------------------------------

_BG      = "#0F1117"
_SURFACE = "#1A1D27"
_GRID    = "#2A2D3A"
_TEXT    = "#E0E4F0"
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
        bgcolor="rgba(26,29,39,0.8)",
        bordercolor=_GRID,
        borderwidth=1,
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
# 1. Overall Wealth Progression  (Mean vs Median, global)
# ---------------------------------------------------------------------------

def plot_wealth_progression(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Line chart of global Average and Median player gold over simulation days
    with a shaded Min–Max band.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Output of ``run_simulation()["metrics"]``.
    """
    days = df_metrics["day"]

    fig = go.Figure()

    # Shaded band
    fig.add_trace(go.Scatter(
        x=pd.concat([days, days[::-1]]),
        y=pd.concat([df_metrics["max_gold"], df_metrics["min_gold"][::-1]]),
        fill="toself",
        fillcolor="rgba(79,142,247,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Min–Max range",
    ))

    fig.add_trace(go.Scatter(
        x=days, y=df_metrics["avg_gold"],
        mode="lines", name="Mean Gold",
        line=dict(color=_ACCENT_BLUE, width=2.5),
        hovertemplate="Day %{x}<br>Mean: %{y:,.0f} gold<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=days, y=df_metrics["median_gold"],
        mode="lines", name="Median Gold",
        line=dict(color=_ACCENT_TEAL, width=2.5, dash="dot"),
        hovertemplate="Day %{x}<br>Median: %{y:,.0f} gold<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title=dict(text="Wealth Progression Over Time", font=dict(size=18))),
        xaxis=dict(title="Day", **_AXIS_STYLE),
        yaxis=dict(title="Gold Balance", **_AXIS_STYLE, tickformat=","),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Overall Wealth Distribution  (Histogram + KDE)
# ---------------------------------------------------------------------------

def plot_wealth_distribution(final_balances: np.ndarray) -> go.Figure:
    """
    Histogram of global player wealth at the final day with KDE overlay.

    Parameters
    ----------
    final_balances : np.ndarray, shape (num_players,)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Histogram(
        x=final_balances,
        name="Player Count",
        nbinsx=50,
        marker=dict(color=_ACCENT_PURPLE, opacity=0.75,
                    line=dict(width=0.4, color=_GRID)),
        hovertemplate="Gold: %{x:,.0f}<br>Players: %{y}<extra></extra>",
    ), secondary_y=False)

    kde_x, kde_y = _gaussian_kde(final_balances, n_points=400)
    fig.add_trace(go.Scatter(
        x=kde_x, y=kde_y,
        name="Density", mode="lines",
        line=dict(color=_ACCENT_AMBER, width=2.5),
        hovertemplate="Gold: %{x:,.0f}<br>Density: %{y:.4f}<extra></extra>",
    ), secondary_y=True)

    mean_val   = float(np.mean(final_balances))
    median_val = float(np.median(final_balances))

    for val, label, color in [
        (mean_val, "Mean", _ACCENT_BLUE),
        (median_val, "Median", _ACCENT_TEAL),
    ]:
        fig.add_vline(x=val, line_width=1.8, line_dash="dash", line_color=color,
                      annotation_text=f"{label}: {val:,.0f}",
                      annotation_font_color=color,
                      annotation_position="top right")

    fig.update_layout(
        **_base_layout(title=dict(text="Final Day Wealth Distribution", font=dict(size=18))),
        xaxis=dict(title="Gold Balance", **_AXIS_STYLE, tickformat=","),
        yaxis=dict(title="Number of Players", **_AXIS_STYLE),
        yaxis2=dict(title="Density", **_AXIS_STYLE, showgrid=False),
        barmode="overlay",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Inflow / Outflow Balance
# ---------------------------------------------------------------------------

def plot_inflow_outflow(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Dual-panel chart: cumulative inflow vs outflow (top) and daily net flow (bottom).

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Output of ``run_simulation()["metrics"]``.
    """
    days = df_metrics["day"]
    net  = df_metrics["net_flow_day"]
    net_colors = [_ACCENT_TEAL if v >= 0 else _ACCENT_ROSE for v in net]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=("Cumulative Gold Flow", "Daily Net Flow"),
    )

    fig.add_trace(go.Scatter(
        x=days, y=df_metrics["cumulative_earned"],
        name="Cumulative Inflow", fill="tozeroy",
        fillcolor="rgba(0,201,167,0.15)",
        line=dict(color=_ACCENT_TEAL, width=2),
        hovertemplate="Day %{x}<br>Cumulative Earned: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=days, y=df_metrics["cumulative_spent"],
        name="Cumulative Outflow", fill="tozeroy",
        fillcolor="rgba(248,113,113,0.15)",
        line=dict(color=_ACCENT_ROSE, width=2),
        hovertemplate="Day %{x}<br>Cumulative Spent: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=days, y=net,
        name="Net Flow", marker_color=net_colors,
        hovertemplate="Day %{x}<br>Net: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(title=dict(text="Gold Inflow vs Outflow", font=dict(size=18))),
        xaxis2=dict(title="Day", **_AXIS_STYLE),
        yaxis=dict(title="Gold (total, all players)", **_AXIS_STYLE, tickformat=","),
        yaxis2=dict(title="Net Gold", **_AXIS_STYLE, tickformat=","),
        hovermode="x unified", showlegend=True,
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size = 14
    for ax in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
        fig.layout[ax].update(_AXIS_STYLE)

    return fig


# ---------------------------------------------------------------------------
# 4. Archetype Median Wealth Trajectory  (v0.2 NEW)
# ---------------------------------------------------------------------------

def plot_archetype_progression(
    archetype_metrics: pd.DataFrame,
    archetypes: List[ArchetypeProfile],
) -> go.Figure:
    """
    Multi-line chart showing the median gold trajectory over time for each
    player archetype.

    Parameters
    ----------
    archetype_metrics : pd.DataFrame
        Long-format DataFrame with columns: day | archetype | median_gold.
        Output of ``run_simulation()["archetype_metrics"]``.
    archetypes : List[ArchetypeProfile]
        Archetype definitions (for color lookup and ordering).
    """
    color_map = {a.name: a.color for a in archetypes}
    arch_order = [a.name for a in archetypes]

    fig = go.Figure()

    for arch_name in arch_order:
        subset = archetype_metrics[archetype_metrics["archetype"] == arch_name]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["day"],
            y=subset["median_gold"],
            mode="lines",
            name=arch_name,
            line=dict(color=color_map.get(arch_name, "#FFFFFF"), width=2.5),
            hovertemplate=f"<b>{arch_name}</b><br>Day %{{x}}<br>Median Gold: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title=dict(text="Archetype Wealth Trajectories (Median)", font=dict(size=18))),
        xaxis=dict(title="Day", **_AXIS_STYLE),
        yaxis=dict(title="Median Gold Balance", **_AXIS_STYLE, tickformat=","),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Archetype Wealth Distribution — Violin Plot  (v0.2 NEW)
# ---------------------------------------------------------------------------

def plot_archetype_distribution(
    final_balances: np.ndarray,
    archetype_ids: np.ndarray,
    archetypes: List[ArchetypeProfile],
) -> go.Figure:
    """
    Violin plot of final-day wealth distribution segmented by archetype,
    with an overlaid box plot for quartile visibility.

    Parameters
    ----------
    final_balances : np.ndarray, shape (P,)
    archetype_ids  : np.ndarray, shape (P,), integer archetype index per player.
    archetypes     : List[ArchetypeProfile]
    """
    fig = go.Figure()

    for a_idx, arch in enumerate(archetypes):
        mask = archetype_ids == a_idx
        if not mask.any():
            continue
        vals = final_balances[mask]
        fig.add_trace(go.Violin(
            y=vals,
            name=arch.name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=_hex_to_rgba(arch.color, 0.25),
            line_color=arch.color,
            marker=dict(color=arch.color, size=2, opacity=0.4),
            points="outliers",
            hovertemplate=(
                f"<b>{arch.name}</b><br>"
                "Gold: %{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(title=dict(text="Final Day Wealth by Archetype", font=dict(size=18))),
        yaxis=dict(title="Gold Balance (Final Day)", **_AXIS_STYLE, tickformat=","),
        xaxis=dict(**_AXIS_STYLE),
        violingap=0.15,
        violinmode="overlay",
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Time-to-Afford Milestones — Bar Chart  (v0.2 NEW)
# ---------------------------------------------------------------------------

def plot_affordability_milestones(
    time_to_afford: Dict[str, dict],
    archetypes: List[ArchetypeProfile],
    tier_1_cost: int,
    tier_2_cost: int,
) -> go.Figure:
    """
    Grouped bar chart: days required for 50% of each archetype to afford
    Tier 1 and Tier 2 gear milestones.

    Parameters
    ----------
    time_to_afford : dict  {archetype_name: {"tier_1": int|None, "tier_2": int|None}}
    archetypes     : List[ArchetypeProfile]
    tier_1_cost    : int, gold cost of Tier 1 milestone (label only).
    tier_2_cost    : int, gold cost of Tier 2 milestone (label only).
    """
    arch_names  = [a.name for a in archetypes]
    t1_days: List[float] = []
    t2_days: List[float] = []

    for name in arch_names:
        d = time_to_afford.get(name, {})
        t1_days.append(d.get("tier_1") or 0)
        t2_days.append(d.get("tier_2") or 0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"Tier 1 ({tier_1_cost:,}g)",
        x=arch_names,
        y=t1_days,
        marker_color=[a.color for a in archetypes],
        marker_opacity=0.9,
        text=[f"Day {int(v)}" if v > 0 else "Never" for v in t1_days],
        textposition="outside",
        textfont=dict(color=_TEXT, size=12),
        hovertemplate="<b>%{x}</b><br>Tier 1 affordable by: Day %{y}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name=f"Tier 2 ({tier_2_cost:,}g)",
        x=arch_names,
        y=t2_days,
        marker_color=[a.color for a in archetypes],
        marker_opacity=0.45,
        marker_pattern_shape="/",
        text=[f"Day {int(v)}" if v > 0 else "Never" for v in t2_days],
        textposition="outside",
        textfont=dict(color=_TEXT, size=12),
        hovertemplate="<b>%{x}</b><br>Tier 2 affordable by: Day %{y}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title=dict(text="Days Until 50% of Archetype Can Afford Gear", font=dict(size=18))),
        xaxis=dict(title="Archetype", **_AXIS_STYLE),
        yaxis=dict(title="Day", **_AXIS_STYLE),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_kde(data: np.ndarray, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a Gaussian KDE on *data* using Silverman's bandwidth.
    Pure NumPy — no SciPy dependency.
    """
    n = data.size
    if n < 2:
        return np.array([data[0]]), np.array([1.0])

    std = data.std(ddof=1)
    if std == 0:
        x = np.linspace(data.min() - 1, data.max() + 1, n_points)
        return x, np.zeros(n_points)

    bw = 1.06 * std * n ** (-0.2)
    x  = np.linspace(data.min() - 3 * bw, data.max() + 3 * bw, n_points)
    z  = (x[:, np.newaxis] - data[np.newaxis, :]) / bw
    y  = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return x, y


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# 7. Diagnostic Health Gauges  (v0.3 NEW)
# ---------------------------------------------------------------------------

def plot_diagnostic_gauges(
    flow_ratio: float,
    gini: float,
    casual_fail_rate: float,
) -> go.Figure:
    """
    Three Plotly Indicator gauges side-by-side visualising the three key
    diagnostic metrics: Flow Ratio, Gini Coefficient, Casual Fail Rate.

    Parameters
    ----------
    flow_ratio        : Income / Sink cumulative ratio.
    gini              : Final-day Gini coefficient.
    casual_fail_rate  : Fraction of Casual players who missed Tier-1 target.

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Flow Ratio (Income/Sink)", "Gini Coefficient", "Casual T1 Fail Rate"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
    )

    # ── Flow ratio gauge ─────────────────────────────────
    # Good zone: 0.9–1.3. Red zones on both ends.
    if flow_ratio >= 2.0:
        flow_color = _ACCENT_ROSE
    elif flow_ratio >= 1.3:
        flow_color = _ACCENT_AMBER
    elif flow_ratio >= 0.9:
        flow_color = _ACCENT_TEAL
    else:
        flow_color = _ACCENT_ROSE

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=round(flow_ratio, 3),
        delta={"reference": 1.1, "relative": False, "valueformat": ".2f"},
        number={"valueformat": ".2f", "suffix": "×", "font": {"color": flow_color}},
        gauge={
            "axis": {"range": [0, 3.0], "tickformat": ".1f",
                     "tickcolor": _TEXT, "tickfont": {"color": _TEXT}},
            "bar": {"color": flow_color, "thickness": 0.25},
            "bgcolor": _SURFACE,
            "bordercolor": _GRID,
            "steps": [
                {"range": [0, 0.9],   "color": "rgba(248,113,113,0.15)"},
                {"range": [0.9, 1.3], "color": "rgba(0,201,167,0.12)"},
                {"range": [1.3, 2.0], "color": "rgba(251,191,36,0.12)"},
                {"range": [2.0, 3.0], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {
                "line": {"color": _ACCENT_TEAL, "width": 2},
                "thickness": 0.8,
                "value": 1.1,
            },
        },
    ), row=1, col=1)

    # ── Gini gauge ────────────────────────────────────────
    gini_color = _ACCENT_ROSE if gini >= 0.55 else (_ACCENT_AMBER if gini >= 0.40 else _ACCENT_TEAL)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(gini, 4),
        number={"valueformat": ".4f", "font": {"color": gini_color}},
        gauge={
            "axis": {"range": [0, 1.0], "tickformat": ".2f",
                     "tickcolor": _TEXT, "tickfont": {"color": _TEXT}},
            "bar": {"color": gini_color, "thickness": 0.25},
            "bgcolor": _SURFACE,
            "bordercolor": _GRID,
            "steps": [
                {"range": [0, 0.40],  "color": "rgba(0,201,167,0.12)"},
                {"range": [0.40, 0.55], "color": "rgba(251,191,36,0.12)"},
                {"range": [0.55, 1.0], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {
                "line": {"color": _ACCENT_AMBER, "width": 2},
                "thickness": 0.8,
                "value": 0.55,
            },
        },
    ), row=1, col=2)

    # ── Casual fail-rate gauge ────────────────────────────
    fail_pct = casual_fail_rate * 100
    fail_color = _ACCENT_ROSE if casual_fail_rate > 0.35 else (_ACCENT_AMBER if casual_fail_rate > 0.15 else _ACCENT_TEAL)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=round(fail_pct, 1),
        number={"valueformat": ".1f", "suffix": "%", "font": {"color": fail_color}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickcolor": _TEXT, "tickfont": {"color": _TEXT}},
            "bar": {"color": fail_color, "thickness": 0.25},
            "bgcolor": _SURFACE,
            "bordercolor": _GRID,
            "steps": [
                {"range": [0, 15],   "color": "rgba(0,201,167,0.12)"},
                {"range": [15, 35],  "color": "rgba(251,191,36,0.12)"},
                {"range": [35, 100], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {
                "line": {"color": _ACCENT_ROSE, "width": 2},
                "thickness": 0.8,
                "value": 35,
            },
        },
    ), row=1, col=3)

    fig.update_layout(
        **_base_layout(
            title=dict(text="Economy Health Gauges", font=dict(size=18)),
            height=300,
            margin=dict(l=30, r=30, t=70, b=20),
        ),
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size  = 13

    return fig


# ---------------------------------------------------------------------------
# 8. Tuner Convergence Chart  (v0.3 NEW)
# ---------------------------------------------------------------------------

def plot_tuner_convergence(probes: list, target_day: int, param_name: str) -> go.Figure:
    """
    Scatter + line chart showing the binary-search probe history:
    iteration vs achieved milestone day, with the target day as a
    reference line and the signed error coloured red/green.

    Parameters
    ----------
    probes      : List of ``TunerProbe`` objects from ``TunerResult.probes``.
    target_day  : Designer's desired milestone day D*.
    param_name  : The parameter being tuned (for axis labels).

    Returns
    -------
    go.Figure
    """
    iterations   = [p.iteration for p in probes]
    achieved     = [(p.achieved_day if p.achieved_day is not None else None) for p in probes]
    param_values = [p.param_value for p in probes]
    objectives   = [p.objective for p in probes]

    # Replace None achieved days with a sentinel for plotting
    plot_days = [d if d is not None else (target_day + 30) for d in achieved]
    colors     = [_ACCENT_TEAL if abs(o) <= 1 else (_ACCENT_AMBER if abs(o) <= 5 else _ACCENT_ROSE)
                  for o in objectives]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.08,
        subplot_titles=("Milestone Day per Probe", f"{param_name} Tested per Probe"),
    )

    # Row 1 – achieved day scatter
    fig.add_hline(
        y=target_day, line_color=_ACCENT_TEAL,
        line_dash="dash", line_width=1.8,
        annotation_text=f"Target: Day {target_day}",
        annotation_font_color=_ACCENT_TEAL,
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(
        x=iterations, y=plot_days,
        mode="lines+markers",
        name="Achieved Day",
        line=dict(color=_ACCENT_BLUE, width=1.8),
        marker=dict(color=colors, size=9, line=dict(width=1, color=_GRID)),
        hovertemplate="Iter %{x}<br>Day: %{y}<extra></extra>",
    ), row=1, col=1)

    # Row 2 – param value tested
    fig.add_trace(go.Scatter(
        x=iterations, y=param_values,
        mode="lines+markers",
        name=param_name,
        line=dict(color=_ACCENT_PURPLE, width=1.8, dash="dot"),
        marker=dict(color=_ACCENT_PURPLE, size=7),
        hovertemplate=f"Iter %{{x}}<br>{param_name}: %{{y:.1f}}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=dict(text=f"Auto-Tuner Convergence — {param_name}", font=dict(size=18))
        ),
        xaxis2=dict(title="Iteration", **_AXIS_STYLE),
        yaxis=dict(title="Milestone Day", **_AXIS_STYLE),
        yaxis2=dict(title=f"{param_name} (gold)", **_AXIS_STYLE),
        hovermode="x unified",
    )
    for ann in fig.layout.annotations:
        ann.font.color = _TEXT
        ann.font.size  = 13
    for ax in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
        fig.layout[ax].update(_AXIS_STYLE)

    return fig


# ---------------------------------------------------------------------------
# 9. Tuner Sensitivity Sweep  (v0.3 NEW)
# ---------------------------------------------------------------------------

def plot_tuner_sensitivity(
    sweep_values: list[float],
    achieved_days: list[int | None],
    param_name: str,
    target_day: int,
    archetype_name: str,
) -> go.Figure:
    """
    Line chart showing how the milestone day responds to varying the
    tuned parameter across its full search range.  Used to communicate
    the sensitivity of the economy to that lever.

    Parameters
    ----------
    sweep_values   : Ordered parameter values tested.
    achieved_days  : Corresponding milestone day for each value (None = never).
    param_name     : Parameter name for axis labels.
    target_day     : Target day reference line.
    archetype_name : Archetype name for labelling.

    Returns
    -------
    go.Figure
    """
    plot_days = [d if d is not None else None for d in achieved_days]
    # Filter out None for the line (show gaps instead)
    x_vals, y_vals = [], []
    for xv, yv in zip(sweep_values, plot_days):
        x_vals.append(xv)
        y_vals.append(yv)

    fig = go.Figure()

    fig.add_hline(
        y=target_day,
        line_color=_ACCENT_TEAL,
        line_dash="dash",
        line_width=2,
        annotation_text=f"Target: Day {target_day}",
        annotation_font_color=_ACCENT_TEAL,
        annotation_position="bottom right",
    )

    # Colour the line segment: green where close to target, amber otherwise
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name=f"{archetype_name} Milestone Day",
        line=dict(color=_ACCENT_BLUE, width=2.5),
        marker=dict(
            color=[
                _ACCENT_TEAL if (d is not None and abs(d - target_day) <= 2) else _ACCENT_AMBER
                for d in achieved_days
            ],
            size=7,
        ),
        connectgaps=False,
        hovertemplate=f"{param_name}: %{{x:.1f}}<br>Milestone: Day %{{y}}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Sensitivity: {archetype_name} T1 Day vs {param_name}",
                font=dict(size=18),
            )
        ),
        xaxis=dict(title=f"{param_name} (gold)", **_AXIS_STYLE),
        yaxis=dict(title="Milestone Day", **_AXIS_STYLE),
        hovermode="x unified",
    )
    return fig
