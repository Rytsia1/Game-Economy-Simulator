"""
visualization/charts.py
-----------------------
Plotly chart builders for the Game Economy Simulator dashboard.

Each function accepts processed data (DataFrames / NumPy arrays) and returns
a fully-configured ``plotly.graph_objects.Figure`` ready to be rendered by
Streamlit's ``st.plotly_chart``.

Design tokens
~~~~~~~~~~~~~
All charts share a consistent dark-themed palette and typography so the
dashboard feels cohesive without importing a shared CSS file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared design tokens
# ---------------------------------------------------------------------------

_BG = "#0F1117"          # deep charcoal – matches Streamlit dark background
_SURFACE = "#1A1D27"     # slightly lighter surface for chart area
_GRID = "#2A2D3A"        # subtle grid lines
_TEXT = "#E0E4F0"        # near-white labels
_ACCENT_BLUE = "#4F8EF7"
_ACCENT_TEAL = "#00C9A7"
_ACCENT_PURPLE = "#A78BFA"
_ACCENT_AMBER = "#FBBF24"
_ACCENT_ROSE = "#F87171"

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
# 1. Wealth Progression  (Mean vs Median over time)
# ---------------------------------------------------------------------------

def plot_wealth_progression(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Line chart of Average and Median player gold over simulation days.

    Also renders a shaded Min–Max band so the spread of player outcomes is
    immediately visible.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Output of ``run_simulation()["metrics"]``.

    Returns
    -------
    go.Figure
    """
    days = df_metrics["day"]

    fig = go.Figure()

    # --- Min-Max shaded band ---
    fig.add_trace(
        go.Scatter(
            x=pd.concat([days, days[::-1]]),
            y=pd.concat([df_metrics["max_gold"], df_metrics["min_gold"][::-1]]),
            fill="toself",
            fillcolor="rgba(79,142,247,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Min–Max range",
            showlegend=True,
        )
    )

    # --- Mean line ---
    fig.add_trace(
        go.Scatter(
            x=days,
            y=df_metrics["avg_gold"],
            mode="lines",
            name="Mean Gold",
            line=dict(color=_ACCENT_BLUE, width=2.5),
            hovertemplate="Day %{x}<br>Mean: %{y:,.0f} gold<extra></extra>",
        )
    )

    # --- Median line ---
    fig.add_trace(
        go.Scatter(
            x=days,
            y=df_metrics["median_gold"],
            mode="lines",
            name="Median Gold",
            line=dict(color=_ACCENT_TEAL, width=2.5, dash="dot"),
            hovertemplate="Day %{x}<br>Median: %{y:,.0f} gold<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(title=dict(text="💰 Wealth Progression Over Time", font=dict(size=18))),
        xaxis=dict(title="Day", **_AXIS_STYLE),
        yaxis=dict(title="Gold Balance", **_AXIS_STYLE, tickformat=","),
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Wealth Distribution  (Histogram + KDE at final day)
# ---------------------------------------------------------------------------

def plot_wealth_distribution(final_balances: np.ndarray) -> go.Figure:
    """
    Histogram of player gold balances at the final simulation day with an
    overlaid smooth KDE curve.

    Parameters
    ----------
    final_balances : np.ndarray, shape (num_players,)

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Histogram ---
    fig.add_trace(
        go.Histogram(
            x=final_balances,
            name="Player Count",
            nbinsx=50,
            marker=dict(
                color=_ACCENT_PURPLE,
                opacity=0.75,
                line=dict(width=0.4, color=_GRID),
            ),
            hovertemplate="Gold: %{x:,.0f}<br>Players: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    # --- KDE via simple Gaussian kernel ---
    kde_x, kde_y = _gaussian_kde(final_balances, n_points=400)
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            name="Density",
            mode="lines",
            line=dict(color=_ACCENT_AMBER, width=2.5),
            hovertemplate="Gold: %{x:,.0f}<br>Density: %{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Vertical mean / median lines
    mean_val = float(np.mean(final_balances))
    median_val = float(np.median(final_balances))
    y_max = int(np.histogram(final_balances, bins=50)[0].max())

    for val, label, color in [
        (mean_val, "Mean", _ACCENT_BLUE),
        (median_val, "Median", _ACCENT_TEAL),
    ]:
        fig.add_vline(
            x=val,
            line_width=1.8,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: {val:,.0f}",
            annotation_font_color=color,
            annotation_position="top right",
        )

    fig.update_layout(
        **_base_layout(title=dict(text="📊 Final Day Wealth Distribution", font=dict(size=18))),
        xaxis=dict(title="Gold Balance", **_AXIS_STYLE, tickformat=","),
        yaxis=dict(title="Number of Players", **_AXIS_STYLE),
        yaxis2=dict(title="Density", **_AXIS_STYLE, showgrid=False),
        barmode="overlay",
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Inflow / Outflow Balance  (Cumulative area + daily bar)
# ---------------------------------------------------------------------------

def plot_inflow_outflow(df_metrics: pd.DataFrame) -> go.Figure:
    """
    Dual-panel chart:
    - Top: Cumulative gold inflow vs outflow (filled area chart).
    - Bottom: Daily net flow bar chart (green = positive, red = negative).

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Output of ``run_simulation()["metrics"]``.

    Returns
    -------
    go.Figure
    """
    days = df_metrics["day"]
    net = df_metrics["net_flow_day"]
    net_colors = [_ACCENT_TEAL if v >= 0 else _ACCENT_ROSE for v in net]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Cumulative Gold Flow", "Daily Net Flow"),
    )

    # --- Cumulative inflow ---
    fig.add_trace(
        go.Scatter(
            x=days,
            y=df_metrics["cumulative_earned"],
            name="Cumulative Inflow",
            fill="tozeroy",
            fillcolor="rgba(0,201,167,0.15)",
            line=dict(color=_ACCENT_TEAL, width=2),
            hovertemplate="Day %{x}<br>Cumulative Earned: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Cumulative outflow ---
    fig.add_trace(
        go.Scatter(
            x=days,
            y=df_metrics["cumulative_spent"],
            name="Cumulative Outflow",
            fill="tozeroy",
            fillcolor="rgba(248,113,113,0.15)",
            line=dict(color=_ACCENT_ROSE, width=2),
            hovertemplate="Day %{x}<br>Cumulative Spent: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # --- Daily net flow bars ---
    fig.add_trace(
        go.Bar(
            x=days,
            y=net,
            name="Net Flow",
            marker_color=net_colors,
            hovertemplate="Day %{x}<br>Net: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **_base_layout(title=dict(text="⚖️ Gold Inflow vs Outflow", font=dict(size=18))),
        xaxis2=dict(title="Day", **_AXIS_STYLE),
        yaxis=dict(title="Gold (total, all players)", **_AXIS_STYLE, tickformat=","),
        yaxis2=dict(title="Net Gold", **_AXIS_STYLE, tickformat=","),
        hovermode="x unified",
        showlegend=True,
    )

    # Style subplot title annotations
    for annotation in fig.layout.annotations:
        annotation.font.color = _TEXT
        annotation.font.size = 14

    # Apply axis styles to both subplots
    for axis in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
        fig.layout[axis].update(_AXIS_STYLE)

    return fig


# ---------------------------------------------------------------------------
# Internal helper – Gaussian KDE
# ---------------------------------------------------------------------------

def _gaussian_kde(data: np.ndarray, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a Gaussian KDE on *data* at *n_points* evenly spaced locations.

    Uses Silverman's rule-of-thumb bandwidth.  Pure NumPy — no SciPy
    dependency required.
    """
    n = data.size
    if n < 2:
        return np.array([data[0]]), np.array([1.0])

    std = data.std(ddof=1)
    if std == 0:
        # All values identical
        x = np.linspace(data.min() - 1, data.max() + 1, n_points)
        y = np.zeros(n_points)
        return x, y

    # Silverman bandwidth
    bw = 1.06 * std * n ** (-0.2)

    x_min = data.min() - 3 * bw
    x_max = data.max() + 3 * bw
    x = np.linspace(x_min, x_max, n_points)

    # Vectorised: shape [n_points, n] → sum over n axis
    z = (x[:, np.newaxis] - data[np.newaxis, :]) / bw
    y = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))

    return x, y
