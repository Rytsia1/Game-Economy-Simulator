"""
visualization/health_cards.py
------------------------------
Streamlit rendering helpers for Economy Doctor UI components.

v0.3: Provides high-level functions that render opinionated, styled
Streamlit blocks for diagnostic health banners, callout alert cards,
delta-comparison widgets, and before/after KPI comparisons.

All functions call ``st.*`` directly and return None.
They are intentionally kept as thin presentation wrappers so that
``app.py`` stays clean and the logic lives in ``analytics/``.

Usage
~~~~~
    from visualization.health_cards import (
        render_health_banner,
        render_alert_cards,
        render_tuner_delta,
        render_before_after_kpis,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Internal design tokens (mirrored from app.py CSS variables)
# ---------------------------------------------------------------------------

_SEVERITY_STYLES: dict[str, dict] = {
    "CRITICAL": {
        "color":    "#F87171",
        "bg":       "rgba(248,113,113,0.10)",
        "border":   "#F87171",
    },
    "WARNING": {
        "color":    "#FBBF24",
        "bg":       "rgba(251,191,36,0.10)",
        "border":   "#FBBF24",
    },
    "INFO": {
        "color":    "#4F8EF7",
        "bg":       "rgba(79,142,247,0.08)",
        "border":   "#4F8EF7",
    },
    "OK": {
        "color":    "#00C9A7",
        "bg":       "rgba(0,201,167,0.08)",
        "border":   "#00C9A7",
    },
}

_OVERALL_LABELS: dict[str, str] = {
    "CRITICAL": "Economy at Risk",
    "WARNING":  "Economy Needs Attention",
    "INFO":     "Economy Informational",
    "OK":       "Economy Healthy",
}

_CSS = """
<style>
/* Alert callout card */
.hc-alert-card {
    border-left: 4px solid;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    line-height: 1.5;
}
.hc-alert-title  { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.2rem; }
.hc-alert-detail { font-size: 0.78rem; color: #8892AA; margin-bottom: 0.35rem; }
.hc-alert-fix    { font-size: 0.78rem; border-top: 1px solid #2A2D3A;
                   padding-top: 0.3rem; color: #E0E4F0; }
.hc-alert-fix::before { content: "Fix: "; font-weight: 600; }

/* Delta comparison row */
.hc-delta-row {
    display: flex; align-items: center; gap: 0.75rem;
    background: #1A1D27; border: 1px solid #2A2D3A;
    border-radius: 10px; padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
}
.hc-delta-label  { font-size: 0.72rem; text-transform: uppercase;
                   letter-spacing: 0.08em; color: #8892AA; min-width: 140px; }
.hc-delta-old    { font-size: 1rem; font-weight: 600; color: #F87171; min-width: 120px; }
.hc-delta-arrow  { font-size: 1.2rem; color: #4F8EF7; }
.hc-delta-new    { font-size: 1rem; font-weight: 700; color: #00C9A7; min-width: 120px; }
.hc-delta-badge  { font-size: 0.72rem; border-radius: 999px; padding: 0.2rem 0.6rem;
                   background: rgba(0,201,167,0.15); color: #00C9A7;
                   border: 1px solid rgba(0,201,167,0.3); white-space: nowrap; }

/* Before/After KPI strip */
.hc-ba-strip {
    display: flex; gap: 1rem; margin-bottom: 0.5rem;
}
.hc-ba-card {
    flex: 1; background: #22263A; border: 1px solid #2A2D3A;
    border-radius: 10px; padding: 0.8rem 1rem;
}
.hc-ba-tag   { font-size: 0.68rem; text-transform: uppercase;
               letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.25rem; }
.hc-ba-value { font-size: 1.55rem; font-weight: 700; line-height: 1.1; }
.hc-ba-sub   { font-size: 0.72rem; color: #8892AA; margin-top: 0.15rem; }
.hc-tag-before { color: #F87171; }
.hc-tag-after  { color: #00C9A7; }
</style>
"""


def _inject_css() -> None:
    """Inject health-card CSS once per Streamlit run (idempotent due to Streamlit dedup)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. Diagnostic Health Banner
# ---------------------------------------------------------------------------

def render_health_banner(overall_health: str, summary: str) -> None:
    """
    Render a top-of-page diagnostic summary banner.

    Parameters
    ----------
    overall_health : str  One of "OK", "INFO", "WARNING", "CRITICAL".
    summary        : str  One-line economy summary from DiagnosticReport.
    """
    _inject_css()
    label   = _OVERALL_LABELS.get(overall_health, overall_health)
    style   = _SEVERITY_STYLES.get(overall_health, _SEVERITY_STYLES["INFO"])
    color   = style["color"]
    bg      = style["bg"]
    border  = style["border"]
    st.markdown(
        f"""<div style="border-left: 4px solid {border}; background: {bg}; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: {color}; font-size: 1rem; margin-bottom: 0.2rem;">{label}</div>
            <div style="color: #E0E4F0; font-size: 0.85rem;">{summary}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 2. Diagnostic Alert Callout Cards
# ---------------------------------------------------------------------------

def render_alert_cards(alerts: list) -> None:
    """
    Render a styled HTML card for each ``DiagnosticAlert`` in ``alerts``.

    Each card shows: severity badge, title, quantitative detail, and
    the actionable design recommendation.

    Parameters
    ----------
    alerts : List of DiagnosticAlert objects (from analytics.diagnostics).
    """
    _inject_css()
    for alert in alerts:
        sev_key = alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity)
        style   = _SEVERITY_STYLES.get(sev_key, _SEVERITY_STYLES["INFO"])
        color   = style["color"]
        bg      = style["bg"]

        st.markdown(
            f"""<div class="hc-alert-card"
                     style="border-left-color:{color}; background:{bg};">
                <div class="hc-alert-title" style="color:{color};">
                    [{sev_key}] {alert.title}
                </div>
                <div class="hc-alert-detail">{alert.detail}</div>
                <div class="hc-alert-fix">{alert.recommendation}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# 3. Tuner Delta Comparison
# ---------------------------------------------------------------------------

def render_tuner_delta(
    param_name:    str,
    old_value:     float,
    new_value:     float,
    old_day:       Optional[int],
    new_day:       Optional[int],
    target_day:    int,
    converged:     bool,
) -> None:
    """
    Render a "Current → Recommended" delta comparison widget.

    Shows:
      - Parameter delta: Current 50 Gold -> Recommended 68 Gold (+36%)
      - Milestone delta: Was Day 28 -> Now Day 20 (target achieved)

    Parameters
    ----------
    param_name  : Economy parameter that was tuned.
    old_value   : Original parameter value before tuning.
    new_value   : Recommended (tuned) parameter value.
    old_day     : Milestone day before tuning (None = never reached).
    new_day     : Milestone day after tuning (None = never reached).
    target_day  : Designer target day D*.
    converged   : Whether the tuner converged within tolerance.
    """
    _inject_css()

    # Parameter delta
    delta_pct = ((new_value - old_value) / old_value * 100) if old_value != 0 else float("inf")
    delta_sign = "+" if delta_pct >= 0 else ""
    param_label = param_name.replace("_", " ").title()

    old_day_str = f"Day {old_day}" if old_day else "Never"
    new_day_str = f"Day {new_day}" if new_day else "Never"

    if new_day is not None and new_day <= target_day:
        milestone_badge = '<span class="hc-delta-badge">Target hit</span>'
    elif converged:
        milestone_badge = f'<span class="hc-delta-badge" style="color:#FBBF24;background:rgba(251,191,36,0.12);border-color:rgba(251,191,36,0.3);">±{abs((new_day or 0) - target_day)}d residual</span>'
    else:
        milestone_badge = '<span class="hc-delta-badge" style="color:#F87171;background:rgba(248,113,113,0.12);border-color:rgba(248,113,113,0.3);">Not converged</span>'

    st.markdown(
        f"""<div class="hc-delta-row">
            <div class="hc-delta-label">{param_label}</div>
            <div class="hc-delta-old">{old_value:.2f} g</div>
            <div class="hc-delta-arrow">→</div>
            <div class="hc-delta-new">{new_value:.2f} g</div>
            <div class="hc-delta-badge">{delta_sign}{delta_pct:.1f}%</div>
        </div>
        <div class="hc-delta-row">
            <div class="hc-delta-label">Milestone Day</div>
            <div class="hc-delta-old">{old_day_str}</div>
            <div class="hc-delta-arrow">→</div>
            <div class="hc-delta-new">{new_day_str}</div>
            {milestone_badge}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 4. Before / After KPI Strip
# ---------------------------------------------------------------------------

def render_before_after_kpis(
    metrics: list[dict],
) -> None:
    """
    Render a horizontal strip of before/after KPI comparison cards.

    Parameters
    ----------
    metrics : List of dicts with keys:
        - label       : str  — metric name
        - before_value: str  — formatted value before tuning
        - after_value : str  — formatted value after tuning
        - before_sub  : str  — optional subtitle before (e.g. "Baseline")
        - after_sub   : str  — optional subtitle after  (e.g. "Tuned")
    """
    _inject_css()
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                f"""<div class="hc-ba-strip">
                    <div class="hc-ba-card">
                        <div class="hc-ba-tag hc-tag-before">Before</div>
                        <div class="hc-ba-value" style="color:#F87171;">{m['before_value']}</div>
                        <div class="hc-ba-sub">{m.get('before_sub', m['label'])}</div>
                    </div>
                    <div class="hc-ba-card">
                        <div class="hc-ba-tag hc-tag-after">After</div>
                        <div class="hc-ba-value" style="color:#00C9A7;">{m['after_value']}</div>
                        <div class="hc-ba-sub">{m.get('after_sub', m['label'])}</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# 5. Section Divider Helper
# ---------------------------------------------------------------------------

def section_title(text: str) -> None:
    """Render a styled section title consistent with the app's design system."""
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:600;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:#8892AA;margin:1.5rem 0 0.6rem 0;"
        f"padding-bottom:0.4rem;border-bottom:1px solid #2A2D3A;'>{text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 6. Notification Box Helper
# ---------------------------------------------------------------------------

def render_notification_box(message: str, level: str = "info") -> None:
    """
    Render a clean styled notification container without icons or circle graphics.

    Parameters
    ----------
    message : str
        Markdown/HTML formatted string.
    level : str
        One of 'info', 'success', 'warning', 'error'.
    """
    _inject_css()
    palette = {
        "info":    {"border": "#4F8EF7", "bg": "rgba(79, 142, 247, 0.08)", "color": "#4F8EF7"},
        "success": {"border": "#00C9A7", "bg": "rgba(0, 201, 167, 0.08)",  "color": "#00C9A7"},
        "warning": {"border": "#FBBF24", "bg": "rgba(251, 191, 36, 0.10)", "color": "#FBBF24"},
        "error":   {"border": "#F87171", "bg": "rgba(248, 113, 113, 0.10)", "color": "#F87171"},
    }
    c = palette.get(level, palette["info"])
    st.markdown(
        f"""<div style="border-left: 4px solid {c['border']}; background: {c['bg']}; border-radius: 8px; padding: 0.8rem 1.1rem; margin: 0.8rem 0; color: #E0E4F0; font-size: 0.85rem; line-height: 1.45;">
            {message}
        </div>""",
        unsafe_allow_html=True,
    )

