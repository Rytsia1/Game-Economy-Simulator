"""
app.py — Game Economy Simulator  (Phase 1 · MVP v0.1)
======================================================
Streamlit dashboard combining:
  • Sidebar parameter controls
  • Top KPI metric cards
  • Three interactive Plotly charts
  • Performance benchmark readout

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation
from visualization.charts import (
    plot_inflow_outflow,
    plot_wealth_distribution,
    plot_wealth_progression,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Game Economy Simulator",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS – dark premium theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Import Google Font ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables ──────────────────────────────────────────────── */
    :root {
        --bg:           #0F1117;
        --surface:      #1A1D27;
        --surface-2:    #22263A;
        --border:       #2A2D3A;
        --text:         #E0E4F0;
        --text-muted:   #8892AA;
        --accent-blue:  #4F8EF7;
        --accent-teal:  #00C9A7;
        --accent-purple:#A78BFA;
        --accent-amber: #FBBF24;
        --accent-rose:  #F87171;
        --radius:       12px;
    }

    /* ── App shell ───────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
        color: var(--text) !important;
    }
    .stApp { background-color: var(--bg); }
    .block-container { padding-top: 1.5rem !important; max-width: 1400px; }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stNumberInput > label {
        font-size: 0.8rem !important;
        color: var(--text-muted) !important;
    }

    /* ── KPI Cards ───────────────────────────────────────────────────── */
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.4rem;
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent-blue);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .kpi-blue   { color: var(--accent-blue); }
    .kpi-teal   { color: var(--accent-teal); }
    .kpi-purple { color: var(--accent-purple); }
    .kpi-amber  { color: var(--accent-amber); }

    /* ── Section headers ──────────────────────────────────────────────── */
    .section-title {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.5rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Perf badge ───────────────────────────────────────────────────── */
    .perf-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .perf-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--accent-teal);
        display: inline-block;
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.45; transform: scale(0.75); }
    }

    /* ── Plotly chart background fix ─────────────────────────────────── */
    .js-plotly-plot { border-radius: var(--radius); overflow: hidden; }

    /* ── Divider ─────────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – parameter controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0 0 0.2rem 0; font-size:1.2rem;'>⚔️ Economy Simulator</h2>"
        "<p style='font-size:0.75rem; color:#8892AA; margin-bottom:1.2rem;'>Phase 1 · MVP v0.1</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>🎮 Simulation Scale</div>", unsafe_allow_html=True)
    num_players = st.slider("Number of Players", 100, 10_000, 1_000, step=100)
    days = st.slider("Simulation Days", 7, 90, 30)
    starting_gold = st.number_input("Starting Gold per Player", 0, 10_000, 500, step=50)
    random_seed = st.number_input("Random Seed", 0, 99_999, 42)

    st.markdown("<div class='section-title'>💰 Gold Sources</div>", unsafe_allow_html=True)
    quest_reward = st.number_input("Quest Reward (gold)", 0, 1_000, 50, step=5)
    quests_per_day = st.slider("Quests per Day", 0.0, 20.0, 3.0, step=0.5)
    enemy_reward = st.number_input("Enemy Reward (gold)", 0, 500, 20, step=5)
    enemies_per_day = st.slider("Enemies per Day", 0.0, 30.0, 5.0, step=0.5)

    st.markdown("<div class='section-title'>🧪 Gold Sinks</div>", unsafe_allow_html=True)
    potion_cost = st.number_input("Potion Cost (gold)", 0, 500, 10, step=5)
    potions_per_day = st.slider("Potions per Day", 0.0, 20.0, 5.0, step=0.5)
    weapon_cost = st.number_input("Weapon Cost (gold)", 0, 5_000, 300, step=50)
    weapon_interval_days = st.slider("Weapon Purchase Interval (days)", 1, 30, 7)

    st.markdown("<div class='section-title'>📡 Noise Model</div>", unsafe_allow_html=True)
    noise_std_pct = st.slider(
        "Income Noise Std-Dev (%)",
        0.0, 0.50, 0.15, step=0.01,
        help="Fraction of daily income used as the standard deviation of player-to-player randomness.",
    )

    st.markdown("---")
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Session state – persist last results across rerenders
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

# Auto-run on first load
if st.session_state.results is None or run_btn:
    eco_cfg = EconomyConfig(
        quest_reward=float(quest_reward),
        quests_per_day=float(quests_per_day),
        enemy_reward=float(enemy_reward),
        enemies_per_day=float(enemies_per_day),
        potion_cost=float(potion_cost),
        potions_per_day=float(potions_per_day),
        weapon_cost=float(weapon_cost),
        weapon_interval_days=int(weapon_interval_days),
    )
    sim_cfg = SimulationConfig(
        num_players=int(num_players),
        days=int(days),
        starting_gold=float(starting_gold),
        random_seed=int(random_seed),
        noise_std_pct=float(noise_std_pct),
    )
    st.session_state.results = run_simulation(sim_cfg, eco_cfg)
    st.session_state.eco_cfg = eco_cfg
    st.session_state.sim_cfg = sim_cfg

results: dict = st.session_state.results
df_metrics = results["metrics"]
final_balances = results["final_balances"]
elapsed_ms = results["elapsed_ms"]

# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------

col_title, col_badge = st.columns([6, 1], vertical_alignment="center")
with col_title:
    st.markdown(
        "<h1 style='margin:0; font-size:1.75rem; font-weight:700;'>"
        "⚔️ Game Economy Simulator"
        "</h1>",
        unsafe_allow_html=True,
    )
with col_badge:
    color = "var(--accent-teal)" if elapsed_ms < 1000 else "var(--accent-rose)"
    st.markdown(
        f"<div class='perf-badge'>"
        f"<span class='perf-dot' style='background:{color};'></span>"
        f"<span>{elapsed_ms:.1f} ms · {num_players:,} players · {days}d</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:0.6rem 0 1.2rem 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

avg_final = float(np.mean(final_balances))
med_final = float(np.median(final_balances))
isr = income_spending_ratio(df_metrics)
gini = gini_coefficient(final_balances)

# Wealth disparity: ratio of top-10% to bottom-10% median wealth
p10 = float(np.percentile(final_balances, 10))
p90 = float(np.percentile(final_balances, 90))
disparity = (p90 / p10) if p10 > 0 else float("inf")

kpi_cols = st.columns(4)

kpis = [
    {
        "label": "Average Final Gold",
        "value": f"{avg_final:,.0f}",
        "sub": f"Starting: {starting_gold:,}  ·  Δ {avg_final - starting_gold:+,.0f}",
        "color": "kpi-blue",
    },
    {
        "label": "Median Final Gold",
        "value": f"{med_final:,.0f}",
        "sub": f"Skew: {(avg_final - med_final) / max(med_final, 1):.2%}",
        "color": "kpi-teal",
    },
    {
        "label": "Income / Spending",
        "value": f"{isr:.2f}×",
        "sub": "Ratio > 1 = economy grows",
        "color": "kpi-amber",
    },
    {
        "label": "Gini Coefficient",
        "value": f"{gini:.4f}",
        "sub": f"P90/P10 disparity: {disparity:.1f}×",
        "color": "kpi-purple",
    },
]

for col, kpi in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{kpi['label']}</div>
                <div class="kpi-value {kpi['color']}">{kpi['value']}</div>
                <div class="kpi-sub">{kpi['sub']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Charts row 1 – Wealth Progression (full width)
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>📈 Wealth Over Time</div>", unsafe_allow_html=True)
fig_wealth = plot_wealth_progression(df_metrics)
st.plotly_chart(fig_wealth, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Charts row 2 – Distribution  +  Inflow/Outflow
# ---------------------------------------------------------------------------

chart_col1, chart_col2 = st.columns([1, 1], gap="medium")

with chart_col1:
    st.markdown("<div class='section-title'>📊 Wealth Distribution (Final Day)</div>", unsafe_allow_html=True)
    fig_dist = plot_wealth_distribution(final_balances)
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.markdown("<div class='section-title'>⚖️ Gold Inflow vs Outflow</div>", unsafe_allow_html=True)
    fig_flow = plot_inflow_outflow(df_metrics)
    st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Raw data expander (optional deep-dive)
# ---------------------------------------------------------------------------

with st.expander("🔍 Raw Simulation Metrics (Day-by-Day)", expanded=False):
    display_df = df_metrics.copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(
        display_df.style.format(
            {
                col: "{:,.1f}"
                for col in display_df.columns
                if col.lower() != "day"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center; color:#8892AA; font-size:0.72rem; margin-top:2rem;'>"
    "Game Economy Simulator · Phase 1 MVP v0.1 · Built with Streamlit + NumPy + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
