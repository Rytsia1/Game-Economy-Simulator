"""
app.py — Game Economy Simulator  (Phase 2 · v0.2)
==================================================
Streamlit dashboard with:
  • Sidebar: archetype population sliders, gear milestone costs, stochastic toggle
  • Top KPI cards: global + Grinder vs Casual wealth gap
  • Archetype trajectory chart (median by archetype)
  • Wealth distribution violin by archetype + global histogram
  • Inflow/Outflow balance chart
  • Time-to-Afford milestone bar chart
  • Raw data expander

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from config.settings import ArchetypeProfile, EconomyConfig, SimulationConfig, DEFAULT_ARCHETYPES
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation
from visualization.charts import (
    plot_affordability_milestones,
    plot_archetype_distribution,
    plot_archetype_progression,
    plot_inflow_outflow,
    plot_wealth_distribution,
    plot_wealth_progression,
)

# ---------------------------------------------------------------------------
# Page config
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

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
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
        color: var(--text) !important;
    }
    .stApp { background-color: var(--bg); }
    .block-container { padding-top: 1.5rem !important; max-width: 1400px; }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    /* KPI Cards */
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.4rem;
        display: flex; flex-direction: column; gap: 0.3rem;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .kpi-card:hover { transform: translateY(-3px); border-color: var(--accent-blue); }
    .kpi-label {
        font-size: 0.72rem; font-weight: 500;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--text-muted);
    }
    .kpi-value { font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
    .kpi-sub   { font-size: 0.78rem; color: var(--text-muted); }
    .kpi-blue   { color: var(--accent-blue); }
    .kpi-teal   { color: var(--accent-teal); }
    .kpi-purple { color: var(--accent-purple); }
    .kpi-amber  { color: var(--accent-amber); }
    .kpi-rose   { color: var(--accent-rose); }

    /* Milestone card */
    .milestone-card {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .milestone-arch { font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem; }
    .milestone-row  { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); }
    .milestone-day  { font-weight: 600; color: var(--text); }

    /* Section headers */
    .section-title {
        font-size: 0.78rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.5rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    /* Perf badge */
    .perf-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 999px; padding: 0.25rem 0.75rem;
        font-size: 0.78rem; color: var(--text-muted);
    }
    .perf-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--accent-teal); display: inline-block;
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.45; transform:scale(0.75); }
    }

    /* Mode pill */
    .mode-pill {
        display: inline-flex; align-items: center; gap: 0.3rem;
        padding: 0.15rem 0.65rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    }
    .mode-stochastic { background: rgba(167,139,250,0.15); color: var(--accent-purple); border: 1px solid rgba(167,139,250,0.4); }
    .mode-deterministic { background: rgba(79,142,247,0.15); color: var(--accent-blue); border: 1px solid rgba(79,142,247,0.4); }

    .js-plotly-plot { border-radius: var(--radius); overflow: hidden; }
    hr { border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0 0 0.2rem 0; font-size:1.2rem;'>⚔️ Economy Simulator</h2>"
        "<p style='font-size:0.75rem; color:#8892AA; margin-bottom:1.2rem;'>Phase 2 · v0.2 — Archetypes</p>",
        unsafe_allow_html=True,
    )

    # ── Simulation scale ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>🎮 Simulation Scale</div>", unsafe_allow_html=True)
    num_players       = st.slider("Number of Players", 100, 10_000, 2_000, step=100)
    days              = st.slider("Simulation Days", 7, 90, 45)
    starting_gold     = st.number_input("Starting Gold per Player", 0, 10_000, 200, step=50)
    random_seed       = st.number_input("Random Seed", 0, 99_999, 42)
    stochastic_mode   = st.toggle("Stochastic Mode (Poisson + Normal draws)", value=True)

    # ── Archetype population shares ───────────────────────────────────
    st.markdown("<div class='section-title'>👥 Archetype Population Shares</div>", unsafe_allow_html=True)
    st.caption("Values are auto-normalised to 100%.")

    default_shares = {a.name: a.population_share for a in DEFAULT_ARCHETYPES}
    raw_shares: dict[str, float] = {}
    arch_colors = {a.name: a.color for a in DEFAULT_ARCHETYPES}
    arch_emojis = {"Casual": "🟦", "Grinder": "🟩", "Collector": "🟣", "Optimizer": "🟡"}

    for arch in DEFAULT_ARCHETYPES:
        raw_shares[arch.name] = st.slider(
            f"{arch_emojis.get(arch.name, '')} {arch.name} %",
            0, 100,
            int(default_shares[arch.name] * 100),
            step=5,
            key=f"share_{arch.name}",
        )

    total_share = sum(raw_shares.values()) or 1  # avoid div-by-zero
    norm_shares = {k: v / total_share for k, v in raw_shares.items()}

    # Show normalised pie hint
    share_strs = "  ·  ".join(f"{k}: {v:.0%}" for k, v in norm_shares.items())
    st.caption(f"Normalised → {share_strs}")

    # ── Gold sources ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>💰 Gold Sources (Base)</div>", unsafe_allow_html=True)
    quest_reward    = st.number_input("Quest Reward (gold)", 0, 1_000, 50, step=5)
    quests_per_day  = st.slider("Quests per Day (base)", 0.0, 20.0, 3.0, step=0.5)
    enemy_reward    = st.number_input("Enemy Reward μ (gold)", 0, 500, 20, step=5)
    enemies_per_day = st.slider("Enemies per Day (base)", 0.0, 30.0, 5.0, step=0.5)
    enemy_std_pct   = st.slider("Enemy Drop Std-Dev (%)", 0.0, 1.0, 0.25, step=0.05,
                                help="σ of per-enemy gold drop as fraction of μ.")

    # ── Gold sinks ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🧪 Gold Sinks</div>", unsafe_allow_html=True)
    potion_cost          = st.number_input("Potion Cost (gold)", 0, 500, 10, step=5)
    potions_per_day      = st.slider("Potions per Day (base)", 0.0, 20.0, 5.0, step=0.5)
    weapon_cost          = st.number_input("Periodic Weapon Cost (gold)", 0, 5_000, 300, step=50)
    weapon_interval_days = st.slider("Weapon Purchase Interval (days)", 1, 30, 7)

    # ── Gear milestones ───────────────────────────────────────────────
    st.markdown("<div class='section-title'>🏆 Gear Milestones</div>", unsafe_allow_html=True)
    tier_1_cost = st.number_input("Tier 1 Weapon Cost (gold)", 100, 10_000, 500, step=100)
    tier_2_cost = st.number_input("Tier 2 Weapon Cost (gold)", 500, 50_000, 2_500, step=500)

    st.markdown("---")
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Build archetype list from sidebar shares
# ---------------------------------------------------------------------------

def _build_archetypes(norm_shares: dict[str, float]) -> list[ArchetypeProfile]:
    result = []
    for arch in DEFAULT_ARCHETYPES:
        result.append(ArchetypeProfile(
            name=arch.name,
            population_share=norm_shares[arch.name],
            quest_mult=arch.quest_mult,
            enemy_mult=arch.enemy_mult,
            potion_mult=arch.potion_mult,
            saving_buffer_mult=arch.saving_buffer_mult,
            color=arch.color,
        ))
    return result

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

if st.session_state.results is None or run_btn:
    _archetypes = _build_archetypes(norm_shares)
    eco_cfg = EconomyConfig(
        quest_reward=float(quest_reward),
        quests_per_day=float(quests_per_day),
        enemy_reward=float(enemy_reward),
        enemies_per_day=float(enemies_per_day),
        enemy_reward_std_pct=float(enemy_std_pct),
        potion_cost=float(potion_cost),
        potions_per_day=float(potions_per_day),
        weapon_cost=float(weapon_cost),
        weapon_interval_days=int(weapon_interval_days),
        tier_1_weapon_cost=int(tier_1_cost),
        tier_2_weapon_cost=int(tier_2_cost),
        archetypes=_archetypes,
    )
    sim_cfg = SimulationConfig(
        num_players=int(num_players),
        days=int(days),
        starting_gold=float(starting_gold),
        random_seed=int(random_seed),
        stochastic_mode=bool(stochastic_mode),
    )
    st.session_state.results = run_simulation(sim_cfg, eco_cfg)
    st.session_state.eco_cfg = eco_cfg
    st.session_state.sim_cfg = sim_cfg

results: dict      = st.session_state.results
eco_cfg: EconomyConfig  = st.session_state.eco_cfg
sim_cfg: SimulationConfig = st.session_state.sim_cfg

df_metrics         = results["metrics"]
archetype_metrics  = results["archetype_metrics"]
final_balances     = results["final_balances"]
archetype_ids      = results["archetype_ids"]
archetypes         = results["archetypes"]
time_to_afford     = results["time_to_afford"]
elapsed_ms         = results["elapsed_ms"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_badge = st.columns([5, 2], vertical_alignment="center")
with col_title:
    st.markdown(
        "<h1 style='margin:0; font-size:1.75rem; font-weight:700;'>"
        "⚔️ Game Economy Simulator"
        "</h1>",
        unsafe_allow_html=True,
    )
with col_badge:
    perf_color = "var(--accent-teal)" if elapsed_ms < 1000 else "var(--accent-rose)"
    mode_class = "mode-stochastic" if sim_cfg.stochastic_mode else "mode-deterministic"
    mode_label = "🎲 Stochastic" if sim_cfg.stochastic_mode else "📐 Deterministic"
    st.markdown(
        f"<div style='display:flex; gap:0.5rem; justify-content:flex-end; align-items:center;'>"
        f"<div class='mode-pill {mode_class}'>{mode_label}</div>"
        f"<div class='perf-badge'>"
        f"<span class='perf-dot' style='background:{perf_color};'></span>"
        f"<span>{elapsed_ms:.1f} ms · {num_players:,}p · {days}d</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:0.6rem 0 1.2rem 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI Cards — Row 1: Global stats
# ---------------------------------------------------------------------------

avg_final  = float(np.mean(final_balances))
med_final  = float(np.median(final_balances))
isr        = income_spending_ratio(df_metrics)
gini       = gini_coefficient(final_balances)

# Grinder vs Casual median gap
def _archetype_median_final(name: str) -> float:
    a_idx = next((i for i, a in enumerate(archetypes) if a.name == name), None)
    if a_idx is None:
        return 0.0
    mask = archetype_ids == a_idx
    return float(np.median(final_balances[mask])) if mask.any() else 0.0

grinder_med = _archetype_median_final("Grinder")
casual_med  = _archetype_median_final("Casual")
wealth_gap  = (grinder_med / casual_med) if casual_med > 0 else float("inf")

p10 = float(np.percentile(final_balances, 10))
p90 = float(np.percentile(final_balances, 90))
disparity = (p90 / p10) if p10 > 0 else float("inf")

kpi_cols = st.columns(5)
kpis = [
    {"label": "Avg Final Gold",      "value": f"{avg_final:,.0f}",  "sub": f"Δ {avg_final - float(starting_gold):+,.0f} from start", "color": "kpi-blue"},
    {"label": "Median Final Gold",   "value": f"{med_final:,.0f}",  "sub": f"Skew {(avg_final - med_final)/max(med_final,1):.2%}",    "color": "kpi-teal"},
    {"label": "Income / Spending",   "value": f"{isr:.2f}×",        "sub": "Ratio > 1 = economy grows",                               "color": "kpi-amber"},
    {"label": "Gini Coefficient",    "value": f"{gini:.4f}",        "sub": f"P90/P10: {disparity:.1f}×",                              "color": "kpi-purple"},
    {"label": "Grinder/Casual Gap",  "value": f"{wealth_gap:.2f}×", "sub": f"Grinder {grinder_med:,.0f} vs Casual {casual_med:,.0f}", "color": "kpi-rose"},
]

for col, kpi in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">{kpi['label']}</div>
                <div class="kpi-value {kpi['color']}">{kpi['value']}</div>
                <div class="kpi-sub">{kpi['sub']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Charts — Row 1: Archetype Trajectory (full width)
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>🧙 Archetype Wealth Trajectories</div>", unsafe_allow_html=True)
fig_arch_traj = plot_archetype_progression(archetype_metrics, archetypes)
st.plotly_chart(fig_arch_traj, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Charts — Row 2: Violin distribution + Milestone bar
# ---------------------------------------------------------------------------

row2_col1, row2_col2 = st.columns([1, 1], gap="medium")

with row2_col1:
    st.markdown("<div class='section-title'>🎻 Wealth Distribution by Archetype</div>", unsafe_allow_html=True)
    fig_violin = plot_archetype_distribution(final_balances, archetype_ids, archetypes)
    st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})

with row2_col2:
    st.markdown("<div class='section-title'>⏱️ Time-to-Afford Milestones</div>", unsafe_allow_html=True)
    fig_afford = plot_affordability_milestones(
        time_to_afford, archetypes,
        tier_1_cost=int(eco_cfg.tier_1_weapon_cost),
        tier_2_cost=int(eco_cfg.tier_2_weapon_cost),
    )
    st.plotly_chart(fig_afford, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Milestone breakdown info cards
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>📋 Affordability Breakdown (Day 50% cross milestone)</div>", unsafe_allow_html=True)
mile_cols = st.columns(len(archetypes))
for col, arch in zip(mile_cols, archetypes):
    tta = time_to_afford.get(arch.name, {})
    t1  = tta.get("tier_1")
    t2  = tta.get("tier_2")
    with col:
        st.markdown(
            f"""<div class="milestone-card">
                <div class="milestone-arch" style="color:{arch.color};">{arch.name}</div>
                <div class="milestone-row">
                    <span>Tier 1 ({eco_cfg.tier_1_weapon_cost:,}g)</span>
                    <span class="milestone-day">{"Day " + str(t1) if t1 else "Never"}</span>
                </div>
                <div class="milestone-row">
                    <span>Tier 2 ({eco_cfg.tier_2_weapon_cost:,}g)</span>
                    <span class="milestone-day">{"Day " + str(t2) if t2 else "Never"}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Charts — Row 3: Global progression + Inflow/Outflow
# ---------------------------------------------------------------------------

row3_col1, row3_col2 = st.columns([1, 1], gap="medium")

with row3_col1:
    st.markdown("<div class='section-title'>📈 Global Wealth Progression</div>", unsafe_allow_html=True)
    fig_wealth = plot_wealth_progression(df_metrics)
    st.plotly_chart(fig_wealth, use_container_width=True, config={"displayModeBar": False})

with row3_col2:
    st.markdown("<div class='section-title'>⚖️ Gold Inflow vs Outflow</div>", unsafe_allow_html=True)
    fig_flow = plot_inflow_outflow(df_metrics)
    st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Charts — Row 4: Global Histogram
# ---------------------------------------------------------------------------

st.markdown("<div class='section-title'>📊 Global Wealth Distribution (Final Day)</div>", unsafe_allow_html=True)
fig_hist = plot_wealth_distribution(final_balances)
st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Raw data expanders
# ---------------------------------------------------------------------------

with st.expander("🔍 Day-by-Day Metrics", expanded=False):
    display_df = df_metrics.copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(
        display_df.style.format(
            {col: "{:,.1f}" for col in display_df.columns if col.lower() != "day"}
        ),
        use_container_width=True, hide_index=True,
    )

with st.expander("🧙 Archetype Metrics (Long Format)", expanded=False):
    st.dataframe(
        archetype_metrics.style.format({"median_gold": "{:,.1f}"}),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center; color:#8892AA; font-size:0.72rem; margin-top:2rem;'>"
    "Game Economy Simulator · Phase 2 v0.2 · Archetypes + Stochastic Distributions"
    "</div>",
    unsafe_allow_html=True,
)
