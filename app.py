"""
app.py — Game Economy Simulator  (Phase 3 | v0.3)
==================================================
Streamlit dashboard with three tabs:
  Tab 1 – Simulation    : All v0.2 archetype + wealth charts
  Tab 2 – Economy Doctor: Diagnostic gauges, alert cards, prescriptions
  Tab 3 – Auto-Tuner    : Binary-search solver UI with convergence chart

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import streamlit as st

from analytics.diagnostics import run_diagnostics
from analytics.optimizer import tune_parameter, TunerResult
from config.scenarios import (
    get_scenario_presets,
    run_scenario_matrix,
    SCENARIO_COLORS,
    SCENARIO_DESCRIPTIONS,
)
from config.settings import ArchetypeProfile, EconomyConfig, SimulationConfig, DEFAULT_ARCHETYPES
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation
from simulation.monte_carlo import run_monte_carlo, MonteCarloResult
from visualization.charts import (
    plot_affordability_milestones,
    plot_archetype_distribution,
    plot_archetype_progression,
    plot_diagnostic_gauges,
    plot_inflow_outflow,
    plot_tuner_convergence,
    plot_tuner_sensitivity,
    plot_wealth_distribution,
    plot_wealth_progression,
)
from visualization.health_cards import (
    render_alert_cards,
    render_before_after_kpis,
    render_health_banner,
    render_notification_box,
    render_tuner_delta,
    section_title as hc_section,
)
from visualization.scenario_charts import (
    plot_scenario_comparison,
    plot_scenario_metrics_bar,
    plot_monte_carlo_fan_chart,
    plot_risk_distribution,
    plot_risk_scatter_or_cdf,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Game Economy Simulator",
    page_icon="GES",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
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
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; color: var(--text) !important; }
.stApp { background-color: var(--bg); }
.block-container { padding-top: 1.5rem !important; max-width: 1440px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }

/* KPI Cards */
.kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.1rem 1.4rem;
    display: flex; flex-direction: column; gap: 0.3rem;
    transition: transform 0.18s ease, border-color 0.18s ease;
}
.kpi-card:hover { transform: translateY(-3px); border-color: var(--accent-blue); }
.kpi-label { font-size:0.72rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); }
.kpi-value { font-size:1.9rem; font-weight:700; line-height:1.15; }
.kpi-sub   { font-size:0.78rem; color:var(--text-muted); }
.kpi-blue   { color: var(--accent-blue); }
.kpi-teal   { color: var(--accent-teal); }
.kpi-purple { color: var(--accent-purple); }
.kpi-amber  { color: var(--accent-amber); }
.kpi-rose   { color: var(--accent-rose); }

/* Alert cards */
.alert-card {
    border-radius: var(--radius); padding: 1rem 1.2rem;
    margin-bottom: 0.8rem; border-left: 4px solid;
    background: var(--surface);
}
.alert-title   { font-size:0.9rem; font-weight:700; margin-bottom:0.25rem; }
.alert-detail  { font-size:0.78rem; color: var(--text-muted); margin-bottom:0.4rem; }
.alert-rec     { font-size:0.78rem; border-top:1px solid var(--border); padding-top:0.35rem; margin-top:0.25rem; }
.alert-rec::before { content:"Fix: "; }

/* Tuner result card */
.tuner-result {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.2rem 1.5rem;
}
.tuner-value   { font-size:2.2rem; font-weight:700; }
.tuner-label   { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted); }

/* Milestone card */
.milestone-card {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
}
.milestone-arch { font-size:0.8rem; font-weight:600; margin-bottom:0.3rem; }
.milestone-row  { display:flex; justify-content:space-between; font-size:0.78rem; color:var(--text-muted); }
.milestone-day  { font-weight:600; color:var(--text); }

/* Section headers */
.section-title {
    font-size:0.78rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
    color:var(--text-muted); margin:1.5rem 0 0.6rem 0;
    padding-bottom:0.4rem; border-bottom:1px solid var(--border);
}

/* Perf badge / mode pill */
.perf-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:var(--surface-2); border:1px solid var(--border);
    border-radius:999px; padding:0.25rem 0.75rem;
    font-size:0.78rem; color:var(--text-muted);
}
.mode-pill { display:inline-flex; align-items:center; gap:0.3rem; padding:0.15rem 0.65rem; border-radius:999px; font-size:0.72rem; font-weight:600; letter-spacing:0.06em; }
.mode-stochastic    { background:rgba(167,139,250,0.15); color:var(--accent-purple); border:1px solid rgba(167,139,250,0.4); }
.mode-deterministic { background:rgba(79,142,247,0.15);  color:var(--accent-blue);   border:1px solid rgba(79,142,247,0.4); }

.js-plotly-plot { border-radius: var(--radius); overflow: hidden; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — shared parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0 0 0.2rem 0;font-size:1.2rem;'>Economy Simulator</h2>"
        "<p style='font-size:0.75rem;color:#8892AA;margin-bottom:1.2rem;'>Phase 3 | v0.3 — Doctor + Tuner</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Simulation Scale</div>", unsafe_allow_html=True)
    num_players       = st.slider("Number of Players", 100, 10_000, 2_000, step=100)
    days              = st.slider("Simulation Days", 7, 90, 45)
    starting_gold     = st.number_input("Starting Gold per Player", 0, 10_000, 200, step=50)
    random_seed       = st.number_input("Random Seed", 0, 99_999, 42)
    stochastic_mode   = st.toggle("Stochastic Mode", value=True)

    st.markdown("<div class='section-title'>Archetype Population %</div>", unsafe_allow_html=True)
    st.caption("Auto-normalised to 100 %.")
    
    raw_shares: dict[str, float] = {}
    for arch in DEFAULT_ARCHETYPES:
        raw_shares[arch.name] = st.slider(
            f"{arch.name} %",
            0, 100, int(arch.population_share * 100), step=5,
            key=f"share_{arch.name}",
        )
    total_share  = sum(raw_shares.values()) or 1
    norm_shares  = {k: v / total_share for k, v in raw_shares.items()}

    st.markdown("<div class='section-title'>Gold Sources (Base)</div>", unsafe_allow_html=True)
    quest_reward    = st.number_input("Quest Reward (gold)", 0, 1_000, 50, step=5)
    quests_per_day  = st.slider("Quests per Day (base)", 0.0, 20.0, 3.0, step=0.5)
    enemy_reward    = st.number_input("Enemy Reward μ (gold)", 0, 500, 20, step=5)
    enemies_per_day = st.slider("Enemies per Day (base)", 0.0, 30.0, 5.0, step=0.5)
    enemy_std_pct   = st.slider("Enemy Drop Std-Dev (%)", 0.0, 1.0, 0.25, step=0.05)

    st.markdown("<div class='section-title'>Gold Sinks</div>", unsafe_allow_html=True)
    potion_cost          = st.number_input("Potion Cost (gold)", 0, 500, 10, step=5)
    potions_per_day      = st.slider("Potions per Day (base)", 0.0, 20.0, 5.0, step=0.5)
    weapon_cost          = st.number_input("Periodic Weapon Cost (gold)", 0, 5_000, 300, step=50)
    weapon_interval_days = st.slider("Weapon Purchase Interval (days)", 1, 30, 7)

    st.markdown("<div class='section-title'>Gear Milestones</div>", unsafe_allow_html=True)
    tier_1_cost = st.number_input("Tier 1 Weapon Cost (gold)", 100, 10_000, 500, step=100)
    tier_2_cost = st.number_input("Tier 2 Weapon Cost (gold)", 500, 50_000, 2_500, step=500)

    st.markdown("---")
    run_btn = st.button("Run Simulation", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Helper: build archetype list from sidebar
# ---------------------------------------------------------------------------

def _build_archetypes(shares: dict[str, float]) -> list[ArchetypeProfile]:
    return [
        dataclasses.replace(arch, population_share=shares[arch.name])
        for arch in DEFAULT_ARCHETYPES
    ]


# ---------------------------------------------------------------------------
# Session state — run simulation
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
    st.session_state.results  = run_simulation(sim_cfg, eco_cfg)
    st.session_state.eco_cfg  = eco_cfg
    st.session_state.sim_cfg  = sim_cfg

results: dict             = st.session_state.results
eco_cfg: EconomyConfig    = st.session_state.eco_cfg
sim_cfg: SimulationConfig = st.session_state.sim_cfg

df_metrics        = results["metrics"]
archetype_metrics = results["archetype_metrics"]
final_balances    = results["final_balances"]
archetype_ids     = results["archetype_ids"]
archetypes        = results["archetypes"]
time_to_afford    = results["time_to_afford"]
elapsed_ms        = results["elapsed_ms"]

# ---------------------------------------------------------------------------
# Global header
# ---------------------------------------------------------------------------

hdr_l, hdr_r = st.columns([5, 2], vertical_alignment="center")
with hdr_l:
    st.markdown(
        "<h1 style='margin:0;font-size:1.75rem;font-weight:700;'>Game Economy Simulator</h1>",
        unsafe_allow_html=True,
    )
with hdr_r:
    perf_color = "var(--accent-teal)" if elapsed_ms < 1000 else "var(--accent-rose)"
    mode_cls   = "mode-stochastic" if sim_cfg.stochastic_mode else "mode-deterministic"
    mode_lbl   = "Stochastic" if sim_cfg.stochastic_mode else "Deterministic"
    st.markdown(
        f"<div style='display:flex;gap:0.5rem;justify-content:flex-end;align-items:center;'>"
        f"<div class='mode-pill {mode_cls}'>{mode_lbl}</div>"
        f"<div class='perf-badge'>"
        f"<span>{elapsed_ms:.1f} ms | {num_players:,}p | {days}d</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:0.6rem 0 1rem 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_scenarios, tab_monte_carlo, tab_tuner = st.tabs(
    [
        "Dashboard & Health",
        "Scenario Stress-Test",
        "Monte Carlo Risk Lab",
        "Auto-Tuning Lab",
    ]
)


# ============================================================
# TAB 1 — DASHBOARD & HEALTH  (Unified Simulation & Diagnostics)
# ============================================================

with tab_dashboard:
    # — KPI row —
    avg_final  = float(np.mean(final_balances))
    med_final  = float(np.median(final_balances))
    isr        = income_spending_ratio(df_metrics)
    gini       = gini_coefficient(final_balances)

    def _arch_med(name: str) -> float:
        idx = next((i for i, a in enumerate(archetypes) if a.name == name), None)
        if idx is None: return 0.0
        m = archetype_ids == idx
        return float(np.median(final_balances[m])) if m.any() else 0.0

    grinder_med = _arch_med("Grinder")
    casual_med  = _arch_med("Casual")
    wealth_gap  = (grinder_med / casual_med) if casual_med > 0 else float("inf")
    p10 = float(np.percentile(final_balances, 10))
    p90 = float(np.percentile(final_balances, 90))
    disparity = (p90 / p10) if p10 > 0 else float("inf")

    kpi_cols = st.columns(5)
    for col, kpi in zip(kpi_cols, [
        {"label":"Avg Final Gold",     "value":f"{avg_final:,.0f}", "sub":f"Δ {avg_final-float(starting_gold):+,.0f}", "color":"kpi-blue"},
        {"label":"Median Final Gold",  "value":f"{med_final:,.0f}", "sub":f"Skew {(avg_final-med_final)/max(med_final,1):.2%}", "color":"kpi-teal"},
        {"label":"Income / Spending",  "value":f"{isr:.2f}×",       "sub":"Ratio > 1 = economy grows",  "color":"kpi-amber"},
        {"label":"Gini Coefficient",   "value":f"{gini:.4f}",       "sub":f"P90/P10: {disparity:.1f}×", "color":"kpi-purple"},
        {"label":"Grinder/Casual Gap", "value":f"{wealth_gap:.2f}×","sub":f"G {grinder_med:,.0f} vs C {casual_med:,.0f}", "color":"kpi-rose"},
    ]):
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{kpi['label']}</div>
                <div class="kpi-value {kpi['color']}">{kpi['value']}</div>
                <div class="kpi-sub">{kpi['sub']}</div>
            </div>""", unsafe_allow_html=True)

    # — Diagnostics calculation —
    if "diag" not in st.session_state or run_btn:
        st.session_state.diag = run_diagnostics(results, eco_cfg, sim_cfg, casual_target_day=20)
    diag = st.session_state.diag

    # — Health Banner & Gauges —
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    render_health_banner(diag.overall_health.value, diag.summary)

    st.plotly_chart(
        plot_diagnostic_gauges(diag.flow_ratio, diag.gini, diag.casual_fail_rate),
        use_container_width=True, config={"displayModeBar": False},
    )

    # — Diagnostic Alerts —
    render_alert_cards(diag.alerts)

    # — Archetype trajectory —
    hc_section("Archetype Wealth Trajectories")
    st.plotly_chart(
        plot_archetype_progression(archetype_metrics, archetypes),
        use_container_width=True, config={"displayModeBar": False},
    )

    # — Row 2: violin + milestones —
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        hc_section("Wealth Distribution by Archetype")
        st.plotly_chart(
            plot_archetype_distribution(final_balances, archetype_ids, archetypes),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c2:
        hc_section("Time-to-Afford Milestones")
        st.plotly_chart(
            plot_affordability_milestones(
                time_to_afford, archetypes, int(eco_cfg.tier_1_weapon_cost), int(eco_cfg.tier_2_weapon_cost)
            ), use_container_width=True, config={"displayModeBar": False},
        )

    # — Milestone info cards —
    hc_section("Affordability Breakdown")
    mcols = st.columns(len(archetypes))
    for col, arch in zip(mcols, archetypes):
        tta   = time_to_afford.get(arch.name, {})
        t1, t2 = tta.get("tier_1"), tta.get("tier_2")
        with col:
            st.markdown(f"""<div class="milestone-card">
                <div class="milestone-arch" style="color:{arch.color};">{arch.name}</div>
                <div class="milestone-row"><span>T1 ({eco_cfg.tier_1_weapon_cost:,}g)</span>
                    <span class="milestone-day">{"Day "+str(t1) if t1 else "Never"}</span></div>
                <div class="milestone-row"><span>T2 ({eco_cfg.tier_2_weapon_cost:,}g)</span>
                    <span class="milestone-day">{"Day "+str(t2) if t2 else "Never"}</span></div>
            </div>""", unsafe_allow_html=True)

    # — Row 3: global progression + inflow/outflow —
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        hc_section("Global Wealth Progression")
        st.plotly_chart(
            plot_wealth_progression(df_metrics),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c4:
        hc_section("Gold Inflow vs Outflow")
        st.plotly_chart(
            plot_inflow_outflow(df_metrics),
            use_container_width=True, config={"displayModeBar": False},
        )

    # — Global histogram —
    hc_section("Global Wealth Distribution")
    st.plotly_chart(
        plot_wealth_distribution(final_balances),
        use_container_width=True, config={"displayModeBar": False},
    )

    # — Raw data expanders —
    with st.expander("Raw Day-by-Day Metrics", expanded=False):
        d = df_metrics.copy()
        d.columns = [c.replace("_", " ").title() for c in d.columns]
        st.dataframe(
            d.style.format({col: "{:,.1f}" for col in d.columns if col.lower() != "day"}),
            use_container_width=True, hide_index=True,
        )
    with st.expander("Archetype Metrics (Long Format)", expanded=False):
        st.dataframe(
            archetype_metrics.style.format({"median_gold": "{:,.1f}"}),
            use_container_width=True, hide_index=True,
        )


# ============================================================
# TAB 2 — SCENARIO STRESS-TEST MATRIX
# ============================================================

with tab_scenarios:
    st.markdown("### Scenario Stress-Testing Matrix")
    st.caption(
        "Evaluate macroeconomic resilience across 4 distinct regimes: "
        "Baseline, Gold Rush (Hyper-Inflation), Economic Crisis (Depression), and Demographic Shock."
    )

    # Regime overview cards
    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.75rem; margin-bottom: 1.2rem;">
        <div style="background:#1A1D27; border:1px solid #2A2D3A; border-radius:10px; padding:0.8rem 1rem;">
            <div style="font-weight:700; color:#00C9A7; font-size:0.9rem;">1. Normal (Baseline)</div>
            <div style="font-size:0.75rem; color:#8892AA; margin-top:0.3rem;">Default balanced equilibrium across all sources and sinks.</div>
        </div>
        <div style="background:#1A1D27; border:1px solid #2A2D3A; border-radius:10px; padding:0.8rem 1rem;">
            <div style="font-weight:700; color:#FBBF24; font-size:0.9rem;">2. Gold Rush (Hyper-Inflation)</div>
            <div style="font-size:0.75rem; color:#8892AA; margin-top:0.3rem;">Quest rewards +100%, enemy drops +80%. Unchecked gold snowballing.</div>
        </div>
        <div style="background:#1A1D27; border:1px solid #2A2D3A; border-radius:10px; padding:0.8rem 1rem;">
            <div style="font-weight:700; color:#F87171; font-size:0.9rem;">3. Economic Crisis (Depression)</div>
            <div style="font-size:0.75rem; color:#8892AA; margin-top:0.3rem;">Sources -30%, item prices +25%, potion usage +20%. Severe poverty trap.</div>
        </div>
        <div style="background:#1A1D27; border:1px solid #2A2D3A; border-radius:10px; padding:0.8rem 1rem;">
            <div style="font-weight:700; color:#A78BFA; font-size:0.9rem;">4. Hardcore Shift (Shock)</div>
            <div style="font-size:0.75rem; color:#8892AA; margin-top:0.3rem;">Demographics: Grinders 50%, Optimizers 30%, Casuals 10%. High inequality.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_scenarios_btn = st.button("Run Stress-Test Matrix", type="primary", use_container_width=True)

    if run_scenarios_btn or "scenario_results" not in st.session_state:
        with st.spinner("Simulating all 4 macroeconomic regimes…"):
            matrix_res = run_scenario_matrix(base_sim_config=sim_cfg, base_eco_config=eco_cfg)
            st.session_state.scenario_results = matrix_res

    m_res = st.session_state.scenario_results
    sum_df: pd.DataFrame = m_res["summary_df"]
    prog_df: pd.DataFrame = m_res["progression_df"]

    # Comparative summary table
    hc_section("Multi-Scenario Comparative Summary")
    st.dataframe(
        sum_df[[
            "Scenario", "Health Status", "Avg Gold", "Median Gold",
            "Flow Ratio", "Gini", "Casual T1 Afford Day", "Casual Fail Rate %"
        ]].style.format({
            "Avg Gold": "{:,.1f}g",
            "Median Gold": "{:,.1f}g",
            "Flow Ratio": "{:.2f}×",
            "Gini": "{:.4f}",
            "Casual Fail Rate %": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    hc_section("Wealth Trajectories Across Regimes")
    st.plotly_chart(
        plot_scenario_comparison(prog_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    hc_section("Key Diagnostic Metrics Comparison")
    st.plotly_chart(
        plot_scenario_metrics_bar(sum_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Per-scenario drill-down
    with st.expander("Detailed Regime Diagnoses & Prescriptions", expanded=False):
        for sc_name, sc_diag in m_res["diagnostics"].items():
            st.markdown(f"#### {sc_name}")
            st.caption(SCENARIO_DESCRIPTIONS.get(sc_name, ""))
            render_alert_cards(sc_diag.alerts)
            st.divider()


# ============================================================
# TAB 3 — MONTE CARLO RISK LAB
# ============================================================

with tab_monte_carlo:
    st.markdown("### Monte Carlo Risk Analysis Lab")
    st.caption(
        "Evaluate macroeconomic resilience across stochastic simulation iterations with "
        "parameter jittering, confidence interval fan charts, and empirical risk quantification."
    )

    # Input controls
    mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
    with mc_c1:
        mc_runs = st.slider("Iterations (M)", 20, 200, 50, step=10, help="Number of simulated economic paths.")
    with mc_c2:
        mc_volatility = st.slider(
            "Macro Volatility (σ)", 5, 30, 15, step=1,
            help="Standard deviation of macro shocks applied to quest rewards and potion costs.",
        )
    with mc_c3:
        mc_players = st.slider(
            "Players per Run", 500, 2_000, 1_000, step=100,
            help="Sample size of players per Monte Carlo iteration.",
        )
    with mc_c4:
        st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)
        run_mc_btn = st.button("Run Monte Carlo Analysis", type="primary", use_container_width=True)

    if run_mc_btn:
        progress_bar = st.progress(0, text="Initializing Monte Carlo risk simulations…")

        def _mc_progress(current: int, total: int):
            frac = current / total
            progress_bar.progress(frac, text=f"Executing run {current}/{total}…")

        _mc_sim = dataclasses.replace(sim_cfg, num_players=int(mc_players))
        mc_out = run_monte_carlo(
            base_sim_config   = _mc_sim,
            base_eco_config   = eco_cfg,
            num_runs          = int(mc_runs),
            volatility        = float(mc_volatility) / 100.0,
            progress_callback = _mc_progress,
        )
        progress_bar.empty()
        st.session_state.mc_result = mc_out

    if "mc_result" in st.session_state:
        mc: MonteCarloResult = st.session_state.mc_result

        # KPI Metric Cards
        hc_section("Empirical Risk Quantifications")
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Inflation Risk P(Flow ≥ 1.5×)</div>
                <div class="kpi-value {'kpi-rose' if mc.prob_inflation > 0.25 else 'kpi-teal'}">{mc.prob_inflation:.1%}</div>
                <div class="kpi-sub">Runs exceeding inflation gate</div>
            </div>""", unsafe_allow_html=True)
        with rk2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Poverty Trap Risk P(Fail ≥ 40%)</div>
                <div class="kpi-value {'kpi-rose' if mc.prob_poverty > 0.25 else 'kpi-teal'}">{mc.prob_poverty:.1%}</div>
                <div class="kpi-sub">Runs with progression bottleneck</div>
            </div>""", unsafe_allow_html=True)
        with rk3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">System Stability Index</div>
                <div class="kpi-value {'kpi-teal' if mc.stability_score >= 0.70 else 'kpi-amber'}">{mc.stability_score:.1%}</div>
                <div class="kpi-sub">Runs passing both risk gates</div>
            </div>""", unsafe_allow_html=True)
        with rk4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Execution Time</div>
                <div class="kpi-value kpi-blue">{mc.elapsed_ms:.0f} ms</div>
                <div class="kpi-sub">{mc.num_runs} runs | avg {mc.elapsed_ms/mc.num_runs:.1f}ms/run</div>
            </div>""", unsafe_allow_html=True)

        # Plotly Fan Chart
        hc_section("Confidence Interval Fan Chart")
        st.plotly_chart(
            plot_monte_carlo_fan_chart(mc),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Risk Distribution & Quadrant Scatter
        mc_r1, mc_r2 = st.columns(2, gap="medium")
        with mc_r1:
            hc_section("Flow Ratio Risk Distribution")
            st.plotly_chart(
                plot_risk_distribution(mc),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with mc_r2:
            hc_section("System Stability Quadrants")
            st.plotly_chart(
                plot_risk_scatter_or_cdf(mc),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Raw runs table
        with st.expander(f"Monte Carlo Run Logs ({mc.num_runs} Iterations)", expanded=False):
            st.dataframe(
                mc.run_summaries.style.format({
                    "quest_reward": "{:.1f}g",
                    "potion_cost": "{:.1f}g",
                    "flow_ratio": "{:.2f}×",
                    "gini": "{:.4f}",
                    "median_wealth": "{:,.0f}g",
                    "casual_fail_rate": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        render_notification_box("Configure the Monte Carlo parameters above and click <b>Run Monte Carlo Analysis</b> to begin.", "info")



# ============================================================
# TAB 3 — PARAMETER AUTO-TUNER
# ============================================================

with tab_tuner:
    st.markdown("### Parameter Auto-Tuner")
    st.caption(
        "Define a pacing target — *'The median Casual player should afford Tier-1 gear by Day D\u002a'* — "
        "and the binary-search solver finds the optimal economy parameter to hit it."
    )

    # — Tuner controls —
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        tune_param = st.selectbox(
            "Parameter to Tune",
            ["quest_reward", "enemy_reward", "potion_cost", "potions_per_day"],
            help="The scalar economy knob the solver will adjust.",
        )
        tune_archetype = st.selectbox(
            "Target Archetype",
            [a.name for a in DEFAULT_ARCHETYPES],
            index=0,
        )
        tune_tier = st.radio("Target Gear Tier", [1, 2], horizontal=True)

    with tc2:
        tune_target_day = st.slider(
            "Target Milestone Day D*",
            3, int(days), min(20, int(days)),
            help="The day by which 50%+ of the target archetype should afford the gear.",
        )
        tune_lo = st.number_input(
            "Search Lower Bound",
            value=5.0 if tune_param in ("quest_reward", "enemy_reward") else 1.0,
            min_value=0.0, step=5.0,
        )
        tune_hi = st.number_input(
            "Search Upper Bound",
            value=300.0 if tune_param in ("quest_reward", "enemy_reward") else 30.0,
            min_value=tune_lo + 1.0, step=10.0,
        )

    with tc3:
        probe_players = st.slider(
            "Players per Probe (speed vs precision)",
            200, 3_000, 1_000, step=100,
            help="Smaller = faster search; larger = more accurate objective per probe.",
        )
        max_iters = st.slider("Max Search Iterations", 5, 30, 15)
        run_tuner_btn = st.button("Run Auto-Tuner", type="primary", use_container_width=True)

    if run_tuner_btn:
        if tune_lo >= tune_hi:
            render_notification_box("Search Lower Bound must be strictly less than Upper Bound.", "error")
        else:
            with st.spinner(f"Binary-searching {tune_param} in [{tune_lo:.1f}, {tune_hi:.1f}]…"):
                tuner_result: TunerResult = tune_parameter(
                    eco_config       = eco_cfg,
                    sim_config       = sim_cfg,
                    param_name       = tune_param,
                    search_lo        = float(tune_lo),
                    search_hi        = float(tune_hi),
                    target_archetype = tune_archetype,
                    target_tier      = int(tune_tier),
                    target_day       = int(tune_target_day),
                    probe_players    = int(probe_players),
                    max_iterations   = int(max_iters),
                )
            st.session_state.tuner_result = tuner_result

    # — Display tuner result if available —
    if "tuner_result" in st.session_state:
        tr: TunerResult = st.session_state.tuner_result

        # ── Delta comparison: Current → Recommended ────────────────────────
        hc_section("Parameter Change Recommendation")
        old_val = getattr(eco_cfg, tr.param_name, None)

        # Baseline milestone day: run a quick single probe at the original value
        if "tuner_baseline_day" not in st.session_state or run_btn:
            from analytics.optimizer import _probe as _opt_probe
            _baseline_sim = dataclasses.replace(sim_cfg, num_players=1_000)
            st.session_state.tuner_baseline_day = _opt_probe(
                eco_cfg, _baseline_sim, tr.param_name,
                float(old_val) if old_val is not None else 0.0,
                tr.target_archetype, tr.target_tier,
            )
        baseline_day = st.session_state.tuner_baseline_day

        if old_val is not None:
            render_tuner_delta(
                param_name    = tr.param_name,
                old_value     = float(old_val),
                new_value     = tr.best_value,
                old_day       = baseline_day,
                new_day       = tr.achieved_day,
                target_day    = tr.target_day,
                converged     = tr.converged,
            )

        # ── KPI strip (before / after) ─────────────────────────────────────
        hc_section("Before / After Economy Snapshot")
        render_before_after_kpis([
            {
                "label":        tr.param_name.replace("_"," ").title(),
                "before_value": f"{float(old_val):.1f}g" if old_val else "—",
                "after_value":  f"{tr.best_value:.1f}g",
                "before_sub":   "Baseline",
                "after_sub":    "Recommended",
            },
            {
                "label":        "Milestone Day",
                "before_value": f"Day {baseline_day}" if baseline_day else "Never",
                "after_value":  f"Day {tr.achieved_day}" if tr.achieved_day else "Never",
                "before_sub":   f"{tr.target_archetype} Tier-{tr.target_tier} (before)",
                "after_sub":    f"{tr.target_archetype} Tier-{tr.target_tier} (after)",
            },
            {
                "label":        "Residual",
                "before_value": "—",
                "after_value":  f"{tr.residual:.1f}d",
                "before_sub":   "n/a",
                "after_sub":    "Converged" if tr.converged else "Not converged",
            },
        ])

        # ── Recommendation banner ─────────────────────────────────────────
        if tr.converged:
            render_notification_box(
                f"<b>Set <code>{tr.param_name}</code> = {tr.best_value:.2f}</b> to achieve "
                f"<b>{tr.target_archetype}</b> Tier-{tr.target_tier} affordability by "
                f"<b>Day {tr.target_day}</b> (validated: Day {tr.achieved_day}, "
                f"residual {tr.residual:.1f}d).",
                "success",
            )
        else:
            render_notification_box(
                f"Search did not fully converge within {tr.iterations} iterations. "
                f"Best candidate: <code>{tr.param_name}</code> = {tr.best_value:.2f} "
                f"(achieved Day {tr.achieved_day}, residual {tr.residual:.1f}d). "
                "Try expanding search bounds or increasing max iterations.",
                "warning",
            )

        # ── Side-by-side Before vs After Progression Curves ───────────────
        hc_section("Before vs After — Archetype Wealth Trajectories")
        import pandas as pd

        # Build "tuned" simulation results for the recommended parameter value
        _tuned_eco = dataclasses.replace(eco_cfg, **{tr.param_name: tr.best_value})
        _tuned_sim = dataclasses.replace(sim_cfg)
        with st.spinner("Generating tuned simulation for comparison…"):
            if (
                "tuned_results" not in st.session_state
                or st.session_state.get("tuned_param") != tr.param_name
                or st.session_state.get("tuned_value") != tr.best_value
            ):
                st.session_state.tuned_results  = run_simulation(_tuned_sim, _tuned_eco)
                st.session_state.tuned_param     = tr.param_name
                st.session_state.tuned_value     = tr.best_value

        tuned_results = st.session_state.tuned_results

        ba_col1, ba_col2 = st.columns(2, gap="medium")
        with ba_col1:
            st.markdown(
                "<div style='text-align:center;font-size:0.8rem;font-weight:600;"
                "color:#F87171;padding:0.3rem 0 0.5rem;'>BEFORE (Baseline)</div>",
                unsafe_allow_html=True,
            )
            fig_before = plot_archetype_progression(archetype_metrics, archetypes)
            fig_before.update_layout(title=dict(text="Baseline Progression", font=dict(size=14)), height=340)
            st.plotly_chart(fig_before, use_container_width=True, config={"displayModeBar": False})

        with ba_col2:
            st.markdown(
                "<div style='text-align:center;font-size:0.8rem;font-weight:600;"
                "color:#00C9A7;padding:0.3rem 0 0.5rem;'>AFTER (Tuned)</div>",
                unsafe_allow_html=True,
            )
            fig_after = plot_archetype_progression(
                tuned_results["archetype_metrics"], tuned_results["archetypes"]
            )
            fig_after.update_layout(
                title=dict(
                    text=f"Tuned: {tr.param_name} = {tr.best_value:.1f}",
                    font=dict(size=14),
                ),
                height=340,
            )
            st.plotly_chart(fig_after, use_container_width=True, config={"displayModeBar": False})

        # ── Convergence trace ──────────────────────────────────────────────
        hc_section("Binary Search Convergence Trace")
        st.plotly_chart(
            plot_tuner_convergence(tr.probes, tr.target_day, tr.param_name),
            use_container_width=True, config={"displayModeBar": False},
        )

        # ── Sensitivity sweep ──────────────────────────────────────────────
        hc_section("Parameter Sensitivity")
        sweep_vals = sorted(set(p.param_value for p in tr.probes))
        probe_map  = {p.param_value: p.achieved_day for p in tr.probes}
        sweep_days = [probe_map.get(v) for v in sweep_vals]
        st.plotly_chart(
            plot_tuner_sensitivity(
                sweep_values   = sweep_vals,
                achieved_days  = sweep_days,
                param_name     = tr.param_name,
                target_day     = tr.target_day,
                archetype_name = tr.target_archetype,
            ),
            use_container_width=True, config={"displayModeBar": False},
        )

        # ── Probe log ──────────────────────────────────────────────────────
        with st.expander("Binary Search Probe Log", expanded=False):
            probe_rows = [
                {
                    "Iteration": p.iteration,
                    f"{tr.param_name}": round(p.param_value, 3),
                    "Achieved Day": p.achieved_day or "N/A",
                    "Objective (error)": round(p.objective, 2),
                }
                for p in tr.probes
            ]
            st.dataframe(pd.DataFrame(probe_rows), use_container_width=True, hide_index=True)
    else:
        render_notification_box("Configure the tuner parameters above and click <b>Run Auto-Tuner</b> to begin.", "info")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='text-align:center;color:#8892AA;font-size:0.72rem;margin-top:2rem;'>"
    "Game Economy Simulator | Phase 3 v0.3 | Economy Doctor + Auto-Tuner"
    "</div>",
    unsafe_allow_html=True,
)
