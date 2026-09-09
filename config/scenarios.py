"""
config/scenarios.py
-------------------
Pre-configured macroeconomic regimes and multi-scenario comparative matrix
for the Game Economy Simulator (Step 4 / v0.4).

Regimes:
1. Normal (Baseline)          – Default balanced parameters.
2. Gold Rush (Hyper-Inflation)– Quest rewards +100%, enemy gold +80%, base sinks unchanged.
3. Economic Crisis (Depression)– Gold sources -30%, item prices +25%, potion usage +20%.
4. Hardcore Shift (Shock)      – Grinder 50%, Optimizer 30%, Casual 10%, Collector 10%.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.diagnostics import run_diagnostics, DiagnosticReport
from config.settings import ArchetypeProfile, EconomyConfig, SimulationConfig, DEFAULT_ARCHETYPES
from simulation.engine import gini_coefficient, income_spending_ratio, run_simulation


# ---------------------------------------------------------------------------
# Scenario descriptions & color mappings
# ---------------------------------------------------------------------------

SCENARIO_COLORS: Dict[str, str] = {
    "Normal (Baseline)":           "#00C9A7",  # Teal
    "Gold Rush (Hyper-Inflation)": "#FBBF24",  # Amber
    "Economic Crisis (Depression)": "#F87171",  # Rose
    "Hardcore Shift (Demographic)": "#A78BFA",  # Purple
}

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "Normal (Baseline)": (
        "Default balanced economy parameters. Steady gold generation and consumption equilibrium."
    ),
    "Gold Rush (Hyper-Inflation)": (
        "Quest reward +100% (50g -> 100g), Enemy drops +80% (20g -> 36g). Sinks remain unchanged. "
        "Extreme wealth snowballing."
    ),
    "Economic Crisis (Depression)": (
        "Gold sources -30% (Quest 35g, Enemy 14g), item prices +25% (Potions 12.5g, Weapons +25%), "
        "potions/day +20%. Severe deflationary pressure."
    ),
    "Hardcore Shift (Demographic)": (
        "Population shifts heavily to power players: Grinders 50%, Optimizers 30%, Casuals 10%, "
        "Collectors 10%. High farming intensity and market inequality."
    ),
}


# ---------------------------------------------------------------------------
# Scenario Presets Factory
# ---------------------------------------------------------------------------

def get_scenario_presets(
    base_sim_config: Optional[SimulationConfig] = None,
    base_eco_config: Optional[EconomyConfig] = None,
) -> Dict[str, Tuple[SimulationConfig, EconomyConfig]]:
    """
    Generate deep-copied (SimulationConfig, EconomyConfig) pairs for the 4
    macroeconomic stress-test scenarios without mutating the baseline inputs.

    Parameters
    ----------
    base_sim_config : SimulationConfig, optional
        Base simulation settings. If None, default SimulationConfig() is used.
    base_eco_config : EconomyConfig, optional
        Base economy settings. If None, default EconomyConfig() is used.

    Returns
    -------
    Dict[str, Tuple[SimulationConfig, EconomyConfig]]
        Dictionary mapping scenario names to (SimulationConfig, EconomyConfig).
    """
    sim_base = copy.deepcopy(base_sim_config) if base_sim_config is not None else SimulationConfig()
    eco_base = copy.deepcopy(base_eco_config) if base_eco_config is not None else EconomyConfig()

    # 1. Normal (Baseline)
    sim_normal = copy.deepcopy(sim_base)
    eco_normal = copy.deepcopy(eco_base)

    # 2. Gold Rush (Hyper-Inflation): Quest +100%, Enemy +80%, sinks unchanged
    sim_goldrush = copy.deepcopy(sim_base)
    eco_goldrush = copy.deepcopy(eco_base)
    eco_goldrush.quest_reward = eco_base.quest_reward * 2.0
    eco_goldrush.enemy_reward = eco_base.enemy_reward * 1.8

    # 3. Economic Crisis (Depression): Gold sources -30%, item prices +25%, potion usage +20%
    sim_crisis = copy.deepcopy(sim_base)
    eco_crisis = copy.deepcopy(eco_base)
    eco_crisis.quest_reward = eco_base.quest_reward * 0.70
    eco_crisis.enemy_reward = eco_base.enemy_reward * 0.70
    eco_crisis.potion_cost = eco_base.potion_cost * 1.25
    eco_crisis.weapon_cost = eco_base.weapon_cost * 1.25
    eco_crisis.tier_1_weapon_cost = int(eco_base.tier_1_weapon_cost * 1.25)
    eco_crisis.tier_2_weapon_cost = int(eco_base.tier_2_weapon_cost * 1.25)
    eco_crisis.potions_per_day = eco_base.potions_per_day * 1.20

    # 4. Hardcore Shift (Demographic Shock): Grinder 50%, Optimizer 30%, Casual 10%, Collector 10%
    sim_hardcore = copy.deepcopy(sim_base)
    eco_hardcore = copy.deepcopy(eco_base)
    hardcore_archetypes: List[ArchetypeProfile] = [
        ArchetypeProfile(
            name="Casual",
            population_share=0.10,
            quest_mult=0.6,
            enemy_mult=0.4,
            potion_mult=0.8,
            saving_buffer_mult=1.05,
            color="#4F8EF7",
        ),
        ArchetypeProfile(
            name="Grinder",
            population_share=0.50,
            quest_mult=1.2,
            enemy_mult=2.0,
            potion_mult=1.5,
            saving_buffer_mult=1.2,
            color="#00C9A7",
        ),
        ArchetypeProfile(
            name="Collector",
            population_share=0.10,
            quest_mult=0.8,
            enemy_mult=0.7,
            potion_mult=1.2,
            saving_buffer_mult=1.0,
            color="#A78BFA",
        ),
        ArchetypeProfile(
            name="Optimizer",
            population_share=0.30,
            quest_mult=1.0,
            enemy_mult=1.3,
            potion_mult=0.3,
            saving_buffer_mult=2.0,
            color="#FBBF24",
        ),
    ]
    eco_hardcore.archetypes = hardcore_archetypes

    return {
        "Normal (Baseline)":           (sim_normal, eco_normal),
        "Gold Rush (Hyper-Inflation)": (sim_goldrush, eco_goldrush),
        "Economic Crisis (Depression)": (sim_crisis, eco_crisis),
        "Hardcore Shift (Demographic)": (sim_hardcore, eco_hardcore),
    }


# ---------------------------------------------------------------------------
# Multi-Scenario Matrix Execution
# ---------------------------------------------------------------------------

def run_scenario_matrix(
    base_sim_config: Optional[SimulationConfig] = None,
    base_eco_config: Optional[EconomyConfig] = None,
    presets: Optional[Dict[str, Tuple[SimulationConfig, EconomyConfig]]] = None,
) -> Dict[str, Any]:
    """
    Simulate all 4 scenario presets, calculate single-run diagnostics,
    and generate unified comparative summary and progression DataFrames.

    Parameters
    ----------
    base_sim_config : SimulationConfig, optional
        Baseline simulation configuration.
    base_eco_config : EconomyConfig, optional
        Baseline economy configuration.
    presets : Dict[str, Tuple[SimulationConfig, EconomyConfig]], optional
        Custom scenario dictionary if not using default presets.

    Returns
    -------
    dict with keys:
        "summary_df"    : pd.DataFrame with summary comparison metrics.
        "progression_df": pd.DataFrame with daily median wealth per scenario.
        "results"       : Dict[str, dict] mapping scenario name to raw sim results.
        "diagnostics"   : Dict[str, DiagnosticReport] mapping scenario name to health report.
    """
    if presets is None:
        presets = get_scenario_presets(base_sim_config, base_eco_config)

    sim_results: Dict[str, dict] = {}
    diagnostic_reports: Dict[str, DiagnosticReport] = {}
    summary_rows: List[dict] = []
    progression_dfs: List[pd.DataFrame] = []

    for name, (s_cfg, e_cfg) in presets.items():
        res = run_simulation(s_cfg, e_cfg)
        sim_results[name] = res

        diag = run_diagnostics(res, e_cfg, s_cfg)
        diagnostic_reports[name] = diag

        fb = res["final_balances"]
        avg_gold = float(np.mean(fb))
        med_gold = float(np.median(fb))
        flow_ratio = diag.flow_ratio
        gini = diag.gini

        # Casual T1 affordability day
        tta = res["time_to_afford"]
        casual_t1 = tta.get("Casual", {}).get("tier_1")
        casual_t1_str = f"Day {casual_t1}" if casual_t1 is not None else "Never"

        summary_rows.append({
            "Scenario":             name,
            "Avg Gold":             round(avg_gold, 1),
            "Median Gold":          round(med_gold, 1),
            "Gini":                 round(gini, 4),
            "Flow Ratio":           round(flow_ratio, 2),
            "Casual T1 Afford Day": casual_t1_str,
            "Health Status":        diag.overall_health.value,
            "Health Emoji":         diag.overall_health.emoji,
            "Casual Fail Rate %":   round(diag.casual_fail_rate * 100, 1),
        })

        # Daily median progression
        metrics = res["metrics"]
        pdf = pd.DataFrame({
            "day":         metrics["day"],
            "scenario":    name,
            "median_gold": metrics["median_gold"],
            "avg_gold":    metrics["avg_gold"],
        })
        progression_dfs.append(pdf)

    summary_df = pd.DataFrame(summary_rows)
    progression_df = pd.concat(progression_dfs, ignore_index=True) if progression_dfs else pd.DataFrame()

    return {
        "summary_df":     summary_df,
        "progression_df": progression_df,
        "results":        sim_results,
        "diagnostics":    diagnostic_reports,
    }
