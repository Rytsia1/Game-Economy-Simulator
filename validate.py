import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import run_simulation, gini_coefficient
from analytics.diagnostics import run_diagnostics
from analytics.optimizer import tune_parameter
import numpy as np


eco = EconomyConfig()
sim = SimulationConfig(num_players=10_000, days=90, random_seed=42, stochastic_mode=True)
r = run_simulation(sim, eco)

fb  = r["final_balances"]
ids = r["archetype_ids"]
tta = r["time_to_afford"]

print("=" * 65)
print(f"1. VECTORIZED SIMULATION BENCHMARK (v0.1 & v0.2)")
print("=" * 65)
print(f"10,000p x 90d stochastic => {r['elapsed_ms']:.1f} ms")
print(f"Gini coefficient: {gini_coefficient(fb):.4f}")
print()
for arch in r["archetypes"]:
    mask = ids == r["archetypes"].index(arch)
    med  = float(np.median(fb[mask]))
    t1   = tta[arch.name]["tier_1"]
    t2   = tta[arch.name]["tier_2"]
    print(f"  {arch.name:12s}  share={arch.population_share:.0%}  "
          f"median={med:>8,.0f}g  "
          f"T1=Day {t1 or 'N/A':>3}  T2=Day {t2 or 'N/A':>3}")

print()
print(f"All balances non-negative: {(fb >= 0).all()}")

# 2. Economy Doctor Diagnostics
print("\n" + "=" * 65)
print("2. ECONOMY DOCTOR DIAGNOSTICS (v0.3)")
print("=" * 65)
diag = run_diagnostics(r, eco, sim)
print(f"Overall Health   : {diag.overall_health.value} ({diag.overall_health.tag})")
print(f"Macro Flow Ratio : {diag.flow_ratio:.2f}x")
print(f"Gini Disparity   : {diag.gini:.4f}")
print(f"Casual Fail Rate : {diag.casual_fail_rate:.1%}")
print(f"Alerts Triggered : {len(diag.alerts)}")
for i, alert in enumerate(diag.alerts, 1):
    print(f"  [{alert.severity.value}] {alert.title}")
    print(f"     Fix: {alert.recommendation[:90]}...")

# 3. Target-Driven Auto-Tuner
print("\n" + "=" * 65)
print("3. AUTO-TUNER BENCHMARK (v0.3)")
print("=" * 65)
tuner_result = tune_parameter(
    eco_config       = eco,
    sim_config       = sim,
    param_name       = "quest_reward",
    search_lo        = 10.0,
    search_hi        = 250.0,
    target_archetype = "Casual",
    target_tier      = 2,
    target_day       = 35,
    probe_players    = 1000,
    max_iterations   = 15,
)
print(f"Target           : Casual Tier-2 Affordability by Day 35")
print(f"Recommended Value : quest_reward = {tuner_result.best_value:.2f}g (approximate heuristic)")
print(f"Achieved Day     : Day {tuner_result.achieved_day} (Residual: {tuner_result.residual:.1f}d)")
print(f"Converged        : {tuner_result.converged} in {tuner_result.iterations} iterations")
print(f"Total Tuner Time : {tuner_result.elapsed_ms:.1f} ms (Target: < 2000 ms)")

# 4. Scenario Stress-Testing Matrix
from config.scenarios import run_scenario_matrix
print("\n" + "=" * 65)
print("4. SCENARIO STRESS-TESTING MATRIX (v0.4)")
print("=" * 65)
s_matrix = run_scenario_matrix(base_sim_config=sim, base_eco_config=eco)
for _, row in s_matrix["summary_df"].iterrows():
    print(f"  {row['Scenario']:30s} | {row['Health Status']:8s} | "
          f"Flow: {row['Flow Ratio']:.2f}x | Gini: {row['Gini']:.4f} | "
          f"T1: {row['Casual T1 Afford Day']}")

# 5. Monte Carlo Risk Analysis Engine
from simulation.monte_carlo import run_monte_carlo
print("\n" + "=" * 65)
print("5. MONTE CARLO RISK ANALYSIS BENCHMARK (v0.4)")
print("=" * 65)
mc_sim = SimulationConfig(num_players=1000, days=30, random_seed=42)
mc_res = run_monte_carlo(mc_sim, eco, num_runs=100, volatility=0.15, random_seed=42)
print(f"Total Iterations : {mc_res.num_runs} runs (N=1,000 players, 30 days)")
print(f"Execution Time   : {mc_res.elapsed_ms:.1f} ms (Target: < 2,000-3,000 ms)")
print(f"Avg Time per Run : {mc_res.elapsed_ms / mc_res.num_runs:.1f} ms/run")
print(f"Inflation Risk   : {mc_res.prob_inflation:.1%} (P[Flow Ratio >= 1.5x])")
print(f"Poverty Risk     : {mc_res.prob_poverty:.1%} (P[Casual Fail >= 40%])")
print(f"Stability Index  : {mc_res.stability_score:.1%} (Passes both gates)")
p50_final = mc_res.percentiles_df.iloc[-1]
print(f"Day 30 Wealth CI : p5={p50_final['p5']:,.0f}g | p25={p50_final['p25']:,.0f}g | "
      f"p50={p50_final['p50']:,.0f}g | p75={p50_final['p75']:,.0f}g | p95={p50_final['p95']:,.0f}g")
print("=" * 65)


