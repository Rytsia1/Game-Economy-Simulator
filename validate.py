from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import run_simulation, gini_coefficient, income_spending_ratio
import numpy as np

eco = EconomyConfig()
sim = SimulationConfig(num_players=10_000, days=90, random_seed=42)
r = run_simulation(sim, eco)
fb = r["final_balances"]
ms = r["elapsed_ms"]

print(f"Players: 10,000 x 90 days => {ms:.1f} ms")
print(f"Avg Final Gold:     {np.mean(fb):,.0f}")
print(f"Median Final Gold:  {np.median(fb):,.0f}")
print(f"Gini Coefficient:   {gini_coefficient(fb):.4f}")
print(f"Income/Spend Ratio: {income_spending_ratio(r['metrics']):.3f}")
print(f"Min balance:        {fb.min():.2f}")
print(f"All non-negative:   {(fb >= 0).all()}")
