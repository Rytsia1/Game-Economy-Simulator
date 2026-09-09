from config.settings import EconomyConfig, SimulationConfig
from simulation.engine import run_simulation, gini_coefficient
import numpy as np

eco = EconomyConfig()
sim = SimulationConfig(num_players=10_000, days=90, random_seed=42, stochastic_mode=True)
r = run_simulation(sim, eco)

fb  = r["final_balances"]
ids = r["archetype_ids"]
tta = r["time_to_afford"]

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
