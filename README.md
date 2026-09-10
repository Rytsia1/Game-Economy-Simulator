# Game Economy Simulator & Tuning Engine

> A lightweight, vectorized stochastic simulation tool to model, stress-test, and auto-tune virtual game economies.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/Core-NumPy%20Vectorized-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-164%20Passed-00C9A7.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

A personal portfolio project that treats game economy balancing as a **quantitative simulation and optimization problem** rather than a spreadsheet exercise.

The tool lets a designer configure gold sources (quests, enemy drops), gold sinks (potions, gear purchases), and a heterogeneous player population — then instantly see:

- how wealth distributes across player archetypes over time,
- whether the economy is healthy, inflating, or trapping casual players in poverty,
- which parameter value approximately hits a target progression milestone,
- how the economy behaves under four stress scenarios and across 100 Monte Carlo runs.

Everything runs in a Streamlit dashboard with no backend, no database, and no external API calls.

---

## Dashboard Overview

The dashboard has five tabs:

| Tab | What it does |
|---|---|
| **Simulation** | Run the core engine. See wealth curves, archetype breakdowns, and time-to-afford milestones. |
| **Economy Doctor** | Automated diagnostics: macro flow ratio, Gini coefficient, and rule-based alert cards with prescriptions. |
| **Auto-Tuner** | Binary search solver. Dial in a target milestone day — the tuner searches for the parameter value that approximately meets it. |
| **Stress-Test** | Run four economic regimes (Normal, Gold Rush, Economic Crisis, Demographic Shock) side-by-side. |
| **Monte Carlo** | Jitter economy parameters across 100 runs. See P5–P95 confidence fan charts and empirical risk probabilities. |

<!-- Screenshot placeholder — see docs/SCREENSHOTS.md for capture instructions -->
<!-- ![Simulation tab](docs/dashboard_simulation.png) -->
<!-- ![Economy Doctor tab](docs/dashboard_economy_doctor.png) -->
<!-- ![Auto-Tuner tab](docs/dashboard_auto_tuner.png) -->

---

## The Problem with Static Spreadsheets

Most game economies start life as a spreadsheet. Spreadsheets compute deterministic averages well — but they break down when:

- **Players don't behave uniformly.** A Grinder farms 2x more enemies than a Casual. Under the same global parameters, one archetype races ahead while the other stalls.
- **Drops are random.** Poisson quest completions and clipped Normal enemy drops produce skewed wealth distributions that static formulas hide.
- **Pacing depends on cumulative balance over time.** "Can the median Casual afford Tier-1 gear by Day 20?" requires integrating a stochastic wealth curve — not dividing a reward by a price.

---

## How It Works

```
  CONFIGURATION
  (EconomyConfig + SimulationConfig)
           |
           v
  VECTORIZED ENGINE  [simulation/engine.py]
  Assigns P players to archetypes, draws a full (P × D)
  income matrix and potion-spend matrix up front using
  NumPy Poisson and Normal draws, then steps through D
  days in a single Python loop over days (not players).
           |
     +-----------+--------------+------------------+
     |           |              |                  |
     v           v              v                  v
  DIAGNOSTICS  AUTO-TUNER  STRESS-TEST      MONTE CARLO
  [analytics/  [analytics/  [config/         [simulation/
  diagnostics] optimizer]   scenarios]       monte_carlo]
```

### Simulation Engine

- Player archetypes are assigned via `numpy.random.Generator.choice` on population shares.
- Daily activity matrices `[P × D]` for quests, enemies, and potions are pre-allocated using vectorized random draws — no per-player Python loops.
- The day loop (unavoidable for cumulative balance tracking) runs over `D` days (typically 30–90), vectorizing across all `P` players per step.
- Sample benchmark: **10,000 players × 90 days in ~0.18–0.22 s** on a standard modern desktop (measured; will vary by hardware).

### Economy Doctor

Runs three quantitative checks automatically after every simulation:

1. **Macro Flow Ratio** (`total income / total spending`): classifies the economy as Balanced, Inflating, or a Poverty Trap.
2. **Gini Coefficient** (vectorized Lorenz formula on final balances): measures wealth inequality.
3. **Casual Fail Rate**: fraction of Casual players who never reach the target milestone within the simulation window.

Each check produces a `DiagnosticAlert` with a severity level and a concrete prescription (e.g., *"increase potion cost by 10–15% to reduce flow ratio"*).

### Auto-Tuner

A binary search loop that approximates a progression constraint:

1. Designer specifies: *"parameter X, archetype Y, milestone tier Z should be reached by Day D*"*.
2. The tuner runs fast probe simulations (default: 1,000 players) at the midpoint of a search interval, evaluates the signed error `(achieved_day − target_day)`, and halves the interval.
3. Stops when `|error| ≤ 1 day` or the interval narrows to ≤ 1 gold.
4. Validates the recommended value on the full configured population.

This approach assumes a monotonic relationship between the parameter and the milestone day — an assumption that holds in expectation for the modelled economy but is not mathematically guaranteed under stochastic noise. The result is an approximate heuristic estimate, not an exact solution.

### Why Vectorization Matters

A naive Python implementation loops over every player on every day — `O(P × D)` interpreter overhead. The vectorized engine pre-computes the full `(P × D)` income tensor before the day loop, trading a slightly higher memory footprint for a ~20–26x wall-clock speedup in sample measurements. This makes interactive parameter sweeps and 100-run Monte Carlo studies responsive in the Streamlit UI.

---

## Player Archetypes

| Archetype | Share | Quest mult | Enemy mult | Potion mult | Saving buffer |
|---|---|---|---|---|---|
| **Casual** | 50% | 0.6x | 0.4x | 0.8x | 1.05x |
| **Grinder** | 25% | 1.2x | 2.0x | 1.5x | 1.2x |
| **Collector** | 15% | 0.8x | 0.7x | 1.2x | 1.0x |
| **Optimizer** | 10% | 1.0x | 1.3x | 0.3x | 2.0x |

Archetypes are assigned stochastically on each run. All multipliers scale the base economy parameters from `EconomyConfig`.

---

## Scenario Stress Matrix

Four economic regimes available in the Stress-Test tab:

| Scenario | Description |
|---|---|
| **Normal (Baseline)** | Default balanced parameters |
| **Gold Rush (Hyper-Inflation)** | Quest rewards +100%, enemy drops +80%, sinks unchanged |
| **Economic Crisis (Depression)** | Sources −30%, item prices +25%, potion usage +20% |
| **Hardcore Shift (Demographic)** | Population skewed to 50% Grinders / 30% Optimizers |

---

## Repository Structure

```
game-economy-simulator/
├── analytics/
│   ├── diagnostics.py       # Macro flow, Gini, rule-based alert engine
│   └── optimizer.py         # Binary-search auto-tuner
├── config/
│   ├── scenarios.py         # 4-regime stress-test matrix
│   └── settings.py          # EconomyConfig, SimulationConfig, ArchetypeProfile
├── simulation/
│   ├── engine.py            # Vectorized (P × D) simulation core
│   └── monte_carlo.py       # Monte Carlo risk engine & fan chart builder
├── visualization/
│   ├── charts.py            # Plotly wealth, archetype, and tuner charts
│   ├── health_cards.py      # Diagnostic banner and alert card renderers
│   └── scenario_charts.py   # Fan charts, risk scatter, scenario comparisons
├── tests/
│   ├── test_engine.py       # Vectorization invariants, benchmark checks
│   ├── test_diagnostics.py  # Gini and flow ratio rule verification
│   ├── test_optimizer.py    # Monotonicity, convergence, edge-case guards
│   ├── test_scenarios.py    # Scenario matrix configuration tests
│   ├── test_monte_carlo.py  # Fan chart percentiles and risk metric tests
│   └── test_integration.py  # End-to-end pipeline integration tests
├── docs/
│   └── SCREENSHOTS.md       # Instructions for adding dashboard screenshots
├── app.py                   # Streamlit dashboard (5 tabs)
├── requirements.txt
└── validate.py              # CLI benchmark & diagnostic verification script
```

---

## Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

```bash
# 1. Clone
git clone https://github.com/Rytsia1/Game-Economy-Simulator.git
cd Game-Economy-Simulator

# 2. Virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
# Full test suite (164 tests)
pytest tests/ -v

# CLI benchmark and diagnostic check
python validate.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

---

## Reproducibility

All stochastic behavior (Poisson quest draws, clipped Normal enemy drops, Monte Carlo parameter jitter) uses `numpy.random.default_rng`:

- **Fixed seed** (`SimulationConfig.random_seed=42` by default): identical results across runs with the same configuration.
- **Variable seed** (`random_seed=None`): natural stochastic variation in wealth curves and milestone days.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Computation | NumPy (vectorized matrix ops, `default_rng`), Pandas |
| Dashboard | Streamlit (session-state caching, custom CSS) |
| Charts | Plotly Graph Objects (fan charts, violin, gauge, scatter) |
| Testing | Pytest — 164 unit + integration tests |

---

## License

MIT — see `LICENSE`.
