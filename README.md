# Game Economy Simulator & Tuning Engine
> A lightweight, vectorized stochastic simulation framework to model, stress-test, and auto-tune virtual game economies.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/Core-NumPy%20Vectorized-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-154%20Passed-00C9A7.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem
Most virtual game economies are conceived on static spreadsheets. While spreadsheets model deterministic averages effectively, they break down when dealing with dynamic player behavior:

1. **Behavioral Variance**: Players do not behave uniformly. High-velocity grinders flood the economy with gold, while casual players face severe progression blockers under the same global parameters.
2. **Stochastic Volatility**: Non-deterministic quest completions and randomized loot drops generate skewed wealth distributions, causing emergent poverty traps or runaway hyper-inflation that static formulas obscure.
3. **Pacing Bottlenecks**: Determining approximately *when* target archetypes can afford critical progression milestones (e.g., Tier-1 or Tier-2 gear) requires iterative multi-agent simulation over time, not static division.

---

## Solution Architecture
This portfolio project models game economy balancing as a quantitative simulation, diagnostic, and numerical optimization problem. Built on vectorized 1D/2D NumPy array operations, sample benchmark runs demonstrate simulating **10,000 players over 90 days in ~0.18–0.22 seconds** on standard modern desktop hardware, enabling fast interactive exploration in the Streamlit UI.

```
                +-------------------------+
                |   DESIGN CONFIGURATION  |
                | Sources, Sinks, Targets |
                +------------+------------+
                             |
                             v
                +-------------------------+
                |  VECTORIZED SIM ENGINE  |
                | Stochastic Multi-Agent  |
                +------------+------------+
                             |
       +---------------------+---------------------+
       v                     v                     v
+--------------------+ +--------------------+ +--------------------+
| ECONOMY DIAGNOSTIC | |  MONTE CARLO LAB   | | TARGET AUTO-TUNER  |
| Gini & Flow Ratios | | 100-Run Risk Bands | | Binary Search Opt  |
+--------------------+ +--------------------+ +--------------------+
```

---

## Key Capabilities

### 1. Heterogeneous Player Archetypes & Stochastic Behavior
Models emergent economic divergence across four distinct player archetypes using Poisson and Clipped Normal probability distributions:
- **Casual (50% population)**: Low quest rate (0.6x), low combat farming (0.4x), low saving buffer (1.05x). Prone to progression stalls when sinks outpace income.
- **Grinder (25% population)**: High engagement (1.2x quests, 2.0x combat farming), rapid gold velocity. Tests source caps and late-game sinks.
- **Collector (15% population)**: Moderate income (0.8x quests), aggressive milestone sink consumption (buys gear as soon as balance covers cost, 1.0x buffer).
- **Optimizer (10% population)**: High income efficiency, minimal consumable expenditure (0.3x potions), strategic wealth hoarding (requires a 2.0x buffer before purchasing).

### 2. Automated Economy Diagnostics ("Economy Doctor")
Monitors macroeconomic health using quantitative criteria and rule-based diagnostic assessments:
- **Macro Flow Ratio (Sources / Sinks)**: Evaluates systemic currency velocity to classify health into `Critical Inflation` (>= 2.0x), `Moderate Inflation` (1.3x–2.0x), `Balanced` (0.9x–1.3x), or `Poverty Trap` (< 0.9x).
- **Gini Coefficient**: Computes Lorenz-curve inequality across the simulated population (vectorized sort) to detect whether wealth concentration renders sinks unattainable for casual segments.
- **Progression Blocker Detection**: Flags warnings when >35% of the casual cohort fails to reach baseline milestone gear within the targeted design window.

### 3. Target-Driven Auto-Tuning Solver
Replaces manual parameter guessing with a 1D numerical root-finding loop:
- **Approach**: Designers specify a target progression constraint (e.g., *"the median Casual player should afford Tier-1 Weapon near Day 20"*). The solver uses binary search over a bounded parameter interval under an assumed monotonic relationship (e.g., higher quest rewards reduce time to afford; higher potion costs increase it).
- **Approximate Search**: Rather than finding an exact closed-form parameter value, the tuner searches for a parameter value that approximately satisfies the target progression constraint.
- **Precision Qualification**: The search evaluates candidate values using a reduced probe population (default: 1,000 players) for rapid iteration, stopping when probe error is within a convergence threshold (default: <= 1 day) or the search interval narrows. The recommended parameter value is then checked in a separate validation run on the full configured player population. Because the underlying simulation is stochastic and uses probe sampling, achieved milestone days on the full population may vary slightly around the target.

### 4. Scenario Stress-Testing Matrix
Executes side-by-side comparative stress-tests across four distinct economic regimes:
1. **Normal (Baseline)**: Default balanced equilibrium across all sources and sinks.
2. **Gold Rush (Hyper-Inflation)**: Quest rewards +100%, enemy drops +80%, unadjusted sinks.
3. **Economic Crisis (Depression)**: Gold sources -30%, item prices +25%, potion usage +20%.
4. **Hardcore Shift (Demographic Shock)**: Population skewed to 50% Grinders, 30% Optimizers, 10% Collectors, 10% Casuals.

### 5. Monte Carlo Tail-Risk Quantification
- Simulates multiple stochastic economic trajectories (e.g., 50–100 runs) with macro parameter jittering (volatility sigma applied to rewards and potion costs).
- Aggregates cross-run percentiles (P5, P25, P50, P75, P95) into interactive confidence interval fan charts.
- Computes empirical risk probabilities: **P(Inflation Risk)** (flow ratio >= 1.5x), **P(Poverty Trap Risk)** (casual fail rate >= 40%), and a composite **System Stability Score**.

---

## Reproducibility & Stochasticity
All stochastic behavior in the engine (Poisson quest/enemy frequencies, clipped Normal loot variance, and Monte Carlo parameter jitter) is powered by NumPy's `default_rng`:
- **Deterministic Runs**: Supplying a fixed integer seed via `SimulationConfig.random_seed` (default: `42`) produces repeatable simulation trajectories across runs with identical configurations.
- **Run-to-Run Variance**: Setting `random_seed=None` or varying the seed will yield natural stochastic variation in wealth curves, milestone days, and diagnostic metrics.
- **Monte Carlo Random Seed**: The Monte Carlo engine accepts an optional `random_seed` for reproducible parameter jittering across runs.

---

## Performance Benchmarks
Simulating individual player agent objects in pure Python loops incurs significant interpreter overhead ($O(N \times D)$ object allocations). By restructuring the engine into 1D/2D broadcasted NumPy arrays:

| Workload | Scale | Pure Python Loop (Ref) | Vectorized Engine | Observed Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Single Run Simulation | 10,000 players x 90 days | ~4.80 s | **~0.18 s** | **~26x** |
| Auto-Tuner Optimization | 15 iterations x 1,000 players | ~6.20 s | **~0.38 s** | **~16x** |
| Monte Carlo Analysis | 100 runs x 1,000 players x 30 days | ~32.50 s | **~1.25 s** | **~26x** |
| Complete Stress-Test Matrix | 4 regimes x 2,000 players x 45 days | ~5.10 s | **~0.22 s** | **~23x** |

*Note: Benchmark figures reflect sample measurements on a representative workstation (Python 3.10+ / modern x86_64 CPU). They illustrate relative algorithmic efficiency of vectorized array operations over iterative Python loops, rather than universal performance guarantees. Actual execution times will vary across different hardware, operating systems, and system loads.*

---

## Repository Structure

```
game-economy-simulator/
├── analytics/
│   ├── diagnostics.py       # Macro flow, Gini calculation, alert rules
│   └── optimizer.py         # Monotonic binary-search auto-tuner
├── config/
│   ├── scenarios.py         # 4-regime scenario stress matrix
│   └── settings.py          # Dataclass configurations & archetype profiles
├── simulation/
│   ├── engine.py            # Vectorized multi-agent simulation core
│   └── monte_carlo.py       # Monte Carlo risk simulation & fan chart builder
├── visualization/
│   ├── charts.py            # Plotly distributions, gauges, violin, timelines
│   ├── health_cards.py      # Styled diagnostic banners, delta comparison cards
│   └── scenario_charts.py   # Fan charts, risk quadrants, scenario comparisons
├── tests/
│   ├── test_engine.py       # Vectorization invariants & benchmark tests
│   ├── test_diagnostics.py  # Gini & flow ratio rule verification
│   ├── test_optimizer.py    # Monotonicity & convergence tests
│   ├── test_scenarios.py    # Scenario matrix configuration tests
│   └── test_monte_carlo.py  # Fan chart percentiles & risk metric tests
├── app.py                   # Streamlit multi-tab analytical cockpit
├── requirements.txt         # Dependencies (Streamlit, NumPy, Pandas, Plotly, Pytest)
└── validate.py              # Command-line benchmark & diagnostic verification script
```

---

## Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Rytsia1/Game-Economy-Simulator.git
cd Game-Economy-Simulator

# 2. Set up virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Verification & Unit Tests
```bash
# Run complete test suite (154 tests)
pytest tests/ -v

# Run command-line validation script (prints benchmarks and diagnostics)
python validate.py
```

### Launch Interactive Dashboard
```bash
streamlit run app.py
```

---

## Technical Stack
- **Language**: Python 3.10+
- **Computational Core**: NumPy (Vectorized matrix broadcasting, vectorized random draws), Pandas
- **Interactive Cockpit**: Streamlit (Session state caching, custom CSS styling)
- **Visual Analytics**: Plotly Graph Objects (Multi-percentile fan charts, violin distributions, quadrant scatter plots)
- **Quality Assurance**: Pytest (154 unit tests covering vectorization invariants, convergence bounds, and edge cases)

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
