# Game Economy Simulator & Tuning Engine
> A high-performance, vectorized stochastic simulation framework to design, stress-test, and auto-tune virtual game economies prior to production.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/Core-NumPy%20Vectorized-013243.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-154%20Passed-00C9A7.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem
Most virtual game economies are conceived on static spreadsheets. While spreadsheets model deterministic averages effectively, they break down in production because they fail to capture:

1. **Behavioral Variance**: Players do not behave uniformly. High-velocity grinders flood the economy with gold, while casual players face severe progression blockers under the same global parameters.
2. **Stochastic Volatility**: Non-deterministic quest completions and randomized loot drops generate fat-tailed wealth distributions, causing emergent poverty traps or runaway hyper-inflation that static formulas obscure.
3. **Pacing Bottlenecks**: Determining exactly *when* target archetypes can afford critical progression milestones (e.g., Tier-1 or Tier-2 gear) requires iterative, multi-agent temporal simulation, not static division.

---

## Solution Architecture
This engine reframes game economy balancing as a quantitative simulation, diagnostic, and numerical optimization problem. Built on vectorized 1D/2D NumPy array broadcasting, it simulates **10,000 players over 90 days in under 0.2 seconds**, providing instant feedback loops for systems designers and technical directors.

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
- **Casual (50% population)**: Low quest completion rate (0.6x), low combat frequency (0.4x), conservative buffer spending. Highly vulnerable to poverty traps.
- **Grinder (25% population)**: High engagement (1.2x quests, 2.0x combat farming), rapid gold velocity. Stresses source caps and tests late-game sinks.
- **Collector (15% population)**: Moderate income (0.8x quests), aggressive milestone sink consumption (buys gear the moment funds suffice).
- **Optimizer (10% population)**: High income efficiency, minimal consumable expenditure (0.3x potions), strategic wealth hoarding (requires 2.0x buffer before spending).

### 2. Automated Economy Diagnostics ("Economy Doctor")
Monitors macroeconomic health using quantitative criteria and actionable systems-design recommendations:
- **Macro Flow Ratio (Sources / Sinks)**: Categorizes systemic stability into `Critical Inflation` (>= 2.0x), `Moderate Inflation` (1.3x–2.0x), `Balanced` (0.9x–1.3x), or `Severe Deflation` (< 0.9x).
- **Gini Coefficient**: Computes Lorenz-curve inequality across the simulated population (O(N log N) vectorized sorting) to detect whether wealth concentration renders sinks unattainable for non-grinders.
- **Progression Blocker Detection**: Quantifies the exact percentage of casual players unable to afford baseline milestones within expected design windows.

### 3. Target-Driven Auto-Tuning Solver
Replaces manual trial-and-error balancing with an inverse root-finding optimization loop. 
- Designers define a design constraint (e.g., *"The median Casual player must afford Tier-1 Weapon on Day 20"*).
- The engine uses a monotonic binary search solver with bounded search spaces to find the exact parameter value (e.g., quest reward or potion price) within +/- 1 day precision in under 0.5 seconds.

### 4. Scenario Stress-Testing Matrix
Executes side-by-side comparative stress-tests across four distinct economic regimes:
1. **Normal (Baseline)**: Default balanced equilibrium across all sources and sinks.
2. **Gold Rush (Hyper-Inflation)**: Quest rewards +100%, enemy drops +80%, unadjusted sinks.
3. **Economic Crisis (Depression)**: Gold sources -30%, item prices +25%, potion usage +20%.
4. **Hardcore Shift (Demographic Shock)**: Population skewed to 50% Grinders, 30% Optimizers, 10% Casuals.

### 5. Monte Carlo Tail-Risk Quantification
- Simulates 100+ independent economic paths applying macro shocks (volatility sigma) to baseline reward rates and sink costs.
- Aggregates cross-run percentiles (P5, P25, P50, P75, P95) into interactive confidence interval fan charts.
- Quantifies objective failure probabilities: **P(Inflation Risk)** and **P(Poverty Trap Risk)** with a composite **System Stability Score**.

---

## Performance Benchmarks
Simulating individual player agent objects in pure Python incurs massive overhead ($O(N \times D)$ object allocations). By restructuring the engine into 1D/2D broadcasted NumPy arrays:

| Workload | Scale | Pure Python Loop | Vectorized Engine | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Single Run Simulation | 10,000 players x 90 days | ~4.80 s | **~0.18 s** | **26x** |
| Auto-Tuner Optimization | 15 iterations x 1,000 players | ~6.20 s | **~0.38 s** | **16x** |
| Monte Carlo Analysis | 100 runs x 1,000 players x 30 days | ~32.50 s | **~1.25 s** | **26x** |
| Complete Stress-Test Matrix | 4 regimes x 2,000 players x 45 days | ~5.10 s | **~0.22 s** | **23x** |

*Benchmarks conducted on Python 3.10+ (AMD/Intel x86_64).*

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
├── requirements.txt         # Production dependencies
└── validate.py              # Zero-dependency terminal verification suite
```

---

## Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/game-economy-simulator.git
cd game-economy-simulator

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
- **Interactive Cockpit**: Streamlit (Dynamic session state, custom CSS responsive design system)
- **Visual Analytics**: Plotly Graph Objects (Multi-percentile fan charts, violin distributions, quadrant scatter plots)
- **Quality Assurance**: Pytest (154 unit tests covering mathematical invariants, convergence bounds, and execution latency)

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
