# Dashboard Screenshots

Place dashboard screenshot files in this folder and update the image paths in `README.md`.

## Recommended Screenshots

| Filename | Tab / View | Notes |
|---|---|---|
| `dashboard_simulation.png` | Tab 1 – Simulation | Wealth progression + archetype breakdown charts |
| `dashboard_economy_doctor.png` | Tab 2 – Economy Doctor | Health banner, Gini gauge, alert cards visible |
| `dashboard_auto_tuner.png` | Tab 3 – Auto-Tuner | Convergence chart + before/after KPI comparison |
| `dashboard_stress_test.png` | Tab 4 – Stress-Test | 4-scenario side-by-side comparison |
| `dashboard_monte_carlo.png` | Tab 5 – Monte Carlo | Fan chart with P5–P95 confidence bands |

## How to Capture

1. Run `streamlit run app.py`
2. Open each tab and configure a representative scenario
3. Take a full-browser screenshot at 1280×800 or wider
4. Save to this `docs/` folder using the filenames above
5. Uncomment the image lines in `README.md`

## Recommended GIF (optional)

A short screen recording (~15 seconds) showing:
- Adjusting a slider on the Simulation tab
- Switching to Economy Doctor and seeing the health banner update
- Running the Auto-Tuner and watching the convergence chart populate

Save as `docs/demo.gif` (keep under 5 MB for GitHub rendering).
