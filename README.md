# pred_el_prices

Probabilistic day-ahead electricity price forecasting for the German bidding zone (DE-LU).

Goal: reproduce the open academic benchmarks (LEAR, DNN — Lago et al. 2021), then beat them
with calibrated predictive distributions built on physics-informed features (residual load,
fuel/carbon costs, merit order structure). Full plan: [docs/pred_el_prices_project_plan.md](docs/pred_el_prices_project_plan.md).

## Layout

- `src/pred_el_prices/` — package: `pipeline` (data acquisition + cache), `features`, `models`, `eval`, `reporting`
- `configs/` — YAML configs for data and runs
- `data/` — local Parquet cache and forward-collected forecast archives (gitignored)
- `reports/` — rendered HTML report pages; every run/analysis becomes a page (no notebooks)
- `tests/` — pytest

## Setup

```
pip install -e ".[dev]"
copy .env.example .env   # then fill in ENTSOE_API_KEY
```

## Principles

- Store UTC internally; every feature must be knowable before the 12:00 CET D-1 auction gate.
- Probabilistic from the start: proper scoring rules (pinball, CRPS), calibration checks.
- No notebooks — all exploration and results render as report pages.
