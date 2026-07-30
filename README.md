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
uv sync --extra dev
```

The ENTSO-E API key lives outside the repo in `~/.config/pred_el_prices/.env`
(`ENTSOE_API_KEY=...`); it is never committed.

## CLI

```
uv run pep archive-weather   # archive today's 00Z ICON-EU-EPS ensemble run
uv run pep fetch-entsoe      # backfill/update the ENTSO-E cache (resumable, monthly chunks)
uv run pep fetch-smard       # SMARD day-ahead prices (keyless cross-check source)
uv run pep fetch-fuels       # daily TTF gas / coal / EUA-proxy settlements (Yahoo)
uv run pep report-qa         # render the data-QA page into reports/data_qa/
```

Weather archiving runs unattended via GitHub Actions into
[pred_el_prices_weather_archive](https://github.com/baakflo/pred_el_prices_weather_archive)
(DWD deletes files after ~24 h; the forward archive is the only history).

## Principles

- Store UTC internally; every feature must be knowable before the 12:00 CET D-1 auction gate.
- Probabilistic from the start: proper scoring rules (pinball, CRPS), calibration checks.
- No notebooks — all exploration and results render as report pages.
