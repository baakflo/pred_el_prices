# pred_el_prices

Day-ahead electricity price forecasting for the German bidding zone (DE-LU):
an open, reproducible pipeline that publishes a real pre-auction forecast
every morning and keeps score in public.

**Live forecast: [predict.baakes-systems-modeling.eu](https://predict.baakes-systems-modeling.eu)** —
tomorrow's hourly price curve, published before the 12:00 CET auction gate
closes, with the actual cleared prices filled in next to it the same
afternoon.

Goal: reproduce the open academic benchmarks (LEAR, DNN — Lago et al. 2021),
then beat them with calibrated predictive distributions built on
physics-informed features (residual load, fuel/carbon costs, merit order
structure). The model in production today is deliberately simple — a daily
recalibrated linear model (LEAR); the probabilistic phase is next. Full plan:
[docs/pred_el_prices_project_plan.md](docs/pred_el_prices_project_plan.md).

## Results so far (linear phase, complete)

Working rule throughout: hypotheses are pre-registered before the test runs;
negative results get the same report treatment as positive ones.

| What | Result | Report |
|---|---|---|
| Reproduce LEAR (Lago et al. 2021) | matches published numbers, all 4 calibration windows | [lear_reproduction](reports/lear_reproduction/) |
| Own data pipeline vs benchmark dataset, same model + horizon | ours wins at every matched config (rMAE 0.482 vs 0.506) | [lear_same_horizon](reports/lear_same_horizon/) |
| Hypothesis: splitting the RES forecast (wind on/offshore + solar) helps post-2021 | **refuted** — the aggregate wins nearly every year; floor is now LEAR(364, academic exog) **rMAE 0.407** on 2019–2026 | [lear_feature_ablation](reports/lear_feature_ablation/) |
| Hypothesis: ECMWF ensemble spread predicts daily model error | **not supported** (2023 era, 10 m wind): zero partial correlation given windiness; hub-height re-test registered | [spread_vs_error](reports/spread_vs_error/) |

(rMAE = MAE relative to the weekly-naive forecast; naive = 1.00 by
definition, lower is better. It is scale-free, so results compare across
price eras.)

Interactive explainers (how LEAR works, L1/L2 geometry, feature packaging):
[reports/explainers/](reports/explainers/). Report pages are plain HTML —
open locally or via the project site.

**Status:** linear phase wrapped; next up is the probabilistic phase
(distributional neural models scored with CRPS/pinball) — the linear model
predicts one number per hour and is structurally blind to spikes and
uncertainty, which is exactly where this market gets interesting.

## The daily forecast, honestly

Every input the model sees is knowable before the day-ahead auction gate
(12:00 CET D-1); the published forecast is logged pre-gate and never
retouched. Each site entry carries honesty flags (`pre_gate`, weather
vintage) so a degraded run is visible as such. Unattended GitHub Actions
workflows (externally triggered — see `.github/workflows/`) archive the
morning weather ensembles, produce the forecast, and fetch the cleared
prices after the auction to score completed days.

Forward-collected archives live in
[pred_el_prices_weather_archive](https://github.com/baakflo/pred_el_prices_weather_archive):
DWD deletes its open-data files after ~24 h and ENTSO-E serves only the
latest forecast version, so the forward archive is the only history that
exists.

## Reproduce it

```
uv sync --extra dev
python -m pytest
```

The only credential you need is a free ENTSO-E API key: register at
[transparency.entsoe.eu](https://transparency.entsoe.eu), request an API
token (free, takes a day), and put it in
`~/.config/pred_el_prices/.env` as `ENTSOE_API_KEY=...` (never committed).

Then build the dataset and run the headline experiment:

```
uv run pep fetch-entsoe      # prices, load + RES forecasts (resumable, monthly chunks)
uv run pep fetch-smard       # SMARD cross-check series (keyless)
uv run pep fetch-capacity    # monthly installed wind/solar capacity (energy-charts.info)
uv run pep fetch-fuels       # daily TTF gas / coal / EUA proxies (Yahoo; local only, see licenses)
uv run pep build-dataset     # leakage-safe hourly feature/target table
uv run pep run lear-de       # rolling LEAR backtest; artifacts + metrics land in runs/
uv run pep report-qa         # render the data-QA page into reports/data_qa/
```

`uv run pep --help` lists the rest (forward archivers, the daily `forecast`
command, experiment overrides via `--set`).

## Data sources & licenses

The code is MIT ([LICENSE](LICENSE)). The data comes from third parties and
is **not** all redistributable — which is why the caches are gitignored and
every result is reproducible from source instead:

| Source | Used for | Terms |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | day-ahead prices, load + wind/solar forecasts | free API key; reuse with attribution ("Source: ENTSO-E Transparency Platform") |
| [SMARD](https://www.smard.de) (Bundesnetzagentur) | cross-check prices/load/RES | CC BY 4.0 |
| [DWD open data](https://opendata.dwd.de) | ICON-EU-EPS weather ensemble | CC BY 4.0 attribution (GeoNutzV) |
| [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data) | IFS ENS weather ensemble | CC BY 4.0 ("contains modified ECMWF open data") |
| [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE) | installed capacity | CC BY 4.0 |
| [PEGELONLINE](https://www.pegelonline.wsv.de) (WSV) | Rhine level at Kaub (fuel-barge logistics) | DL-DE-BY-2.0 |
| Yahoo Finance | TTF gas / coal / EUA daily proxies | **not redistributable** — never committed anywhere; rebuild locally with `pep fetch-fuels` |

The public archive repo carries per-dataset attribution for everything it
redistributes.

## Layout

- `src/pred_el_prices/` — package: `pipeline` (data acquisition + cache), `features`, `models`, `eval`, `reporting`
- `configs/` — YAML configs for data and runs
- `data/` — local Parquet cache and forward-collected forecast archives (gitignored)
- `reports/` — rendered HTML report pages; every run/analysis becomes a page (no notebooks)
- `tests/` — pytest

## Principles

- Store UTC internally; every feature must be knowable before the 12:00 CET D-1 auction gate.
- Proper scoring, honest metrics: rMAE against the weekly naive for the linear phase; pinball/CRPS and calibration checks as the probabilistic phase lands.
- No notebooks — all exploration and results render as report pages.
- Pre-register predictions before running tests; refutations are results, not failures.
