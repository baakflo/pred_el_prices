# pred_el_prices

Probabilistic day-ahead electricity price forecasting for the German bidding zone (DE-LU).

Goal: reproduce the open academic benchmarks (LEAR, DNN — Lago et al. 2021), then beat them
with calibrated predictive distributions built on physics-informed features (residual load,
fuel/carbon costs, merit order structure). Full plan: [docs/pred_el_prices_project_plan.md](docs/pred_el_prices_project_plan.md).

## Results so far (linear phase, complete)

Working rule throughout: hypotheses are pre-registered before the test runs;
negative results get the same report treatment as positive ones.

| What | Result | Report |
|---|---|---|
| Reproduce LEAR (Lago et al. 2021) | matches published numbers, all 4 calibration windows | [lear_reproduction](reports/lear_reproduction/) |
| Own data pipeline vs benchmark dataset, same model + horizon | ours wins at every matched config (rMAE 0.482 vs 0.506) | [lear_same_horizon](reports/lear_same_horizon/) |
| Hypothesis: splitting the RES forecast (wind on/offshore + solar) helps post-2021 | **refuted** — the aggregate wins nearly every year; floor is now LEAR(364, academic exog) **rMAE 0.407** on 2019–2026 | [lear_feature_ablation](reports/lear_feature_ablation/) |
| Hypothesis: ECMWF ensemble spread predicts daily model error | **not supported** (2023 era, 10 m wind): zero partial correlation given windiness; hub-height re-test registered | [spread_vs_error](reports/spread_vs_error/) |

Interactive explainers (how LEAR works, L1/L2 geometry, feature packaging):
[reports/explainers/](reports/explainers/). Report pages are plain HTML —
open locally or via the project site.

**Status:** linear phase wrapped; next up is the probabilistic phase
(distributional neural models scored with CRPS/pinball) — the linear model
predicts one number per hour and is structurally blind to spikes and
uncertainty, which is exactly where this market gets interesting.

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
- Pre-register predictions before running tests; refutations are results, not failures.

## License

MIT — see [LICENSE](LICENSE).
