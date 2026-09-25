# Production v2: networks + 15-minute layer (go-live 2026-10-01)

Design note shared by the forecast pipeline and the website rebuild. Decisions
from the Phase 2 roadmap (see `pred_el_prices_project_plan.md`): 12 JSU + 12
quantile-head networks, Vincentized, rolling PIT recalibration, clipped to SDAC
bounds; 15-minute percentiles = hourly percentiles + one median shape
(`models/qh_shape.py`). LEAR stays live as the benchmark.

## Inputs (all gate-available; see the 60-day replay, `experiments/qnn_replay.py`)

| Input | Source | Status in production today |
|---|---|---|
| DE prices, lags d-1..d-7 (hours 22-23 of d-1 filled from d-2) | ENTSO-E / SMARD / energy-charts caches | fetched daily |
| DE load forecast (hourly + native 15 min) | ENTSO-E 6.1(b), pre-gate | fetched daily |
| DE wind/solar | own RES forecast (`daily_forecast.own_res_forecast`), never the TSO's 18:00 forecast | computed daily |
| Neighbour load forecasts FR NL BE AT PL CZ CH DK1 DK2 (sum) | ENTSO-E 6.1(b), pre-gate | **not fetched** -> add to the daily fetch + seed history |
| TTF gas, EUA proxy (settlements lagged 2 days) | Yahoo proxies (`pipeline/fuels.py`) | **not fetched** -> add + seed |

## Jobs

- **Weekly train** (new workflow, public repo, GitHub-hosted runner; Sunday
  evening UTC): build the dataset in memory from the state caches, fit 12 JSU +
  12 quantile networks on the trailing 1460-day window ending 2 days before the
  coming Monday (same convention as the backtests), persist a model bundle
  (weights + both scalers + config + training window) as a release asset
  `models-latest` on the public repo (replaced weekly; not committed to git).
  If a week's training fails, the previous bundle keeps serving.
- **Daily predict** (inside the existing `pep forecast`, pre-gate slots):
  load the bundle, build the delivery day's gate-safe row exactly as the replay
  does, forward pass (24 networks), Vincentize, recalibrate with the rolling
  365-day PIT window, clip to [-500, 4000]; fit the 15-minute shape model
  (HGB, cheap) and produce quarter-hour percentiles. LEAR runs as before.
- **Logs**: `site-state/site/nets_log.parquet` (99 hourly percentiles per
  delivery hour + 99 per quarter-hour, generated_utc, flags). PIT history for
  recalibration is seeded from the backtest/replay quantiles and then grows
  from this log.

## Site data contract v2 (`latest.json`, `history.json`)

Backward compatible: all v1 fields stay (they carry LEAR), v2 adds keys.

`latest.json`
```json
{
  "schema": 2,
  "generated_utc": "...", "delivery_day": "YYYY-MM-DD",
  "pre_gate": true, "weather_vintage": "00Z", "evening": false, "load_surrogate": false,
  "model": "LEAR ...", "hours": [{"t": "...", "forecast": 0.0, "actual": 0.0}],
  "levels": [1, 5, 10, 25, 50, 75, 90, 95, 99],
  "nets": {
    "model": "24 networks (12 JSU + 12 quantile), recalibrated",
    "trained_through": "YYYY-MM-DD",
    "hours":    [{"t": "...", "q": [9 values at `levels`], "actual": 0.0}],
    "quarters": [{"t": "...", "q": [9 values], "actual": 0.0}]
  }
}
```
`actual` is null until the auction result is in. Timestamps UTC ISO; quarters
are 15-min starts. DST days have 23/25 hours (92/100 quarters).

`history.json` days (last 60 scored days), v1 fields = LEAR, plus
```json
"nets": {
  "mae": 0.0, "pinball": 0.0, "cov80": 0.0, "mae_qh": 0.0, "pinball_qh": 0.0,
  "hours": [{"t": "...", "q10": 0.0, "q50": 0.0, "q90": 0.0, "actual": 0.0}]
}
```
`pinball` = mean over the 99 percentiles (hourly), `cov80` = share of hours
inside [q10, q90], `_qh` = the same on quarter-hours. LEAR's `mae` stays the
v1 `mae`. Days before go-live carry no `nets` key.

## Website (dev branch, local only until approved)

Nets fan (bands 5-95, 10-90, 25-75, median line) is the hero, 15-minute view
toggle, LEAR as a thin benchmark line; scorecard compares MAE nets vs LEAR per
day plus coverage; copy rewritten from "LEAR is the model" to "networks, LEAR
as the benchmark", EN + DE. Honest-inputs story stays prominent.
