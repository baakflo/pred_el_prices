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
- **Logs**: `site-state/site/nets_log/YYYY-MM.parquet`, one partition per month
  (99 final + 99 raw hourly percentiles per delivery hour, 99 per quarter-hour,
  generated_utc, flags incl. `backfill`); a run rewrites only its month. A single
  `nets_log.parquet` found next to it is split into partitions on first read and then
  left untouched. PIT history for recalibration is seeded from the backtest/replay
  quantiles (`nets_pit_seed.parquet`) and then grows from this log.

## Site data contract v2 (`latest.json`, `history.json`, `days/`)

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
    "trained_through": "YYYY-MM-DD", "generated_utc": "...",
    "replay": true,
    "hours":    [{"t": "...", "q": [9 values at `levels`], "actual": 0.0}],
    "quarters": [{"t": "...", "q": [9 values], "actual": 0.0}]
  }
}
```
`actual` is null until the auction result is in. Timestamps UTC ISO; quarters
are 15-min starts. Delivery blocks are UTC days: always 24 hours, 96 quarters.
`replay: true` marks rows produced by `pep backfill-nets` (a historical replay, not a
live pre-gate forecast); live forecasts carry no `replay` key. It appears in every
nets block: latest, history days and day files.

`history.json` days (last 60 scored days), v1 fields = LEAR, plus
```json
"nets": {
  "mae": 0.0, "pinball": 0.0, "cov80": 0.0, "cov90": 0.0, "cov98": 0.0,
  "mae_qh": 0.0, "pinball_qh": 0.0, "cov80_qh": 0.0, "cov90_qh": 0.0, "cov98_qh": 0.0,
  "replay": true,
  "hours": [{"t": "...", "q1": 0.0, "q5": 0.0, "q10": 0.0, "q50": 0.0,
             "q90": 0.0, "q95": 0.0, "q99": 0.0, "actual": 0.0}]
}
```
`pinball` = mean over the 99 percentiles (hourly). `cov80`, `cov90` and `cov98` are the
shares of hours inside [q10, q90], [q5, q95] and [q1, q99]. `_qh` is the same on
quarter-hours. The outer tails are kept per hour so a spike can be seen inside (or
outside) the predicted range. LEAR's `mae` stays the v1 `mae`. Days before go-live
carry no `nets` key. Size on the 60-day replay: about 455 KB as written (indent 1), or
278 KB compact.

`days/YYYY-MM-DD.json`: one per delivery day with a forecast (LEAR log, nets log, or
a curve in the published history), compact JSON, about 16 KB with nets. The same
structure as `latest.json` for that day, plus the day's scores:
```json
{
  "generated_utc": "...", "delivery_day": "YYYY-MM-DD", "model": "LEAR ...",
  "pre_gate": true, "weather_vintage": "00Z", "evening": true, "load_surrogate": true,
  "note": "...", "hours": [{"t": "...", "forecast": 0.0, "actual": 0.0}],
  "schema": 2, "levels": [1, 5, 10, 25, 50, 75, 90, 95, 99],
  "mae": 0.0, "partial": 22,
  "nets": {
    "model": "...", "trained_through": "YYYY-MM-DD", "generated_utc": "...", "replay": true,
    "hours": [{"t": "...", "q": [9 values], "actual": 0.0}],
    "quarters": [{"t": "...", "q": [9 values], "actual": 0.0}],
    "mae": 0.0, "pinball": 0.0, "cov80": 0.0, "cov90": 0.0, "cov98": 0.0,
    "mae_qh": 0.0, "pinball_qh": 0.0, "cov80_qh": 0.0, "cov90_qh": 0.0, "cov98_qh": 0.0
  }
}
```
Every `q` array has all 9 `levels`, including the 1st and 99th percentiles, on every day
(live or replayed).
`mae` (LEAR) is null until an hour is scored, and `partial` is the number of scored hours
while fewer than 24 are in. The nets scores are null until an hour or quarter-hour is
scored. `evening`, `load_surrogate`, `partial` and `replay` appear only when they apply. A
day that exists only as a published history curve has `delivery_day`, `model`,
`pre_gate`, flags and `hours` (no generated_utc, vintage or note); a nets-only day has
`delivery_day` and `nets`. A file is rewritten only when its content changes (new
actuals or scores) and is never deleted. The publish job copies them to the website's
`public/data/days/`.

## Implementation notes (2026-09-25)

Code: `production/nets.py` (training, gate row, recalibration, 15-min shape),
`production/site.py` (log + v2 JSON blocks, torch-free), `production/backfill.py`;
CLI `train-nets`, `forecast --nets-bundle`, `seed-nets-pit`, `backfill-nets`, `build-site`;
workflows `train-nets.yml` (new) and `publish-forecast.yml` (bundle download).

- **Delivery blocks are UTC days**, as in v1 and every backtest: always 24 hours and
  96 quarters. The "DST days have 23/25 hours" line above only holds for local-day
  blocks, which nothing produces today.
- **Weekly train: Saturday 18:30 UTC**, not Sunday evening. Trained through Saturday
  (= Monday - 2), the bundle then serves Sunday's run (delivery Monday) through
  Saturday's, exactly the backtest week. A Sunday-evening bundle is equally gate-safe
  but serves deliveries Tuesday..Monday (one-day shift from the backtests).
- **Recalibration runs daily** (PIT of forecast days <= D-2, 365 days), not in weekly
  steps as in the backtests/replay. Fewer than 28 PIT days: raw percentiles, flagged.
- **PIT seed** (`site-state/site/nets_pit_seed.parquet`): raw percentiles of `runs/b16`
  (8+8 blend, TSO inputs) for the 400 days before the backfill window, its
  recalibrated median (`b16-cal`, for the shape model), plus the production-path
  backfill's own log (07-27 onward) — the most production-like history there is.
- **Evening edition** (21:35 UTC, target D+2): the nets run on substitute inputs.
  - Own RES comes from the 12Z ENS run.
  - DE load comes from the load-de surrogate.
  - The summed neighbour load comes from `nets.neighbour_load_surrogate`. This is an HGB
    model per hour on the 9-zone sum, using the same hour 24 h and 7 days earlier,
    calendar, holidays and ENS weather, trained on days up to D-2.
  - The 15-min shape runs on the interpolated DE surrogate.
  - Log and nets blocks carry `evening`, `load_surrogate` and `neighbour_surrogate` (true
    only).
  - The morning run replaces the evening rows. A pre-gate morning slot retries a nets
    step that failed, and a standing evening nets forecast does not count as done.
  - The cost is registered as P53-P55 in the plan, tested with `pep evening-nets`.
- **Timing:** one bundle (24 fits) takes 15-17 min wall on a 12-core/16-thread laptop
  (16 workers); the daily nets step ~9 s incl. own RES and the shape model. Estimate
  for a 4-vCPU GitHub runner: 45-90 min (not measured). Bundle file 49 MB.
- **Log growth:** the nets log holds ~70 KB per delivery day (about 2.1 MB per month
  partition, measured on the replay). Monthly partitions bound each commit to the current
  month's file, so the state repo grows by roughly 30 MB a month (~0.4 GB a year) instead
  of quadratically. If that is still too much, store quarter rows as the 96-value shape
  instead of 99 percentiles each (about 5x smaller).
- **Rebuilding the site from logs:** `pep build-site --nets-log ... --lear-log ...
  [--history ...] [--end DAY] --out DIR` writes latest/history/days without running
  any network. `backfill-nets` uses it for its own output.

## Website (dev branch, local only until approved)

Nets fan (bands 5-95, 10-90, 25-75, median line) is the hero, 15-minute view
toggle, LEAR as a thin benchmark line; scorecard compares MAE nets vs LEAR per
day plus coverage; copy rewritten from "LEAR is the model" to "networks, LEAR
as the benchmark", EN + DE. Honest-inputs story stays prominent.
