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
CLI `train-nets`, `forecast --nets-bundle`, `seed-nets-pit`, `seed-t0-pit`, `seed-mix-learned`, `backfill-nets`, `build-site`;
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

## nets+t0 mix (`production/mix.py`, shadow from 2026-10-29)

- **What**: the equal-weight average of the nets' recalibrated percentiles and t0-beta's
  (`production/t0_shadow.py`), level by level. t0's 21 levels go onto the 99-level grid
  by PCHIP on the quantile function (monotone, exact at t0's levels, no extrapolation
  beyond its q01/q99). Quarter-hours = the nets' quarter-hours shifted per level by what
  the mix moved their hour (= the mixed hours + the nets' median shape), sorted, clipped.
- **t0 is recalibrated** where it is consumed (t0 block, mix, mix log), never in the t0
  log (raw, as the model made it): the nets' rule (`models.recalibrate.recalibrate_day`)
  on the 99-level grid, PIT of t0's own past forecasts against the clearing prices,
  forecast days <= D-2 in a 365-day window, raw below 28 PIT days. Any edition, as the
  nets: the standing row per hour (morning replaces evening). Reproducible from the logs:
  every site build recalibrates each day on its own history.
- **t0 PIT seed** (`site-state/site/t0_pit_seed.parquet`, `pep seed-t0-pit --quantiles
  runs/t0-compare/t0_B.parquet --before 2026-09-30`): the P65-P69 arm B year of the
  neural repo, 2025-09-01..2026-09-24 (384 days; 08-12..08-17 skipped there), t0's 21
  levels. Gate-safe on the shadow's conventions (context to local midnight of D,
  residual load = TSO load - own RES of D from the 00Z ENS, i.e. the morning edition;
  same weights, which MODEL_REVISION pins). The t0 log wins where both exist; without
  the seed only days with 28 live PIT days recalibrate (from 2026-10-30).
- **Runs**: t0 runs in both editions (evening rows flagged) before the site JSON; every
  run with the nets logs the mix to `site-state/site/mix_log/YYYY-MM.parquet` (nets log
  layout, final percentiles only, the nets run's flags, `components` = `nets+t0` or
  `nets` when t0 failed, `nets_generated_utc`, `t0_recal_days`). Never across editions:
  a morning run whose nets failed logs no mix. A pre-gate morning retry reruns t0 + mix when no morning mix
  with t0 stands.
- **Publishing**: `pep forecast --site-models nets,t0,mix --primary mix --mix-from
  YYYY-MM-DD` (workflow: repository variables `PEP_SITE_MODELS`, `PEP_PRIMARY`,
  `PEP_MIX_FROM`, or the dispatch input `primary` for one run). Default `--site-models
  nets`: neither the t0 nor the mix log is read and the site JSON is byte-identical to
  the nets-only site. LEAR stays as it is either way (v1 fields, `forecast_log`).
  - **Sibling blocks** `t0` and `mix` next to `nets`, wherever `nets` appears (latest,
    history days, day files), in the nets block's shape: own `model`, the 9 `levels`,
    hours, quarters, the same scores on the 99 levels, the same flags. Absent for a day
    without them (history days before the t0 log: none). The `nets` block is always the
    nets.
  - **t0**: its standing log rows on the 99-level grid (PCHIP, as for the mix),
    recalibrated; no `trained_through` (zero-shot). Quarters = t0's recalibrated hours +
    the nets' within-hour shape of the day, from whichever nets edition stands; a day
    without nets rows repeats t0's hour (`flat_quarters: true`). The t0 and mix blocks
    carry `t0_recal_days` (t0's PIT days) and `t0_uncalibrated: true` when that is 0
    (raw t0); the nets block does not.
  - **mix**: from the mix log where a mix with recalibrated t0 (`t0_recal_days` logged)
    stands on the day's standing nets run,
    else mixed at site-build time from the standing nets and t0 rows (same math), so the
    blocks reach back to the start of the t0 log (2026-09-24). Adds `nets_generated_utc`
    and `t0_generated_utc` (its `generated_utc` = the later of the two). Flags are either
    component's, and `mixed_editions: true` marks nets and t0 from different editions
    (2026-10-25: the morning nets failed, the evening nets stood next to the morning t0).
  - **`primary`** (per day: latest, history days, day files): the model the site shows
    as its headline fan, `nets` | `mix` | `t0`. `--primary` from the cut-over
    `--mix-from` (UTC delivery day, required unless `nets`) on, `nets` before. A day
    without that model's block falls back to `nets`, recorded as `"primary_fallback":
    "<wanted>"`. Days without any model block (LEAR only) carry no `primary`.
  - **History merge**: a block of a published history day survives a loss of its log, as
    the nets blocks do; back on `--site-models nets` the t0/mix blocks and `primary` are
    dropped.

```json
"primary": "mix",
"nets": {...},
"t0":  {"model": "t0-beta foundation model (zero-shot), 21 levels on the 99-level grid, recalibrated",
        "generated_utc": "...", "t0_recal_days": 354, "hours": [...], "quarters": [...]},
"mix": {"model": "24 networks + t0-beta foundation model, equal-weight quantile mix",
        "trained_through": "YYYY-MM-DD", "generated_utc": "...",
        "nets_generated_utc": "...", "t0_generated_utc": "...", "mixed_editions": true,
        "t0_recal_days": 354,
        "evening": true, "hours": [...], "quarters": [...]}
```

### Learned mixes M1 / M3 (`production/mix_learned.py`, shadow log only)

- **What**: the mix with a fitted nets weight w instead of 0.5, q = w·q_nets + (1−w)·q_t0
  level by level (clipped), on exactly the equal mix's inputs (nets final, t0 recalibrated
  on the 99 grid). Settings = the neural repo's learned-mix experiment (`_scratch/mix/`,
  `runs/mix-learned`, plan Step 1, results 2026-11-03), chosen there on Oct–Mar:
  - **M1**: one w per UTC hour, minimising window-mean 99-level pinball + λ(w−0.5)², λ = 1,
    + μ·Σ(w_h−w_h+1)², μ = 3; exact on a 0.02 grid (dynamic programming over the hours).
  - **M3**: w = sigmoid(θ·x), 16 gate-safe features (hour sin/cos ×3, Saturday, Sunday,
    holiday share, residual load = DE load − own RES, own-RES share, D−1 mean price UTC
    00–21, nets−t0 median spread signed and absolute, both q90−q10 widths), standardised on
    the window, ridge 0.1. Refit every run, cold start (the experiment refit weekly,
    warm-started: weights differ by ≤ 3e-4).
  - Fit window: the last 90 calendar days up to D−2 with both models and all 24 prices
    (M3: and complete features). Fewer than 60 such days: w = 0.5, `fallback: true`.
  - Parity with the experiment, fed its inputs: M1 weights identical on all 172 report-half
    days; M3 max |Δw| 3.2e-4, max |Δq| 0.05 €/MWh (12 days). The seed reproduces the
    experiment's arrays exactly (175 days, nets/t0/features/prices Δ = 0).
- **History**: nets = the standing nets log rows from 2026-10-17 on (`NETS_LIVE_FROM`:
  the live nets before ran with EUA = 0), else the seed's; t0 = the standing t0 log rows,
  each day recalibrated on its own PIT history (as the site does), else the seed's; live
  wins per component. M3 features: those logged with M3's rows, else the seed's, else
  rebuilt from the dataset (`proxy_days`): TSO load forecast as the run had it, and the
  TSO RES forecast in place of the unlogged own RES (a published value for a day <= D−2;
  the experiment used own RES). Only live days before the M3 log starts need this.
  - Day-D features: the run's own DE load input (UTC 22–23 from 24 h earlier or the
    surrogate; the experiment's `load_tso` had the TSO's own value there), own RES of D
    (12Z in the evening edition), holiday share, D−1 prices.
- **Seed** (`site-state/site/mix_learned_seed.parquet`, 175 days, 4.9 MB), built from the
  neural repo's backtests as the experiment built them (nets = morning replay log, t0-B
  recalibrated with its own rows as PIT history, features from the t0 comparison's inputs):

  ```
  pep seed-mix-learned --nets-logs runs/replay-2526/morning/nets_log.parquet \
      --t0 runs/t0-compare/t0_B.parquet --inputs runs/t0-compare/inputs \
      --cache-dir data/cache --before 2026-09-25 --out site-state/site/mix_learned_seed.parquet
  ```
  Without it M1/M3 log `fallback: true` (w = 0.5) until 60 live days exist; nothing fails.
- **Log**: `mix_m1_log/` and `mix_m3_log/` (monthly partitions), every run with t0 in both
  editions, the mix log's layout (24 `h` + 96 `qh` rows; quarters = the nets' quarters
  shifted by what the mix moved their hour) and flags (evening etc., `components`,
  `nets_generated_utc`, `t0_recal_days`), plus `variant`, `w` (the hour's weight, repeated on
  its quarters), `fallback`, `fit_days`, `seed_days`; M3's `h` rows add the raw features
  `f_load`, `f_res`, `f_holiday`, `f_price_dm1` (the history for later fits) and
  `proxy_days`. The equal `mix_log/` is not touched. No t0 rows: nothing logged.
- **Fail-soft**: any error logs `::warning::learned mix ...` and skips only those rows (one
  variant or both); the forecast, the equal mix and the site JSON are unaffected. Not on
  the site, not a `--primary` choice. Runtime ~1–2 s per run (history 0.5 s on the real
  seed, M1 0.3 s, M3 0.1 s).

## Website (dev branch, local only until approved)

Nets fan (bands 5-95, 10-90, 25-75, median line) is the hero, 15-minute view
toggle, LEAR as a thin benchmark line; scorecard compares MAE nets vs LEAR per
day plus coverage; copy rewritten from "LEAR is the model" to "networks, LEAR
as the benchmark", EN + DE. Honest-inputs story stays prominent.
