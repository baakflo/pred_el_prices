# German Day-Ahead Electricity Price Forecasting: Implementation Plan

## Context and goal

Solo project by a computational scientist (battery/energy domain background, strong ML) to build a **probabilistic day-ahead electricity price forecaster for the German bidding zone (DE-LU)** that:

1. Reproduces and then beats the open academic benchmarks (LEAR, DNN from Lago et al. 2021, epftoolbox).
2. Produces calibrated **predictive distributions**, not point forecasts, with special attention to spikes and negative prices.
3. Uses **physics-informed features** (residual load, merit order structure, fuel/carbon costs) as the differentiator.
4. Is fully reproducible on free/open data, suitable for public write-ups (LinkedIn/blog series) and later a live daily forecast page.

## Key design principles (agreed in prior discussion)

- **Never learn weather-to-price directly.** Weather uncertainty enters via the TSO day-ahead forecasts (wind, solar, load), which are already capacity-normalized. This makes the model robust to renewable capacity growth.
- **Make non-stationarity explicit instead of implicit.** Fuel (TTF gas) and carbon (EUA) prices as features so the biggest merit order shifts become inputs, not drift. Rolling recalibration window (retrain daily/weekly on trailing 1 to 2 years) for the rest.
- **Residual load is the central engineered feature** (load forecast minus wind forecast minus solar forecast). Price vs residual load approximates the merit order curve (hockey stick). Engineer it explicitly, do not hope the network learns the subtraction.
- **Probabilistic from the start.** Train with proper scoring rules (pinball/quantile loss, or parametric NLL with a skewed heavy-tailed family like Johnson's SU). Evaluate with CRPS, pinball loss, and calibration/reliability diagrams. Point metrics (MAE, rMAE) reported too for benchmark comparability.
- **Honest evaluation is part of the brand.** Rolling out-of-sample test protocol matching the Lago benchmark conventions, statistical significance tests (Diebold-Mariano), all code and data pipeline public.

## Data sources (all free)

| Data | Source | Notes |
|---|---|---|
| Day-ahead prices DE-LU | ENTSO-E Transparency API | Free API key on registration. NOTE: Germany moved to 15-min MTU in 2025; harmonize to hourly for the historical benchmark, keep raw 15-min for later. |
| Load forecast + actual | ENTSO-E | Day-ahead TSO forecast, published pre-auction |
| Wind + solar day-ahead forecasts | ENTSO-E | These are the capacity-normalized weather proxies |
| Generation by type (actuals) | ENTSO-E | For analysis, not as pre-auction features |
| Scheduled exchanges, NTC, outages | ENTSO-E | Cross-border and availability features |
| Cross-check for German data | SMARD.de API, Energy-Charts (Fraunhofer ISE) API | Often cleaner for DE |
| TTF gas, coal, EUA carbon (daily) | Stitched: Ember (free carbon data), EEX public pages, Yahoo Finance proxies | Daily settlement level is sufficient; slow-moving features |
| Neighbor fundamentals (FR load/nuclear outages, Nordic, PL/CZ) | ENTSO-E | Phase 2, ablation experiment |
| Historical weather reanalysis (optional) | ERA5 via Copernicus CDS | Hindsight weather; use with care (not what forecasters knew) |
| Ensemble weather forecasts (later) | DWD ICON-EPS open data, ECMWF open data | Free archives are short; start a cron archiver early |

Python: `entsoe-py` client library for ENTSO-E. Rate limits are annoying; build a local cache (Parquet) once, incremental updates after.

Data volume is trivial: ~10 years hourly, 50 to 100 columns, well under 1 GB. Fits in RAM; everything runs on a laptop.

## Phase 0: Environment and data pipeline

1. Repo scaffold: `src/` (pipeline, features, models, eval), `data/` (gitignored Parquet cache), `reports/` (generated HTML, see Reporting workflow below), `configs/`, tests, CI. **No notebooks anywhere.** All exploration, QA output, and results go through rendered report pages.
2. ENTSO-E downloader with polite pagination, retry, and local Parquet cache. Target: 2015 to present, DE-LU (and DE-AT-LU pre-Oct-2018; handle the bidding zone split explicitly).
3. Data QA: missing hours, DST transitions (23h/25h days: handle in UTC internally, convert for display), the 2025 15-min switch, outlier sanity checks. Write a small data-validation report.
4. Fuel/carbon price stitcher (daily series, forward-filled to hourly).
5. **Weather forecast archiver, build on day one, not optional:** a cron job that fetches and archives, every day, (a) ICON-EPS ensemble fields relevant to wind/solar/temperature over Germany (DWD open data) and (b) the ENTSO-E day-ahead forecasts exactly as published, timestamped. Free archives of forecast data are short, so the only way to have an ensemble training set is to collect it forward. Target: a usable dataset in ~3 months, feeding Phase 4. Make the archiver robust (retries, gap detection, alerting on missed days) since silent gaps are unrecoverable.

Deliverable: one command produces a clean feature/target table from raw APIs.

## Reporting and review workflow (applies to every phase)

- Every unit of work renders **immediately as a webpage**: data QA reports, feature analyses, each individual training run (config, metrics, calibration plots, comparison to baselines), ablations, benchmark tables. No notebooks, no loose PNGs; a run that isn't on a page doesn't exist.
- These pages are the review surface during development and become the polished public write-ups later.
- Style and page conventions: read them from the existing website project at `../website` (relative to this repo). Match its structure and components rather than inventing a new template; a rough default style is fine at first, polishing happens later, but build on the site's conventions from the start.
- Practical shape: static site generation from run artifacts (e.g. each run writes a JSON/Parquet summary plus figures, a small builder renders pages into `reports/`), so the whole history of experiments stays browsable.

## Phase 1: Baselines (reproduce, do not innovate yet)

1. Naive baselines: similar-day (same hour yesterday / 7 days ago).
2. **LEAR** (LASSO-estimated autoregressive with the standard feature set: price lags 1/2/7 days, load forecast, wind+solar forecast, calendar dummies). Use epftoolbox conventions; validate against published numbers on their datasets first, then run on the fresh German data.
3. **DNN benchmark** (Lago-style MLP, hyperparameters via their protocol or Optuna).
4. Evaluation harness: rolling daily re-estimation, out-of-sample windows covering distinct regimes (pre-crisis 2019 to 2020, crisis 2021 to 2022, post-crisis 2023+). Metrics: MAE, rMAE, RMSE, Diebold-Mariano tests. This harness is reused everywhere.

Deliverable: benchmark table on modern German data, including how badly crisis-era shift hurts each model. (This alone is publishable content: "the academic SOTA meets 2022".)

## Phase 2: Probabilistic models (the core contribution)

1. Quantile regression versions: start with **LEAR-quantile / linear quantile regression** as a probabilistic baseline, then gradient boosting (LightGBM with pinball loss, one model per quantile or multi-quantile), then a **multi-quantile neural net** (99 quantiles, shared trunk, monotonicity enforced by sorting or non-crossing penalty).
2. **Distributional net (DDNN-style):** network outputs Johnson's SU parameters, trained by NLL. Compare against quantile approach.
3. Evaluation: CRPS, average pinball loss, empirical coverage of 50%/90% intervals, reliability diagrams, and **tail-specific metrics** (pinball at q95/q99, spike hit-rate where spike = price above a rolling high percentile; separate scoring for negative-price hours).
4. Feature ablations on the physics features: with/without residual load as explicit feature, with/without fuel+carbon, with/without neighbor fundamentals (FR nuclear availability first). Each ablation is a content piece.

Deliverable: a probabilistic model demonstrably better-calibrated than the baselines, with the ablation evidence for which physics features matter.

## Phase 3: Robustness to regime change (the narrative experiment)

Train naive-features model vs structured model (explicit residual load + fuel/carbon inputs) on 2019 to 2020 only; test both frozen on 2022 to 2023. Expected result: naive collapses, structured degrades gracefully. Then show rolling recalibration closes most of the remaining gap. Produce the comparison chart. This is the flagship story: "make the merit order explicit and your model survives a crisis."

## Phase 4: Weather-uncertainty extension (iteration two, only after 1 to 3 are solid)

1. **Ensemble spread as feature:** std/quantiles of wind+solar generation across ICON-EPS members (from the archive started in Phase 0) as additional inputs to the distributional model. Test on spike days: does knowing "today's weather forecast is unusually uncertain" widen the predicted distribution correctly?
2. Optional later: scenario propagation (run each ensemble member through a weather-to-generation model consistent with TSO forecasts, mix the resulting price distributions).

## Phase 5: Going live (credibility engine)

1. Daily automated pipeline: fetch pre-auction data, produce tomorrow's hourly predictive distributions before the 12:00 CET auction, publish timestamped (simple static page or repo commit).
2. Public running scorecard: realized prices vs predicted distributions, cumulative CRPS and coverage, updated daily. Six months of honest timestamped forecasts is the credential.
3. Content cadence: each phase deliverable maps to a LinkedIn post / blog article.

## Guardrails and known pitfalls

- **No data leakage:** every feature must be knowable before the day-ahead auction gate closure (12:00 CET D-1). Actual generation/load are analysis-only, never inputs. Be paranoid about ENTSO-E publication timestamps.
- **Bidding zone history:** DE-AT-LU split Oct 2018; either start the modeling dataset at 2019 or handle the break explicitly.
- **DST and timezones:** store UTC, join carefully; the 23/25-hour days break naive hourly reshaping.
- **Price caps and negative prices:** EPEX day-ahead has technical price bounds; distributions should respect them (censoring or bounded families are acceptable refinements later).
- **Don't over-transform prices:** common tricks (log transform) break on negative prices; use asinh or model raw with heavy-tailed families.
- **Benchmark honestly:** claims of beating SOTA only against reproducible baselines run under identical rolling protocols, with DM tests. No cherry-picked windows.
- **Scope discipline:** Germany-only until Phase 2 is done. Neighbors, intraday markets, 15-min resolution, and BESS dispatch optimization are explicitly out of scope for v1 (dispatch optimization is the natural sequel project).

## Success criteria

1. Reproduced LEAR/DNN numbers within tolerance on the reference datasets.
2. Probabilistic model with correct empirical coverage (within a few percentage points at 50% and 90%) and CRPS improvement over quantile-LEAR, DM-significant.
3. Documented ablation results for residual load, fuel/carbon, and FR availability features.
4. The Phase 3 regime-change chart.
5. Pipeline runs end-to-end unattended for the daily forecast.
---

## Roadmap (2026-09-24): after the quantile network

State: the best model is the tuned, recalibrated quantile network (`qnn-de-20260923-182551`,
`-cal`). Report page `reports/qnn_architecture/` (published as an artifact). Five work items
from the user, in suggested order. Every experiment is registered here before its run.

**Q&A behind the items.**
- *Why scale by a gas plant's cost?* In the hours that set the German price, the
  marginal plant is mostly gas, sometimes coal, and its bid is roughly fuel plus carbon
  cost. When gas went up 10× in 2021–22, the whole price curve scaled with it
  multiplicatively. A network trained in € learns patterns at its training window's
  level and trails a level shift: v1 lagged 2022 by −26 €/MWh. In units of gas cost the
  target is closer to stationary. An evening at 1.3× gas cost and a solar noon at 0.2× look
  the same whether gas costs 20 or 200. In renewable-surplus hours (price near or below
  zero) gas is not marginal and the scaling carries no meaning, but it does no harm there
  either: small numbers stay small. Caveat: the level fix changed three things at once
  (rolling window, weekly refits, fuel scaling). **Ablation owed:** tuned network without
  fuel scaling.
- *How many seeds?* Seed noise in an averaged forecast shrinks like σ/√n. Going from 4 to 9
  seeds removes a third of it, 4 to 16 half. There is no universal number: the papers
  use 4 (Lago et al. 2021, Marcjasz et al. 2023), set by compute, not measurement.
  Measure it (item 1).

**1. Seed count, measured.** On 2025 (validation), 16 seeds per 4-week refit with the
tuned configuration, per-seed percentiles saved. For k = 1…16, the pinball of 50 random
k-subsets averaged, giving a curve of mean pinball and its spread over k. Pick the smallest k
whose expected pinball is within 0.5 % of k = 16 and whose subset spread is below the
DM-detectable difference. Expectation: the knee lies at 8–12. Cost ≈ 16 × 14 fits, about 20
min on a pod. Production cost of k = 12 weekly: 12 fits × ~2 min, parallel, so trivial.

*Registered (2026-09-24, before the run).* Run: `qnn-de` with the tuned configuration
(3×256, dropout .278, lr 4.98e-4, wd 8.55e-5, batch 32, window 1460, fuel scaling), first
fit 2025-01-01, test end 2025-12-31, refit every 4 weeks, 16 seeds, `save_seeds=true`.
Scorer `_scratch/phase2_nonlinear/seed_curve.py`: 50 random k-subsets per k, Vincentized,
mean pinball; fit pinball(k) = L∞ + c/k.
- **P35.** The smallest k within 0.5 % of the 16-seed pinball is between 6 and 12.
- **P36.** Single seeds differ: max − min pinball over the 16 exceeds 1 % of the 16-seed
  pinball, and the 16-seed ensemble beats the average single seed by 1–3 %.
- **P37.** Four seeds (today's setting) are within 1 % of 16: more seeds are a cheap
  cleanup, not a big lever.

*Result (run `qnn-de-20260924-074708`, 2025, 8,760 h).* Mean pinball of k-seed
ensembles (50 random subsets each), gap to 16 seeds, median DM of the subset vs all 16:

| k | pinball | gap to 16 | subset sd | DM vs 16 |
|---|---|---|---|---|
| 1 | 4.587 | +4.86 % | 0.062 | 4.9 |
| 2 | 4.472 | +2.22 % | 0.041 | 3.7 |
| 4 | 4.417 | +0.98 % | 0.028 | 2.4 |
| 6 | 4.405 | +0.69 % | 0.018 | 2.3 |
| 7 | 4.393 | +0.44 % | 0.021 | 1.5 |
| 8 | 4.385 | +0.25 % | 0.017 | 1.1 |
| 10 | 4.381 | +0.15 % | 0.010 | 0.9 |
| 12 | 4.380 | +0.13 % | 0.009 | 1.2 |
| 16 | 4.374 | 0 | – | – |

Fit pinball(k) = 4.361 + 0.226/k, so 16 seeds are still 0.31 % above the infinite
ensemble and 12 seeds 0.44 %. Single seeds range 4.511–4.721.
- P35 **holds**: the smallest k within 0.5 % is 7.
- P36 **half holds**: the single-seed spread is 4.8 % (> 1 %), but the 16-seed ensemble beats
  the average single seed by 4.6 %, above the predicted 1–3 %.
- P37 **holds on the number, not in spirit**: 4 seeds sit 0.98 % above 16, just inside 1 %,
  but the gap is DM-significant (median DM 2.4). The DM drops below 2 from k = 7 on and
  near 1 from k = 8. Caveat: for large k the subsets share most seeds with the 16, so the
  DM column flatters them.

**Decision: 12 seeds** for weekly production refits and for future backtests where compute
allows (8 is the floor). This matches the user's 9–12 and costs 3× today's fits, which is
still minutes per week.

**Fuel-scaling ablation (item 5), queued on the same pod.** The tuned run `182551`
exactly, with `fuel_scale=false` (weekly, 2020-01..2026-09-21, 4 seeds). Scored with
`qnn_compare.py` against `182551`.
- **P38.** Without fuel scaling the network is worse: 2022 median MAE higher by at least
  2 €/MWh, overall pinball worse, and DM on daily pinball over 2020–26 significant (|DM| > 2)
  in favour of scaling. The rolling window and weekly refits alone do not close the 2022
  gap.

**2. Distributional head (DDNN), the parameter route.** Same inputs and body, head
256 → 24 × 4. Per hour it outputs the parameters of a Johnson's SU distribution (location
ξ, scale λ > 0 via softplus, skew γ, tail weight δ > 0 via softplus), trained by negative
log-likelihood in the scaled space (Marcjasz, Narajewski, Weron, Ziel 2023). The 99
percentiles are then read off the fitted distribution, so it is scored on exactly the same
pinball, coverage and DM as the quantile head. Arms: JSU and Normal (the simple baseline),
each with the tuned body, the same weekly backtest and 4 seeds (or k from item 1).
Hypotheses to register before the run: fewer outputs (96 vs 2,376) means less overfitting
and smoother tails; the price is that the shape is fixed per hour, which may miss bimodal
hours (zero-price vs gas-set). Optional third arm: head outputs both, with the quantile
head regularised toward the JSU.

*Registered (2026-09-24, before the run).* `qnn-de --set head="jsu"` and `head="normal"`
(`models.qnn.DistMLP`, NLL in the scaled space, early stopping on validation NLL,
percentiles = exact quantiles of the fitted distribution). Everything else as in the tuned
run `182551` (body, optimiser settings, weekly refit, 1460-day window, fuel scaling,
4 seeds, Vincentized), but first fit 2024-01-01 to save pod time; compared with
`182551` on the same days (2024-01..2026-09-21) and on the 2026 holdout, with
`qnn_compare.py`. Handicap to keep in mind: the body's hyperparameters were tuned for the
quantile head.
- **P39.** JSU beats Normal on mean pinball over 2024-01..2026-09, DM on daily pinball > 2
  (prices are skewed and heavy-tailed even after asinh scaling).
- **P40.** JSU vs the quantile head: overall pinball within ±2 %, no significant DM on the
  2026 holdout; JSU is better in the tails (mean pinball of q01–q05 and q95–q99) and worse
  or equal in the body (q25–q75).
- **P41.** Before recalibration, JSU's 80 % band coverage is closer to 80 % than the quantile
  head's (the quantile head sat at 75.6 %).

*Registered after seeing the JSU result, before computing it (so weaker evidence):*
- **P42.** A 50/50 Vincentized blend of the JSU and quantile heads (average percentile by
  percentile, then recalibrate) beats both single heads on 2026 holdout pinball, after
  recalibration, by at least 1 % against the better one.

*Result.* Runs: JSU `qnn-de-20260924-075415` (`-clip`: percentiles clipped to the SDAC
limits [−500, 4000] after the fact; 17 of 23,904 hours were outside, one JSU q99 reached
113,038 €/MWh through the sinh inverse; the clip is now in `backtest`), Normal
`qnn-de-20260924-080723`, blend `blend-qnn-jsu`. Reference: quantile head `182551`.

| 2024-01..2026-09-21, raw | pinball | median MAE | cov80 | tail pb | body pb |
|---|---|---|---|---|---|
| quantile head | 4.434 | 12.10 | 75.0 % | 1.355 | 5.613 |
| JSU | **4.380** | **11.74** | 80.3 % | 1.344 | 5.522 |
| Normal | 4.465 | 11.99 | 81.7 % | 1.360 | 5.636 |

| 2026 holdout, recalibrated | pinball | DM vs quantile-cal |
|---|---|---|
| quantile head | 4.625 | – |
| JSU | 4.598 | 0.43 |
| blend JSU + quantile | **4.559** | **2.07** |

- P39 **holds**: JSU beats Normal, DM 4.39 over 2024–26.
- P40 **holds on the numbers, not on the mechanism**: JSU is within 2 % (1.2 % better raw,
  DM 2.28 over 2024–26; holdout DM 0.95, not significant). But it wins in the body
  (q25–q75), not in the tails, which are equal.
- P41 **holds**: raw JSU coverage 80.3 % vs 75.0 %.
- P42 **half holds**: the blend beats both heads, but by 0.85 % against JSU-cal (DM 1.32),
  below the 1 % bar. Caveat: the blend averages 8 networks vs 4 per head, and the seed
  curve puts 4→8 seeds at about 0.7 %, so part of the gain is ensemble size.

**Decision (user, 2026-09-24):** the production candidate is the JSU head plus the quantile
head, blended, 12 seeds, recalibrated. Open check: the blend vs a single head at the same
total network count, which settles the caveat above.

**3. 15-minute products (a plan, since only ~1 year of 15-minute prices exists).**
SDAC went to 15-minute MTU for day-ahead on 2025-10-01; the ENTSO-E cache keeps native
resolution, so the history is 2025-10 onward (verify in the cache). Our hourly target is
already the mean of four quarter-hours.
- Data: prices at 15 min from the cache. TSO load and wind/solar day-ahead forecasts are
  published at 15 min for DE (check neighbours). Everything before 2025-10 stays hourly.
- Model: *hourly level + quarter-hour shape.* The hourly network keeps its long history.
  A second, small model predicts the four quarter-hour deviations from the hourly value
  (they sum to zero), driven by within-hour ramps of solar and load and the hour of day.
  It trains on the ~1 year that exists; shapes are mostly physical (ramps), so a year is
  plausible. Alternative once there is more data: one network with a 96 × 99 head,
  pretrained on hourly history (targets repeated ×4) and fine-tuned on 15-minute data.
- Percentiles: add the shape to each hourly percentile (a comonotone assumption), then
  recalibrate at 15-minute resolution. Score: 15-minute pinball vs the baseline
  "hourly forecast repeated ×4"; hourly aggregates must not get worse.
- Leakage check: the gate logic is unchanged (the same auction), but the UTC 22–23 exclusion
  becomes 8 quarter-hours.

**4. Weather: ECMWF ENS → DWD ICON.** Production's own RES forecast uses ECMWF ENS 00Z
(open data). It is slow to publish and to download, and at 0.25° it is coarser than ICON-EU (~7 km) / ICON-D2
(~2 km, German domain). DWD open data keeps only about 24 h, so there is **no history
except what we archive**. An ICON-EU-EPS archiver exists (`pep archive-weather`, daily cron
in the public repo), but locally there is one file (2026-07-30). First step: check the
data repo for how many days are archived. Plan:
- Start archiving ICON-D2 (and ICON-EU deterministic) now, alongside ICON-EU-EPS.
- Train res-de on ICON features where the archive allows, otherwise use a bridge. Map
  ICON features onto the ECMWF-trained model's inputs (same aggregates: capacity-weighted
  wind at hub height, GHI over the solar regions), fit a linear bridge on the overlap
  days, and switch only when the bridged RES forecast beats ECMWF on the overlap
  (MAE vs TSO actuals, hours 00–06 separately: the night-wind problem).
- Win condition: an earlier forecast slot (ICON 00Z is out about 2 h before ENS), with
  the RES MAE no worse. Re-score the price model with the new RES.

**5. Owed from 2026-09-23.** Ablation of fuel scaling (above). Daily 10:00 UTC outage
snapshots for a clean outage re-test in a year. The production path for the network
(weekly retrain job, publishing fans on the website, where the page's fan chart and the
"chance above X €" row are the natural UI). Site `history.json` backfill days 07-25..08-17
(leaky). The `promo/peaks-*` re-derivation. The both-hinges anomaly.

## Status addendum (2026-09-23, afternoon): scarcity inputs for the tree correction — registered

**Why.** The best honest model (GBM on low-hinge gate-safe LEAR, `lear-gbm-de-…094701`)
still misses peaks: on 2020-01..2026-09-21 hours with an actual price above 200 EUR/MWh
(6,123 h) it has MAE 41.7 and bias −13.9 (LEAR under it: 44.1 / −28.8). Its inputs describe
German demand net of wind and solar only. They say nothing about supply: how much capacity is
out, and whether the neighbours can export. A tree cannot learn a scarcity kink from a
variable it never sees. Correction to an earlier claim: TTF and EUA were already features of
`lear-gbm-de`; what is new in step A is using them as a price *level*.

**Arms** (all `lear-gbm-de`, base `lear-de-…093701`, first fit 2020-01, monthly refits;
scored on 2020-01-01..2026-09-21 with `_scratch/phase2_nonlinear/peaks.py`; reference G0 =
`094701` rerun on the rebuilt dataset):
- **G1 (step A, fuel level):** `scale_target=true`. The tree learns LEAR's error in units
  of a gas-plant marginal cost, max(20, 2·TTF + 0.37·EUA) with 2-day-lagged settlements
  (EUA proxy counts 0 before 2021-10). Same features.
- **G2 (step B, neighbours):** G0 or G1, whichever has the lower MAE, plus
  `features=["neighbours"]`. Features: day-ahead residual load (TSO load forecast minus
  wind and solar forecasts) of FR, and summed over FR, NL, BE, AT, PL, CZ, CH, DK1, DK2;
  the regional total including DE; its daily max. Same convention as DE's inputs. Caveat:
  wind/solar day-ahead forecasts are due by 18:00 D−1 under EU 543/2013, so their
  availability before the gate has to be checked live before production use.
- **G3 (step C, outages and margin):** G2 plus `features=[..., "outages"]`. Unavailable
  capacity by fuel type in DE and FR (ENTSO-E 15.1/15.2, planned plus forced), each outage
  message in the version that was current at D−1 10:00 UTC. The version rule must leave
  out later edits and withdrawals, which are the backtest leak here. Plus a margin feature:
  installed dispatchable capacity minus unavailable capacity minus DE residual load. The
  implementation of the version rule is recorded before the run.
- **D3 (fully non-linear):** `direct=true` with G3's feature set and target handling. The
  tree forecasts the price itself from calendar, forecasts, fuels, and price lags (D−1 from
  UTC hours 0–21 only, and D−7). No LEAR involved.

**Predictions** (G0 today: MAE 14.675, rMAE 0.387, h16–18 18.93, >200 MAE 41.7 / bias
−13.9, Sept-2026 h16–18 34.1):
(19) G1 beats G0 by ≥ 0.05 MAE (DM p < 0.05). Its >200 bias shrinks to ≥ −11: a fixed
error in € is wrong when gas costs move 10×.
(20) G2 beats its parent by DM p < 0.05, with h16–18 MAE down ≥ 0.3 and Sept-2026 h16–18
down ≥ 2. Scarcity evenings are regional.
(21) G3 beats G2 by DM p < 0.05, with >200 MAE down ≥ 2 vs G2. Outages are the supply side
that no input has covered so far.
(22) D3 loses to G3 overall by ≥ 0.3 MAE and has a worse >200 bias: trees cannot
extrapolate above their training range, and LEAR's linear base is what lets the stack do so.
D3 still beats LEAR `093701` alone.

**Results so far (same day).** G0 rerun on the rebuilt dataset (`151234`) reproduces `094701`
exactly.

| arm | run | MAE | rMAE | h16–18 | >200 MAE / bias | Sept h16–18 | DM vs G0 |
|---|---|---|---|---|---|---|---|
| G0 | `151234` | 14.675 | 0.387 | 18.93 | 41.7 / −13.9 | 34.1 | — |
| G1 fuel level | `145542` | 14.803 | 0.390 | 19.22 | 42.4 / **−1.0** | 33.1 | −0.8 (p 0.79) |
| G2 neighbours | `151236` | **14.175** | **0.373** | 18.58 | 41.0 / −15.2 | 34.5 | **7.3** |

(19) **missed** on MAE: fuel scaling costs 0.13. Its bias half came true: the >200 bias
nearly vanishes. (20) **half met:** DM 7.3 and h16–18 −0.36 hold, Sept-2026 h16–18 does not
(+0.5). G2 therefore builds on G0.

**Registered after these results, before the run:** arm G2s = G2 + `scale_target=true`.
Prediction (23): G2s is within 0.1 MAE of G2 and keeps most of G1's bias fix
(>200 bias ≥ −5). If so, G3 builds on G2s: a model that stops under-forecasting peaks is
worth a tie on MAE ahead of the quantile step.

Result: G2s (`151653`) scored MAE 14.371, +0.20 over G2 (DM −1.4), with >200 bias −3.3 and
Sept h16–18 32.6. **(23) missed** on the MAE margin; G3 builds on G2. Reading: bias on
hours *selected by a high outcome* is not a defect of a median forecast. Any forecast that
minimises absolute error regresses towards the typical price, so it under-shoots the hours
that turned out extreme. Fuel scaling buys back that bias by forecasting higher peaks on
average, and pays for it on every other hour. What the peaks need is a forecast
distribution, not a shifted median. That is the quantile step's job, and G1/G2s show that
the fuel level is the right unit for its upper quantiles.

**Outage version rule (recorded before any G3 run).** Finding: ENTSO-E re-stamped every
outage message dated before the 2025-10-05..07 platform migration. All revisions of a
January-2024 message carry `createdDateTime` values from those three days (checked on the
probe sample and on four multi-revision messages queried per mRID). Revision contents
survive, their publication times do not. So a true as-of reconstruction exists only from
2025-10 on. Rules, with cutoff c = 10:00 UTC on D−1:
- *Exact* (2025-10 onward, validation only): per message, the latest revision with
  createdDateTime ≤ c; withdrawn messages count only for revisions before the withdrawal.
- *Approximate* (the whole history, used by G3): latest version only.
  - **Planned** messages: counted from the latest schedule. Leak: postponements and
    cancellations published after c.
  - **Forced** messages: counted only if the outage started before c − 1 h (REMIT's
    one-hour publication rule makes it public by c). The unavailable MW at c is held flat
    over all of D, a persistence forecast, so later revisions to its end date never enter.
  - Only generation units (A80) of zones DE_LU and FR; wind and solar outages dropped.
- Features: unavailable MW for DE thermal (nuclear, lignite, hard coal, gas, oil), DE
  total, FR nuclear, FR total. Plus DE margin = installed dispatchable capacity (ENTSO-E
  installed capacity per type, yearly: fossil, nuclear, hydro storage, biomass) − DE
  unavailable dispatchable − DE residual load forecast.
- Validation before trusting G3: on 2025-10..2026-09, the approximate vs the exact
  features hour by hour. Required: correlation ≥ 0.9 and mean |diff| ≤ 10 % of the mean.
  Otherwise the G3 result is reported as optimistic.

**Registered (2026-09-23, before any full run): quantile network `qnn-de`.** The user's
call: the pinball step goes straight to a neural network with 99 percentiles, run after G3,
on the pod, with fixed sensible settings first and an Optuna search only if it performs
reasonably. Design (commit `c3c5313`):
- **Inputs:** one sample per UTC day. LEAR's gate-safe blocks: price lags d−1 (hours 0–21),
  d−2, d−3, d−7. Load, RES and neighbour-sum residual load at d, d−1 (0–21), d−7. TTF and
  EUA (2-day lag). Weekday dummies. 313 inputs, median/MAD + asinh scaled on the training
  window. G3's outage features are added as a second arm if they pass validation.
- **Network:** MLP 256-256, ELU, dropout 0.1. Output 24 hours × 99 percentiles, built from
  a median plus softplus steps outwards, so they never cross. Loss: mean pinball in the
  scaled space. Monotone scaling keeps percentiles valid after the inverse transform.
- **Training:** AdamW, lr 1e-3, weight decay 1e-4, batch 32, ≤ 400 epochs, early stop
  (patience 30) on a random 15 % of training days. Expanding window from 2018-12, monthly
  refits from 2020-01. Each refit trains on days ≤ M−2 only: UTC 22–23 of M−1 belong to
  M's auction. 4 seeds, averaged percentile by percentile.

A smoke test (Jan–Feb 2024, 2 seeds, before a fix to the head's start-up spread) gave
median MAE 6.84 vs G2 9.23 and LEAR 10.72 on the same hours. Too short to trust, and
suspicious enough that the leak tests were written before any further run.

Predictions for 2020-01..2026-09-21:
(24) median MAE < G2's 14.175, DM p < 0.05.
(25) calibration: 80 % central interval covers 75–85 %; the 1st and 99th percentiles are
each exceeded 0.5–3 % of the time (smoke: 11 % below the 1st, so this can miss at the low
end, where the negative hours are).
(26) on hours > 200 EUR/MWh the 90th percentile is exceeded ≤ 35 % of the time, and the
median's bias there is ≥ −10 (G2: −15.2).
Decision: if (24) holds or the median ties G2 within 0.3 MAE with (25) met, run Optuna
(pod, validation year 2025 trained through 2024, 30–60 trials). If not, stop there.

**Result (same day, pod, 20 min, run `qnn-de-20260923-155904`).** Median MAE **16.18**
(rMAE 0.426) vs G2 14.175 and LEAR 16.32; DM vs G2 −8.1. Mean pinball 5.84 (CRPS ≈ 11.7).
**(24) missed**, and not within 0.3, so by the rule above no Optuna yet. **(25) missed:** the
80 % interval covers 69 %, the 98 % interval 93.6 %; the median is exceeded 56.7 % of the
time (biased low). **(26) half:** the 90th percentile is exceeded 33 % of the time on
>200 hours (met), but the median's bias there is −52 (missed). By year (MAE):

| year | QNN | G2 | LEAR | QNN bias |
|---|---|---|---|---|
| 2020 | 4.28 | 4.26 | 4.70 | +0.7 |
| 2021 | 16.42 | 12.43 | 14.19 | −10.2 |
| 2022 | 40.31 | 30.97 | 34.56 | −26.3 |
| 2023 | **13.97** | 15.45 | 18.60 | +1.3 |
| 2024 | 12.09 | 11.43 | 13.50 | −0.8 |
| 2025 | 12.09 | 10.87 | 12.81 | −2.6 |
| 2026 | 13.54 | 13.73 | 15.79 | −3.3 |

Sept 2026: QNN 19.2 (h16–18 30.9) vs G2 22.0 (34.5). Reading: the loss is almost all
2021–22, where the median trails the price rally by 10–26 EUR/MWh. Outside the crisis the
network alone is level with LEAR + trees + neighbours. The failure is structural, not
hyperparameters. Targets are scaled by the training window's median and MAD, which the
expanding window anchors to the cheap 2019–20 levels, and refits come only once a month.
So in 2022 the prices sit far out in the asinh tail, where the output saturates. The
smoke test's 6.84 on Jan–Feb 2024 was two calm months, not a leak.

**Registered (2026-09-23, after that result, before the run): QNN level fix.** One arm, the
same network and settings, three structural changes: `window_days=730` (rolling two years,
so the scaler follows the current level), `refit="week"` (training still ends two days
before each week), and `fuel_scale=true` (target and price lags divided by that day's
max(20, 2·TTF + 0.37·EUA), a positive per-day factor, so percentiles scale back exactly).
Predictions: (27) 2022 MAE drops below 34 (from 40.3) and the 2021–22 bias halves;
(28) overall median MAE ≤ 14.5, i.e. within 0.3 of G2 (14.175); (29) 80 % coverage moves
to ≥ 74 %. If (28) holds, Optuna runs on this configuration.

**Result (pod, ~12 min, run `qnn-de-20260923-170301`).** Median MAE **13.36** (rMAE 0.352)
vs G2 14.175: **DM 4.2, p < 0.001, the best point forecast so far**, and from a network on
its own. Mean pinball 5.84 → 4.94. By year vs G2: 2020 4.23/4.26, 2021 11.42/12.43, 2022
27.48/30.97, 2023 12.78/15.45, 2024 12.19/11.43, 2025 12.12/10.87, 2026 13.43/13.73. Bias
2021/22: +2.9/−2.0 (was −10.2/−26.3). Peaks: >200 MAE 32.4 (G2 41.0), bias −13.8; h16–18
16.6 (G2 18.6); Sept 2026 18.1, h16–18 **25.1** (G2 34.5). (27) **met** (27.5, bias
gone). (28) **met, and then some.** (29) **missed:** 80 % coverage 68.5 %, 98 % 92.4 %.
The bands are still too narrow, although the median is now unbiased (exceeded 50.3 %).
On >200 hours the 90th percentile is exceeded 22 % of the time.

**Registered (before the search): Optuna on the level-fix network (`qnn-tune`).** Objective:
mean pinball over the 99 percentiles in EUR/MWh on **2025** (validation), refits every 4
weeks, 2 seeds, fuel scaling on. Search space: 1–3 layers, width 64–512, dropout 0–0.4,
lr 1e-4–3e-3, weight decay 1e-6–1e-2, batch 16–128, rolling window 1–4 years. TPE sampler
(seed 0), about 60 trials, the level-fix configuration enqueued as trial 0. The winner
then reruns the full weekly backtest with 4 seeds. **Holdout: 2026-01-01..09-21**, never
seen by the search; the full span 2020–26 includes the tuning year and is reported but
marked as such. Predictions: (30) the best trial beats trial 0 on 2025 pinball by ≥ 3 %;
(31) on the 2026 holdout the tuned network beats the level fix on pinball, DM p < 0.05 on
per-day CRPS-approximating pinball; (32) the 80 % coverage stays below 75 %. Pinball
tuning alone will not fix calibration; that needs a separate recalibration step (e.g.
conformal widening), registered separately if (32) holds.

**Results (same evening).** Search `qnn-tune-20260923-172006`, 60 trials, ~45 min on the
pod. Best: 3 × 256, dropout 0.28, lr 5.0e-4, weight decay 8.5e-5, batch 32, window 1,460
days. 2025 pinball 4.391 vs trial 0's 4.744 (−7.4 %). The top trials all used the
four-year window. The full weekly backtest with 4 seeds is `qnn-de-20260923-182551`
(~35 min):

| span | metric | level fix `170301` | tuned `182551` |
|---|---|---|---|
| **holdout 2026-01..09-21** | pinball / median MAE | 5.021 / 13.33 | **4.723 / 12.66** |
| | 80 % / 98 % coverage | 70.0 / 93.5 % | 77.2 / 96.3 % |
| | DM on daily pinball | — | **3.56, p = 0.0002** |
| Sept 2026 (1–21) | pinball / MAE / h16–18 | 6.53 / 18.08 / 25.1 | **5.71 / 16.64 / 25.8** |
| full 2020–26 (contains the tuning year) | pinball / MAE | 4.935 / 13.36 | **4.784 / 13.16** (DM 4.9) |
| | 80 % / 98 % coverage | 68.5 / 92.4 % | 75.6 / 96.2 % |

Against G2, full span: MAE 13.16 vs 14.18 (DM 5.05), >200 MAE 32.7 vs 41.0 with bias
−11.4 vs −15.2, h16–18 16.95 vs 18.58. 2026 alone: 12.75 vs 13.73. By year (tuned / G2):
2020 4.32/4.26, 2021 11.20/12.43, 2022 27.76/30.97, 2023 12.26/15.45, 2024 11.79/11.43,
2025 12.02/10.87. **(30) met** (−7.4 %). **(31) met** (holdout DM 3.56). **(32) missed, in
the good direction:** tuning widened the bands (dropout 0.28, four-year window), and 80 %
coverage rose to 75.6 % overall and 77.2 % on the holdout. Still short of 80 %: the 1st
percentile is undercut 1.5 % of the time, the 99th exceeded 2.2 %, the 90th 12.7 %; on
>200 hours the 90th is exceeded 18 %. **Best model of the project: the tuned quantile
network.** Next: a small calibration step (conformal widening per hour, fitted on a
trailing window) for the last ~3–5 pp of coverage, then the fan-chart page.

**Registered (2026-09-24, before the run): rolling PIT recalibration** (`models/recalibrate.py`)
of `182551`. Weekly: fit on the PIT values of the forecast hours of days up to S−2, trailing
365 days, all hours pooled. Serve level τ from the model's level G⁻¹(τ), with exponential
tails beyond the 1st/99th percentiles. First applied 2021-01-01 (2020 is the first
window). Predictions on the 2026 holdout: (33) 80 % coverage lands in 78–82 % and 98 %
coverage in 97–99 %; (34) mean pinball is not worse than 4.72 + 0.02, since recalibration
should cost pinball only if the past year misleads.

**Result** (`runs/qnn-de-20260923-182551-cal`, `_scratch/phase2_nonlinear/calibrate_run.py`):

| span | pinball raw → cal | 80 % cov | 98 % cov | worst level error |
|---|---|---|---|---|
| 2021–2026-09-21 | 5.349 → 5.352 | 76.4 → **80.0 %** | 96.6 → **97.8 %** | 4.6 → 1.2 pp |
| holdout 2026 | 4.723 → **4.625** | 77.2 → 84.3 % | 96.3 → **98.6 %** | 11.2 → 2.9 pp |
| Sept 2026 | 5.713 → 5.837 | 80.6 → 85.7 % | 98.6 → 100 % | 7.9 → 8.5 pp |

**(33) half:** 98 % coverage in range, 80 % overshoots to 84.3 %. 2026 was calmer than its
trailing year, so the widening learned in 2025 is a bit too much. **(34) met:** pinball
improves by 0.10. Over 2021–26 the percentiles are now within ~1 pp of nominal. The
recalibrated run is the reference for the fan-chart page.

**Outage validation (2026-09-23, before G3): the registered approximation fails.** Exact
pre-gate unavailability was reconstructed from real revision timestamps (617 of 625
multi-revision DE messages complete, nothing pending) for Nov–Dec 2025 (DE) and Nov
2025–Jan 2026 (FR), then compared hour by hour with rule A (the registered one) and rule B
(only outages running at the cutoff, planned or forced, held flat):

| feature | exact mean | rule A: bias / corr / mean abs diff | rule B |
|---|---|---|---|
| DE thermal | 6,706 MW | +1,964 / 0.48 / 31 % | +1,975 / 0.27 / 36 % |
| DE dispatchable | 8,210 | +1,986 / 0.36 / 25 % | +2,119 / 0.13 / 31 % |
| FR nuclear | 9,735 | +1,539 / 0.65 / 17 % | +1,555 / 0.65 / 18 % |
| FR total | 12,172 | +1,889 / 0.65 / 17 % | +2,100 / 0.63 / 19 % |

Required was correlation ≥ 0.9 and ≤ 10 %; both rules miss by far. Cause: German REMIT
publication is late. Among "planned maintenance" messages starting in Nov–Dec 2025, the
median went out 13 h before the start, a quarter after it had begun, and 10 % more than
61 h late. Forced messages: median 0.2 h after the start, 10 % more than two days late. A
message's final version therefore contains outages nobody could see at the gate, whatever
rule filters it. **Pre-2025-10 pre-gate unavailability is not recoverable from ENTSO-E.**
Per the rule above, G3 runs with rule A and its result is an optimistic upper bound: no
gain there means outages are out; a gain is untrustworthy until it is re-measured on
exact data. Clean route: daily 10:00 UTC outage snapshots from now on, plus the exact
reconstruction from 2025-10, then re-test once there is a year or more of it.

**G3 and D3 results (same evening, laptop).**

| arm | run | MAE | h16–18 | >200 MAE / bias | Sept 26 / h16–18 | DM vs G2 |
|---|---|---|---|---|---|---|
| G2 | `151236` | 14.175 | 18.58 | 41.0 / −15.2 | 22.0 / 34.5 | — |
| G3 = G2 + outages (rule A, upper bound) | `175737` | 14.231 | 18.48 | 41.7 / −11.9 | 20.7 / 32.6 | −0.9 (p 0.82) |
| D3 direct trees, G2 features | `180452` | 17.367 | — | — | — | — |

**(21) missed:** even the leaky outage features add nothing overall. They nudge the peak bias
and September, but not beyond noise. Outages stay out of the tree correction. The only
route left is exact snapshots, re-tested once they cover a year. **(22)** D3 loses to the
stack by 3.1 (met), but also to LEAR alone (16.32, missed). Its by-year profile mirrors
the first QNN: fine in calm years (2023 13.2 vs LEAR 18.6), lost in the rally (2022 43.7).
D3 ran on G2's features rather than G3's, since G3's outage columns are not honest. Both
fully non-linear failures in 2021–22 are the level problem that fuel scaling and a
rolling window fixed for the network.

**Decision rule.** The best G arm becomes the point reference for the pinball (quantile)
step. If (21) misses, outages stay out of the production path; the as-of pipeline is the
expensive part.

## Status addendum (2026-09-23): Phase 2 start — bench verified, hinge experiment registered

**Data refreshed** through 2026-09-23 via `pep fetch-{entsoe,smard,energy-charts,fuels,capacity}`
+ `pep build-dataset` (102,705 rows; forecast columns gap-free after SMARD patching; API2 coal
ticker dead since 2025-12-30 — unused by any model, but it freezes `complete_rows`).

**September bench = plain `pep run lear-de`** (`window=371`, academic, 2026-09-01..21): MAE
23.284 (h16–18 39.85, h00–06 11.97). The model inputs are identical to the `postgate_rescore.py` frame except
one hour (2025-10-25 22:00 UTC RES: SMARD-patched 38,684 vs interpolated 40,597 MW); with
that hour interpolated the dataset run reproduces the post-gate forecast **bit-exactly**
(23.187). So no scratch frame needs promoting. **Noise floor, measured:** that single
1.9 GW input hour, 11 months before the test window, moves September MAE by 0.1 and single
hours by up to 25 EUR/MWh (LASSO selection flips). September deltas of a few tenths are not
evidence; verdicts come from DM on the long tier.

**On residual load:** no run in any repo ever trained a model on it. Linear residual load is
redundant in LEAR by construction (L − R lies in the span of L and R already present), so a
non-result would have been expected. A hinge is not in that span.

### Registered experiment (2026-09-23): hinge-LEAR

**Design.** `pep run lear-de --set exog=academic --set hinge=[0.1,0.9]`: plain academic LEAR
plus same-day (lag 0) terms max(0, k_lo − RL) and max(0, RL − k_hi), RL = load − RES
forecast; knots = q10/q90 of RL over each calibration window (re-estimated daily, never
the target day); terms divided by the window's RL MAD and exempt from the asinh scaling
(`models/lear.py::hinge_features`). 247 → 295 weights (n > p at 357 rows). Arms:
long tier window 364, 2019-01-01..2026-09-21 vs plain academic on the same fresh data;
September tier window 371. Informational (inputs only, no fit): over 2025-08-26..2026-08-31
q10 = 5.9 GW, q90 = 47.3 GW, and 88 % of negative-price hours have RL below q10.

**Pre-registered predictions (before any hinge fit):**
1. Long tier, negative hours: sign recall rises by ≥ 10 pp over the baseline, and the depth
   ratio (median forecast / median actual on jointly negative hours) at least halves.
2. Long tier, hours with actual ≥ 0: MAE no worse than baseline + 0.10 EUR/MWh.
3. Long tier overall: rMAE lower by ≥ 0.005, DM (hinge better) p < 0.05.
4. September: hours 16–18 improve by **less than** 3 EUR/MWh on 39.85. The high hinge does
   not fix scarcity evenings, because scarcity is not a function of RL alone (imports,
   outages, the slope moves within the year). Overall September change within ±1.0.

**Registered add-on (2026-09-23, after the September hinge run, before any long-tier
result).** September already showed hours 16–18 at 39.85 → 33.21 (prediction 4 refuted)
and negatives *deeper* (median −26.3 vs −18.8, 34 h). Single-hinge ablations on the long
tier separate the two terms: high-only `hinge=[0.0,0.9]`, low-only `hinge=[0.1,1.0]` (a knot
at quantile 0/1 zeroes that term on the training window). Predictions: (5) high-only
carries ≥ 70 % of the full hinge's MAE gain at hours 16–18; (6) low-only moves sign recall
more than high-only does, and (in light of September) its depth ratio does **not** halve.

**Registered step 3 (2026-09-23, before any fit): gradient-boosted correction of LEAR.**
Target: out-of-sample error of the long-tier LEAR run (actual − forecast). Model: sklearn
HistGradientBoosting, absolute-error loss, expanding window, monthly refits, first
prediction month 2020-01. Features: hour, weekday, day-of-year harmonics, load, RES and
residual load forecasts plus the day's RL max/min, TTF and EUA (2-day lag), LEAR's own
forecast, and LEAR's error at the same hour on D−1 and D−7 (legal: D−1 prices come out of
the D−2 auction). Predictions: (7) on 2020-01..2026-09 it beats its own LEAR base by
≥ 0.02 rMAE, DM p < 0.01; (8) on September 2026 it cuts hours 16–18 by more than the
hinge did; (9) negative-hour depth ratio at least halves — trees can learn the floor that
a hinge could not.

**Registered (2026-09-23, before the run): backtest night-hour leak.** UTC 22–23 of D−1
are local 00:00–01:00 of delivery day D, cleared in the auction being forecast; plain
`lear-de` feeds them to LEAR as lag-1 prices (and as the D−1 training target), production
cannot (`daily_forecast.lear_forecast` heals them from 24h-lag). Evidence so far: the
long-tier plain run has MAE 3.7 at hour 0 UTC, rising monotonically to 10.4 at hour 4.
`--set gate_safe_prices=true` mirrors production's healing for prices only. Predictions:
(10) hour-0 MAE at least doubles and hours 0–1 together lose ≥ 3 EUR/MWh; (11) overall
rMAE worsens by ≥ 0.005 — every lear-de backtest number so far is flattered by that much.

**Registered (2026-09-23, after long-tier hinge and GBM-on-plain results, before this run):
GBM on hinge-LEAR.** Trees extrapolate flat, a hinge extrapolates linearly — on September
evenings GBM-on-plain lost (38.8 vs 36.3) where the hinge won (32.6). Prediction (12):
`lear-gbm-de` with the long hinge run as base keeps GBM-on-plain's overall gain
(MAE ≤ 13.80 on 2020-01..2026-09-21) and lands September hours 16–18 at ≤ 34.

**Registered (2026-09-23, after GBM-on-hinge, before any run): window ensemble.** Result
that motivates it: GBM-on-hinge ties GBM-on-plain overall but September hours 16–18 fall
back to 36.6 — whatever is trained on history cannot see a new scarcity regime, so the
remaining lever is adaptation speed. Arms: plain academic LEAR at windows 56 and 84
(2019-01-01..2026-09-21), averaged hour by hour with the 364 run (epftoolbox-style
ensemble, computed offline). Predictions: (13) the {56, 84, 364} mean beats 364 alone by
≥ 0.01 rMAE overall; (14) it cuts September hours 16–18 by ≥ 3 EUR/MWh vs 364's 36.3.

**Registered (2026-09-23, after the high-only ablation beat both hinges, before this run):**
(15) `lear-gbm-de` on the high-only base beats GBM-on-plain by ≥ 0.05 EUR/MWh MAE on the
common span with DM p < 0.05, keeping depth ratio ≤ 1.6.

**Registered (2026-09-23, after GBM-on-high-only, before this run):** (16) GBM on the
low-only base does **not** beat GBM on high-only (MAE ≥ 13.45): trees already learn the
floor the low hinge gives LEAR, while the high hinge supplies the one thing trees lack —
linear extrapolation above the knot.

**Registered (2026-09-23, before the long run): consistent gate-safe LEAR.** The first
`gate_safe_prices` run healed only the forecast day (production's scheme) and scored rMAE
0.532 vs 0.409 (hour 0: 3.7 → 19.3) — a train/test mismatch: every training row still
carried the target-auction hours, so the model leaned on them and then met stale values.
Redefined (commit after ee88ca8): lag-1 hours 22–23 come from d−2 on EVERY row. September
check (window 371, official inputs): MAE 29.40 vs leaky 23.28 vs **live 29.74**; hours
0–3 at 20–22 vs leaky 5–8 vs live 24–30. Predictions for the long tier: (17) consistent
gate-safe plain lands at rMAE 0.46–0.51 — worse than every published backtest number,
better than the production-style mismatch (0.532); (18) the high-only hinge's gain
survives it (≥ 0.005 rMAE, DM p < 0.05).

**Decision rule.** 1–3 met → hinge-LEAR becomes the linear baseline that step 3 (gradient-
boosted correction of LEAR's out-of-sample error) must beat. 1 met, 3 not → kept as an
ablation. 1 missed → the zero regime needs more than a hinge; recorded as such.

### Results (2026-09-23, same day)

Long tier = 2019-01-01..2026-09-21 on a 32-core RunPod (Python 3.13; laptop runs Python
3.11 — all long-tier arms share the pod, so comparisons are like for like). Step-3 runs on
the laptop (2020-01..). Scored with `pep score-run` / `_scratch/phase2_nonlinear/table.py`.
Reference: plain academic LEAR `lear-de-20260923-081457`, rMAE 0.4085, sign recall 59.6 %,
depth ratio 3.76.

| # | prediction | verdict | numbers |
|---|---|---|---|
| 1 | hinge: recall +10 pp, depth halves | **missed** | recall +7.1 pp; depth 3.76 → **4.35** (deeper) |
| 2 | hinge: no MAE loss on hours ≥ 0 | met | 13.74 → 13.52 |
| 3 | hinge: rMAE −0.005, DM p < 0.05 | **missed narrowly** | −0.0042; DM 2.3, p = 0.010 |
| 4 | Sept 16–18 gain < 3, overall ±1.0 | **refuted** | −6.6 (39.85 → 33.21); overall −1.08 |
| 5 | high-only carries ≥ 70 % of evening gain | met | 116 % (h16–18 18.89 → 18.41); rMAE 0.3976, DM 8.4 |
| 6 | low-only moves recall more; depth not halved | met | recall 65.5 vs 60.0 %; depth 4.19; rMAE 0.3942, DM 9.1 |
| 7 | GBM on plain: −0.02 rMAE, DM p < 0.01 | met | 0.408 → **0.362** (2020–); DM 13.0; wins 66 % of days |
| 8 | GBM beats hinge on Sept 16–18 | **refuted** | GBM 38.8 vs plain 36.3 vs hinge 32.6 |
| 9 | GBM halves the depth ratio | met | 3.96 → **1.42**; recall 60.4 → 68.9 % |
| 10–11 | gate-safe (production scheme): h0 doubles, rMAE +0.005 | met, massively | h0 3.7 → 19.3; rMAE 0.409 → 0.532 (a train/test mismatch — see below) |
| 12 | GBM on hinge: keeps gain, Sept 16–18 ≤ 34 | half | 13.75 (met); Sept 36.6 (missed); ties GBM-on-plain, DM p = 0.61 |
| 13 | window ensemble {56,84,364}: −0.01 rMAE | **missed** | −0.005, DM p = 0.13 |
| 14 | ensemble cuts Sept 16–18 by 3 | **refuted** | 39.9 (w56 47.4, w84 46.9) vs 36.3 |
| 15 | GBM on high-only beats GBM on plain | met | 13.45 vs 13.74, DM 5.0; depth 1.39 |
| 16 | GBM on low-only does not beat GBM on high-only | **missed (literal)** | 13.433 vs 13.452 — a tie, DM p = 0.36 |
| 17 | consistent gate-safe plain at rMAE 0.46–0.51 | **missed (better)** | **0.4432** |
| 18 | high-only gain survives gate-safe | met | 0.4432 → 0.4330, DM 8.9 |

**Reading.** (a) A hinge is not what the zero regime needs (decision rule, prediction 1):
the low term *deepens* negatives. Trees learn the floor (depth 1.4) where no linear term
did. (b) Each hinge alone beats plain LEAR clearly, yet both together are worse than
either — the both-hinge run logged many unconverged Lasso fits (duality gaps 1–3);
suspected optimisation artefact, **open**. (c) September's scarcity evenings defeat
everything trained on history: trees extrapolate flat, short windows lack data; only the
linear high hinge helped (and the trees on top wash it out again). (d) Best model today:
GBM on a single-hinge LEAR base, rMAE 0.354 vs 0.408 — under the leaky convention below.

**Finding: every `lear-de` backtest so far leaks two target-auction prices.** LEAR's UTC
day D−1 hours 22–23 are local 00:00–01:00 of delivery day D, i.e. cleared in the auction
being forecast; plain `lear-de` feeds them in as lag-1 prices (hour-0 MAE 3.7, rising
monotonically with distance from midnight). Production cannot see them and heals them
from 24h-lag *on the forecast day only* — while training on history that has them: a
train/test mismatch. `gate_safe_prices=true` (commit 1a63aae) drops lag-1 hours 22–23 on
every row. Long tier:

| scheme | rMAE | h0 MAE | Sept MAE (1–21) |
|---|---|---|---|
| leaky backtest (all published numbers) | 0.4085 | 3.7 | 23.04 |
| production scheme (heal forecast day only) | 0.5321 | 19.3 | 29.09 |
| **consistent gate-safe** | **0.4432** | 9.6 | 27.31 |

Consequences: (1) published backtest numbers are flattered by ≈ 0.035 rMAE; (2) the
production scheme costs ≈ 0.09 rMAE (≈ 3 EUR/MWh) against the consistent one — the
September production-scheme backtest (29.09) sits next to the real live MAE (29.74);
(3) **the 2026-09-22 surrogate cost is mostly this leak**: the post-gate rescore used the
leaky scheme. September, window 371, official inputs: leaky 23.28 → consistent gate-safe
27.37 → live 29.74, so the real live-input cost is ≈ 2.4, not 6.5, and the "night-wind"
attribution (hours 00–06) is largely the leak (gate-safe night hours 20–22 vs leaky 5–8
vs live 24–30). Step 3's own D−1 error features exclude UTC 22–23 (fixed before scoring;
the fix moved GBM MAE by 0.02).

**Everything re-run gate-safe (the honest numbers).** Long-tier LEAR arms 2019-01..,
GBM arms 2020-01.. (DM vs the first row of each block):

| arm | rMAE | Sept MAE | Sept h16–18 | depth | DM |
|---|---|---|---|---|---|
| LEAR gate-safe `091935` | 0.4432 | 27.31 | 38.2 | 3.83 | — |
| + high hinge `092728` | 0.4330 | 26.97 | 36.9 | 3.52 | 8.9 |
| + low hinge `093701` | 0.4296 | 26.77 | 37.8 | 4.38 | 7.8 |
| GBM on LEAR gate-safe `lear-gbm-de-…092912` (2020–) | 0.3914 | 23.42 | 39.6 | 1.59 | 12.5 |
| GBM on high-hinge gate-safe `…093730` | 0.3867 | 21.48 | 35.5 | 1.58 | 13.5 |
| GBM on low-hinge gate-safe `…094701` | 0.3866 | 21.16 | 34.1 | 1.55 | 13.3 |

(The GBM rows' DM is against LEAR gate-safe on 2020–, where it scores 0.443.) GBM on a
single-hinge base beats GBM on plain LEAR (DM 3.1, p < 0.001); high vs low base is a tie
(DM p = 0.48). **Best honest model: GBM correction on single-hinge gate-safe LEAR, rMAE
0.387** — better than every leaky LEAR number ever published here (0.407).

### Next steps (proposed 2026-09-23, for the user)

1. **SHIPPED 2026-09-23 (public repo `24bbbbb`, CI green):** production `lear_forecast`
   runs `forecast_day(gate_safe=True)`, which drops lag-1 hours 22–23 of prices AND TSO
   forecasts on every row (the production path also lacks the boundary TSO forecasts, so
   the prices-only variant was not enough — a 21-day check had it losing). Replay of the
   production function on a pod, 2024-10-01..2026-09-21 (721 days, pre-gate data,
   official exog; `_scratch/phase2_nonlinear/pod_prod_dryrun.py`): MAE 16.24 → 14.73,
   hour 0 16.5 → 8.0, DM 10.7, 473/721 days won; filling the boundary with a perfect
   surrogate instead ties (DM −0.74). Repos now diverge: nl's `build_xy(gate_safe)` drops
   price lags only — reconcile on the next sync.
2. **Re-stated 2026-09-23 (website `76501c6`),** pod runs with the shipped definition
   (code d8a18e8), long tier 2019-01-01..2026-09-21 (67,704 h): academic-364 MAE 15.21 /
   rMAE 0.442 (`133950`; was 13.9/0.41), w56 17.56/0.511 (`134504`), w84 17.13/0.498
   (`134542`), extended-364 18.08/0.526 (`134644`); naives 34.37 / 27.88. Input costs,
   2024-10-01..2026-08-25, gate-safe vs leaky rerun (leaky reproduces the published
   +0.14/+0.31/+0.43): 12Z +0.18, load surrogate +0.28, evening +0.45 EUR/MWh (runs
   `140326..141251`). Same-horizon 2016–17 gate-safe: w56 4.597, w84 4.469 vs benchmark
   4.593/4.529 — the "our data 4–6 % better" claim was mostly the leak. Still open:
   `promo/peaks-*` (built on leaky arm-C backtest through 2026-08-15) and the 23
   backfilled post-gate days 07-25..08-17 in `history.json`. The 2026-09-22 surrogate-cost
   claim (≈ 2.4, not 6.5) is corrected in the results block above.
3. **Ship candidate:** GBM correction on single-hinge gate-safe LEAR — needs a production
   path (monthly refit of the tree on the rolling backtest's errors) and a live A/B.
4. **Open anomaly:** both hinges together < either alone; rerun with higher Lasso
   `max_iter` to test the optimisation-artefact reading.
5. **Probabilistic step:** the same tree correction with pinball loss (quantiles) is the
   natural bridge to Phase 2 proper.

## Status addendum (2026-09-22): why live MAE doubled — regime, linearity, and a measured surrogate cost

**Observation:** the published 30-day mean has been climbing since early
September (live pre-gate MAE, first 14 live days 19.3 → last 14 live days
36.3 EUR/MWh). Suspicion was surrogate misalignment or a refit that is not
really happening. Both were checked; neither is the driver.

**1. Skill is flat, the market moved.** The naive same-hour-yesterday
benchmark doubled with us (≈29 → ≈55); MAE/naive has stayed between 0.7 and
0.9 every week since mid-August. Level error and shape error grew in step
(|daily bias| 12.8 → 21.9, bias-removed MAE 15.6 → 28.1). September is a new
regime: TSO evening residual load reached 54–57 GW on 09-14 and 09-22 (August
max ≈46), the evening price-vs-residual-load slope tripled (≈3 → ≈8 EUR/MWh
per GW, hours 16–18 UTC), daily means swung 24 → 108 → 226 EUR/MWh over
09-20..09-22 with a 596 EUR/MWh hour, and 697 on 09-14. Errors sit exactly
there: hours 16–18 UTC went from MAE ≈17 to ≈50 (3×), the rest of the day
only 2×. The 09-22 forecast (MAE 93.8, bias −93.8) was additionally dragged
down by the D-1/D-2 lags from the two cheap weekend days; 09-23 recovered to
≈20 once they rolled through.

**2. The refit is real but is not adaptation.** `run_daily` fits a fresh
LassoLarsIC + Lasso per hour every morning (models/lear.py, no cached
coefficients). But the window is a fixed 364 days with uniform weights: one
new day is 0.3 % of the training data, and a linear model fitted across a
whole year carries an averaged merit-order slope. It structurally cannot emit
a 500 EUR/MWh evening when the current slope is 3× the annual average. This
is the model-class limit Phase 2 exists for.

**3. Surrogate cost measured directly: −6.5 EUR/MWh (≈22 %), constant.**
Post-gate rescoring of 2026-09-01..09-21 with the identical LEAR (371-day
calibration = production's 364 training days) but the *official* ENTSO-E load
and wind/solar forecasts instead of own-RES v2 + load-de:

| period | live MAE | post-gate MAE | delta |
|---|---|---|---|
| 09-01..09-08 | 26.7 | 20.1 | −6.6 |
| 09-09..09-21 | 31.6 | 25.1 | −6.5 |

Official inputs win 18/21 days (losses ≤ +1.9, gains up to −24 on 09-05 and
09-19). The delta is identical in both halves of the month, so the surrogates
are not what is getting worse — but the cost is far above the badge numbers
(+0.4 evening, +0.3 load) and worth fixing. Hourly profile (09-09..09-22):
nearly the whole gain is **hours 00–06 UTC** (post-gate MAE 5–22 vs live
24–40) — own-RES night wind, most plausibly the 00Z-vintage handicap against
the TSO's fresher runs. At **hours 17–18 UTC the official inputs are worse**
(63 vs 27 at hour 17): the evening spikes are the linear model, not the
inputs. Caveat: the TSO cache holds the latest revision, not the 18:00 D-1
first publication, so −6.5 is an upper bound on the real pre-gate gap. 09-22
and 09-23 were not scoreable (TSO wind/solar for local day 09-23 still
unpublished at 16:20 UTC).

**4. Two housekeeping findings.**
- `post_gate: true` days that sit inside the live log (08-18, 08-25, 08-28,
  08-31, 09-09) are bit-identical to the live forecast rows: they used
  own-RES + surrogates and were merely generated late. Only 07-25..08-17 are
  true official-exog reconstructions (reproduced within 0.1 EUR/MWh with
  `calibration_window=364`, i.e. the backfill convention). The site legend
  ("calculated after gate closed") and the 30-day-mean exclusion conflate the
  two meanings.
- Nothing scores the surrogates in production: `forecast_log.parquet` keeps
  only the final price, and the 18:00 TSO snapshot (`pep
  archive-entsoe-forecasts`) is written daily and never read back. The
  hypothesis above was untestable from the pipeline's own artifacts.

Scripts and outputs (local only, `_scratch/` is gitignored):
`_scratch/postgate_rescore_2026-09/` (`postgate_rescore.py` main run,
`fetch_patch.py` cache-gap overlay, `score_extra.py`, `sanity_backfilled.py`,
`postgate_win364.py`; `daily_scores_final.csv`, `hourly_profile_0909_0922.csv`,
`postgate_forecast.parquet`) and `_scratch/analyze.py`, `decomp.py`,
`drivers.py` (history.json decomposition, naive benchmark, TSO drivers,
ENS-vs-TSO wind proxy). Paths inside point at the state-repo clone and the
website repo.

### Next steps (agreed 2026-09-22, for the 09-23 full-time session)

1. **Make the surrogate cost a tracked number.** Log own-RES and load-de
   outputs (MW per hour) in `forecast_log.parquet`; add a refresh-slot step
   that scores them against the archived 18:00 TSO snapshot and against the
   next day's actuals, so `history.json` can carry a per-day surrogate error.
2. **Night-wind fix as the cheap win.** Target hours 00–06 UTC: fresher
   vintage (12Z evening edition already exists; ICON-EU-EPS switch is parked
   and would cut download time), or an explicit hour-of-day / lead-time
   feature in res-de. Measure against the 6.5 baseline above.
3. **Fix the `post_gate` semantics** on the site: split into "late-generated,
   live inputs" (counts like live, or at least is labelled so) vs
   "reconstructed with official inputs".
4. **First non-linear run (Phase 2 start).** The evening-spike numbers are the
   brief: 364-day linear fit undershoots every scarcity evening. Start with
   the hinge-at-zero / piecewise residual-load idea already registered
   (2026-08-29 addendum) and a gradient-boosted residual on top of LEAR,
   scored on 2026-09 with the post-gate frame from
   `postgate_rescore.py` so input noise is held constant. Also worth a quick
   ablation: shorter-window ensemble (epftoolbox LEAR uses 56/84/1092/1456)
   to see how much of the level-tracking error a faster window buys.



**Production incident (2026-08-30/31): the ENTSO-E Transparency Platform went
down and took the daily forecast with it.** The TSO day-ahead load forecast
for delivery 2026-08-31 never appeared pre-gate anywhere (SMARD had nothing
either — a source-data stall, not just the platform); by 08-31 morning the
whole platform (web + API) returned 503. Delivery 08-31 is the first missed
day since going live. All other inputs survived: weather (ECMWF/DWD chain),
capacity (energy-charts), price history (cached), and every archive dataset
kept landing. Precedent says this can last: the platform's December 2025
crash ran **seven days**. Conclusion: the TSO load forecast is the daily
forecast's only hard delivery-time dependency on ENTSO-E — so it gets a
fallback, priced the same way the 12Z weather fallback was priced.

### Registered experiment (2026-08-31): load-forecast surrogate (`load-de`)

**Hypothesis: a small tree model on weather + calendar imitates the TSO
day-ahead load forecast well enough that losing the ENTSO-E feed costs the
price forecast about as little as the 12Z weather fallback (+0.004 rMAE /
+0.14 EUR/MWh).**

Design (mirrors `res-de`): HistGradientBoosting, expanding window with
monthly refits, every month predicted by a model trained strictly on earlier
data. **Target is the TSO day-ahead load forecast, not actual load** — LEAR's
weights were calibrated against that series including its biases, so the
surrogate imitates the missing *input*, not the physical quantity. Features:
the D−7 load forecast at the same hour (the naive copy, demoted from method
to feature — HGB tolerates it going NaN in a long outage), ENS temperature
and radiation ensemble stats from `ens_features.parquet` (t2m nat/south,
ssrd nat/east/west/south; the ECMWF chain is independent of ENTSO-E and
stayed up through this outage), hour, weekday, day-of-year, and
`holiday_share` for D−1/D/D+1 — population-weighted share of Germany on
public holiday (computus + fixed dates + regional dict; deterministic,
offline, no calendar service to fail). Span: 6-variable ENS era, first fit
2024-10-01 (the `res-de` v2 protocol).

Endpoints, decision rule pre-committed:

1. **Surrogate level (E1)**: OOS MAE/nMAE vs the D−7 copy baseline, overall
   and on the holiday-affected slice (`holiday_share > 0` on D−1/D/D+1).
   The tree must beat the copy overall — else the copy ships and the tree is
   recorded as refuted.
2. **Price level (E2)**: `lear-de` academic, window 364, test 2024-10-01..,
   production-mode arms — (A) own-RES 00Z + true load forecast (the
   2026-08-27 00Z swap arm, reference rMAE 0.409, rerun on the identical
   span) vs (B) own-RES 00Z + surrogate load. The delta is the cost of
   losing the ENTSO-E load feed entirely.
3. **Wiring rule**: E2 cost ≤ +0.005 rMAE → the surrogate becomes a flagged
   automatic fallback that still counts in the public scorecard (like 12Z
   weather days: badge + measured-cost note). +0.005 to +0.015 → fallback
   publishes but is excluded from the headline mean (like post-gate days).
   Above +0.015 → no auto-publish; a missed day stays a missed day.

Guess, written in advance: the tree lands near 1.5–2.5% nMAE against the
copy's 3–5%, and E2 comes in at or under the 12Z scale — load's LEAR
coefficients matter, but less than RES post-2021. The risk case is holiday
clusters, which is why E1 gets a holiday slice.

**RUN (2026-08-31, laptop, runs load-de-20260831-072618 +
lear-de-20260831-{073835,075144}): both endpoints decided, middle band.**
E1: surrogate nMAE **2.58%** vs D−7 copy 3.56% (16,653 h / 694 days); on the
holiday-touched slice **3.79% vs 7.63%** — the copy's error doubles on
exactly the days the tree was built for. Tree beats copy → copy refuted as
the shipping method. E2 (arm A reran locally at rMAE 0.409, matching the
pod reference exactly): surrogate-load arm rMAE **0.418** — cost
**+0.0090 rMAE / +0.31 EUR/MWh**, concentrated in Q4-2024 (+1.0 MAE);
2025 barely notices (+0.09). The advance guess ("at or under the 12Z
scale") was too optimistic by ~2× — recorded as such. Per the
pre-committed wiring rule, +0.005 < 0.0090 ≤ +0.015 → **the surrogate
ships as an automatic fallback that publishes with its own flag and stays
OUT of the headline 30-day mean** (post-gate-style exclusion, unlike the
counted 12Z weather days). Production wiring is the follow-up step.

**Ship decision (2026-08-31, same day, supersedes the plain-fallback
wiring): the surrogate enables the evening vintage** — the parked
first-post-launch experiment becomes the shipping shape. Flow: on D−2
evening, once the 12Z ENS vintage is archived (~20:35 UTC), publish a
first forecast for delivery day D from 12Z weather + surrogate load (the
TSO load forecast for D does not exist yet in the evening — the surrogate
is the enabler). The normal D−1 morning run then replaces it pre-gate:
true TSO load + 00Z weather when ENTSO-E delivers, surrogate load + 00Z
weather when it does not. The evening forecast carries its own flag; a day
whose *standing* forecast is evening-vintage or surrogate-built stays out
of the headline mean; a day replaced by a true-data morning run is a
normal day. Registered arm C (before running): **12Z weather + surrogate
load — the exact evening product.** Guess in advance: costs are roughly
additive, ≈ rMAE 0.421–0.423 against the 0.409 true/true anchor.

**Arm C RUN (2026-08-31, lear-de-20260831-082242): rMAE 0.421 / MAE 15.02
— +0.012 rMAE / +0.43 EUR/MWh vs the true/true anchor. The additive guess
hit exactly. Middle band confirmed for the full evening product → ships as
designed: published D−2 evening with its own flag, replaced pre-gate the
next morning, excluded from the headline mean whenever it is the standing
forecast.**

## Status addendum (2026-08-29): registered observation — the missing kink at zero

**The linear model mis-shapes negative prices: frequency roughly right, depth
and duration wrong.** Checked 2026-08-29 against the seed backtest
(`lear-de-20260811-074202`, 66,408 h 2019–2026) and the live pre-gate log
(310 h since 2026-08-16). The model calls *fewer* negative hours than clear
(3.2% vs 3.7% backtest; 2.3% vs 5.5% live) but its negatives are far too
deep: median forecast negative **−15.0 €/MWh** (backtest) / **−20.3** (live)
against actual medians **−2.9** / **−1.2** — 5× / 17× too deep. The market's
negatives pin just below zero in long shallow spells (live: every actual
negative hour sat in [−5, −0.4], mean spell 5.7 h); the model either stays
positive or dives. Sign recall 60% backtest / 41% live at 70% / 100%
precision — live it has yet to produce a single false negative-hour call,
it just calls them an order of magnitude too deep.

**Reading: the market has a behavioral floor just under zero that a linear
response cannot represent.** Must-run units bid small negatives rather than
cycle off, and §51 EEG cuts the market premium during negative-price
stretches, so curtailment bids from subsidized RES cluster at small negative
prices and pin the clearing price there. LEAR extrapolates the
residual-load→price slope straight through zero, and the Invariant/asinh
transform is symmetric around the *median* price (far above zero) — no kink
anywhere. Verified: no code inhibits negative price predictions; the only
clip in the chain is the own-RES capacity factor (≥ 0, physical).

**Registered implication for phase 2 (written before any nonlinear model is
built):** a model that can represent the zero regime should (a) lift sign
recall on negative hours well above 60%/41%, and (b) shrink the depth error
on jointly-negative hours from 5–17× toward 1×, without giving back MAE on
positive hours. Cheapest falsification first: a **hinge feature at low
residual load inside plain LEAR**. If that alone captures most of the
effect, "this needs a neural net" is overstated and gets recorded as such —
the same bar every other improvement here has had to clear.

## Status addendum (2026-08-27): 12Z backfill COMPLETE; open-data access tightened

**12Z vintage archive complete: 890/890 dates (2024-03-19..2026-08-25), QA
green** (all files readable, 50-51 members, 6 variables, consistent row
counts). This unblocks the registered 12Z-vs-00Z swap experiment (day-9
OPEN item): rerun the res-de swap on a pod to price the fallback-day
accuracy penalty for honest site disclosure.

**Why the pod backfill had stalled (~300 missing dates): ECMWF/AWS
tightened open-data access ~2026-08-18, unannounced** (forum-only trail;
ECMWF: "storage is in charge of AWS"). Measured empirically 2026-08-26:
(1) the S3 bucket deterministically 503s the default `python-requests`
User-Agent from ANY ip; (2) cloud/datacenter IPs (AWS EC2, RunPod NAT) are
hard-throttled even with allowlisted UAs — bulk range reads refused;
(3) Azure mirror now requires a SAS token. The "download from inside AWS"
hypothesis was tested and REFUTED: SigV4-signed requests from an EC2
instance in the bucket's own region throttle identically (~$1 of EC2 spend,
account kept for future use). **Fix that worked: Google mirror**
`storage.googleapis.com/ecmwf-open-data` — full archive, no UA filter, no
cloud-IP throttling, 30-150 MB/s — plus curl-class UA for the S3 fallback
and env-tunable request pacing (`PEP_ENS_PACING_S`, default 0.5 s kept for
production; 0.2 s used for the backfill). Four cheap pods overnight
(28 workers, ~50 dates/h peak; OOM on 4 GB pods at 6 workers — cap at 4).

**12Z-vs-00Z swap RUN (2026-08-27, 32-core pod, runs
{res,lear}-de-20260827-*): fallback-day penalty is NEGLIGIBLE at the price
level.** Same registered setup as 2026-08-15 (academic exog, window 364,
test 2024-10-01.., n=16,680 h / 695 days, identical eval span both arms):
LEAR swap rMAE 0.409 (00Z) vs 0.413 (12Z) — **+0.004 rMAE / +0.14 EUR/MWh
MAE**. The weather-level degradation is real but the price model barely
feels it: onshore nMAE 2.23→2.40%, offshore 4.73→5.20%, solar unchanged
(0.92%), aggregate RES MAE 2,109→2,262 MW. Site can disclose: "fallback
mornings use 12-hour-older weather, historically ~+0.1 EUR/MWh MAE". The
00Z swap on this extended span (0.409) also revalidates the 2026-08-15
pre-gate cost (+0.026 vs the 0.381 true-exog baseline, was 0.406 on the
shorter span).

**Production incident + lesson (2026-08-26): mirror order is a freshness
decision.** Putting GCS first broke the nightly 12Z fallback job — GCS
syncs a fresh run HOURS after publication, so both evening slots 404'd the
just-published 12Z while backfills (old dates) never noticed. Order is now
freshness-first (S3, data.ecmwf.int, GCS last; commit f3a9dd1); the missing
2026-08-26 vintage was healed by hand into the archive repo (ea8f468).
Corollary of the access-tightening: "we can backfill later" is no longer a
safe assumption — the daily pre-gate archiver is the only reliable source
of history.

## Status addendum (2026-08-17, day 9): 12Z fallback vintage; publish path live

**Production robustness: 12Z ENS fallback shipped.** The publish-forecast
workflow failed 2026-08-15..17 (six runs): first the missing
`WEBSITE_REPO_TOKEN` (now set; website repo pushed to
`baakflo/pred_el_prices_website`, seed JSON in `public/data/`), then —
three runs in a row — S3 `503 Slow Down` throttling exhausted the ECMWF
download retries. The 00Z herd right after publication can throttle the
bucket for longer than any sane in-run retry budget, and a missed morning
is an unrecoverable scorecard gap. Changes:

1. **Evening fallback vintage.** New `archive-ens-12z` workflow (20:35 /
   21:35 UTC) archives the same-day 12Z ENS run (steps +33h..+60h — covers
   delivery D = run date + 2) into the weather archive. `ecmwf.py`,
   `run_features` and `update_features` are run-hour aware; a delivery day
   is "primary-backed" when its rows come from the 00Z run of D-1, and
   fallback rows are replaced by 00Z rows once those backfill (the table
   converges to the backtest vintage).
2. **Fallback is retry-slot-only.** `pep forecast --allow-ens-fallback` is
   passed on the 09:50 UTC slot and manual dispatches, NOT on 09:15 — a
   late 00Z should wait for the retry, not lock in stale weather
   (the forecast log is idempotent per delivery day).
3. **Retry hardening.** ECMWF `_get`: 12 attempts / 15 min per request
   (was 8 / ~6.5 min), exponential backoff to 180 s plus 0-20 s jitter,
   pacing 0.5 s. The morning run also best-effort re-archives the last
   three 00Z runs, healing archive gaps from failed mornings.
4. **Epistemics note:** the fallback does not weaken the pre-gate claim —
   inputs stay pre-gate either way; freshness affects accuracy only.
   OPEN: quantify the 12Z-vs-00Z accuracy cost (backfill 12Z history via
   `pep backfill-ecmwf --run-hour 12`, rerun the res-de swap on a pod) so
   the site can disclose the fallback-day penalty honestly.

## Status addendum (2026-08-15, day 7): pre-gate availability audit; own-RES forecast registered

**Finding (changes the production design): no public source publishes German
day-ahead wind/solar generation forecasts before the 12:00 CEST auction
gate.** Audited empirically on 2026-08-15 morning (D-1 for 2026-08-16) plus
documented schedules: all four TSO portals, netztransparenz (incl. the
WebAPI Vermarktungsprognose successor), ENTSO-E 14.1.D (zone and
control-area), SMARD, energy-charts — all publish at/by 18:00 CEST D-1,
~6 h post-gate. Anchored in law: EEV § 3 (18:00 deadline; the forecast IS
the TSOs' marketed quantity, so pre-gate publication would reveal their
auction bids) and Reg. 543/2013 (18:00 Brussels). The forecasts are
*generated* pre-gate (Amprion's CSV column: "8:00 Uhr Prognose"; 50Hertz:
"data as of 09:00, published 18:00") — so backtests using ENTSO-E 14.1.D
exog carry a mild publication (not information) look-ahead; disclose, keep.
Consequences: (1) the live daily forecast must generate its own RES
forecast from the pre-gate weather archive (ECMWF ENS 00Z, on S3 ~07:00
UTC); (2) the entsoe-forecasts snapshot cron slots are pre-gate and will
never capture wind/solar — add an evening slot (~16:30 UTC) and treat the
series as "TSO evening vintage", not a pre-gate snapshot.

### Registered experiment (2026-08-15): own-RES pre-gate forecast (`res-de`)

**Design (registered before any fit).** Targets: the three TSO day-ahead
forecast series (wind onshore/offshore, solar) from `hourly.parquet`,
normalized to capacity factors via monthly installed capacity
(energy-charts/MaStR, linearly interpolated; Solar AC). Rationale: the TSO
forecast — not the outturn — is what LEAR trained on and what the market
prices off; CF normalization removes the fleet-growth non-stationarity
that a tree model cannot extrapolate. Features: hourly-interpolated
ensemble means from the archived 00Z ECMWF ENS run of D-1 (ws100
nat/north/south/sea computed per member-cell before averaging; ws10 nat;
t2m nat; ssrd de-accumulated, nat/south) plus hour-of-day and day-of-year
harmonics. Model: one sklearn `HistGradientBoostingRegressor` per target,
near-default params. Split: expanding-window monthly refits, first fit
2024-10-01 (≥6.5 months train, 6-var era starts 2024-03-19), out-of-sample
predictions 2024-10-01..2026-07-29. LEAR swap: `lear-de` academic config
with a predict-day-only exog override (training rows keep published TSO
values — exactly production's information set), evaluated on override days.

**Pre-registered predictions (before any fit):**
1. Solar: all-hours nMAE <= 1.5% of AC capacity, R^2 >= 0.97.
2. Wind onshore: nMAE <= 3% of capacity, R^2 >= 0.93.
3. Wind offshore (lumpy fleet, coarse sea cells): nMAE <= 6%, R^2 >= 0.85.
4. Aggregate own-RES vs TSO aggregate: MAE <= 3 GW.
5. LEAR(364, academic) with own-RES override degrades <= +0.02 rMAE vs the
   TSO-exog baseline on the same days.

**Results (2026-08-15; res-de-20260815-061233, lear-de-20260815-{062141
baseline, 062802 override}, 16-core pod):** (1) MET — solar nMAE 0.99%,
R^2 0.977. (2) MET — onshore 2.51%, 0.942. (3) MISSED — offshore 7.46%,
R^2 0.779: an 11 GW fleet in two small sea patches vs one coarse >=54N
cell group mixing North Sea, Baltic and coast. (4) MET — aggregate MAE
2.49 GW (~1.3% of installed RES). (5) **MISSED** — on the 667 override
days MAE 13.53 -> 14.83 EUR/MWh, rMAE 0.380 -> 0.417, i.e. **+0.037**,
nearly double the registered +0.02 bound. Mechanism verified, not
artifact: non-override days bit-identical between runs; own-vs-TSO
same-day corr 0.977 vs 0.731 day-shifted (no alignment bug); daily
price-MAE delta correlates +0.34 with daily own-RES error (input-error
propagation). Read: LEAR is more sensitive to RES-input error than
predicted — 2.5 GW aggregate MAE is not yet cheap. The honest live
number today: pre-gate legality costs ~+0.035 rMAE (0.382 -> 0.417 on
2024-10..2026-08; still ~2.4x better than naive). Improvement levers,
in expected order: offshore North/Baltic cell split, capacity-weighted
cells, wind-speed member quantiles, hub-height power-curve features.

### Registered experiment (2026-08-15): res-de v2 feature iteration

**Motivation.** v1 swap attribution: the worst price-damage days were
driven by onshore wind (up to 4.7 GW daily MAE) and solar (3.7-4.8 GW)
blowups, NOT offshore (0.7 GW mean). One bounded feature iteration before
launch; same model, same expanding-monthly split, same spans.

**Design.** Feature set v2 (21 features): wind 100 m speed over
north/center/south belts plus separate North Sea (>=54N, <=8E) and Baltic
(>=54N, >=10E) groups; ensemble q10/q90 across members for ws100
nat/northsea/baltic and ssrd nat; ssrd east/west split (morning/evening
cloud asymmetry); t2m south added. Swap rerun: same baseline run
(lear-de-20260815-062141), override with v2 predictions.

**Pre-registered predictions (before any fit):**
1. Wind onshore nMAE <= 2.2% (v1: 2.51).
2. Wind offshore nMAE <= 6.0%, R^2 >= 0.85 (v1: 7.46 / 0.779).
3. Solar nMAE <= 0.85% (v1: 0.99).
4. Aggregate MAE <= 2.1 GW (v1: 2.49).
5. Swap degradation <= +0.025 rMAE on override days (v1: +0.037).

**Results (2026-08-15; res-de-20260815-064815, lear-de-20260815-065153;
note: dataset refresh extended eval to 680 days vs v1's 668):**
(1) MISSED by a hair — 2.24 vs 2.2 (from 2.51). (2) MET — 4.76%, R^2
0.907 (from 7.46/0.779): the North Sea/Baltic split was the right
mechanism. (3) MISSED — 0.92 vs 0.85 (from 0.99). (4) MISSED by 10 MW —
2.11 vs 2.1 GW (from 2.49). (5) MISSED by 0.0007 — **+0.0257** vs
+0.025 (from +0.037; baseline 0.381 -> own-RES 0.406 on the 680
override days). Read: the iteration cut the pre-gate price by ~30% and
fixed offshore emphatically; the other bars were set aggressively and
missed by rounding-level margins. DECISION: ship v2 for launch — live
expectation rMAE ~0.41, still ~2.4x better than naive, with the honest
"+0.026 cost of pre-gate legality" as a first-class site number. v3
levers (capacity-weighted cells via MaStR coordinates, hub-height power
curve, per-farm offshore features) are post-launch work.

## Status addendum (2026-07-30, end of day 1)

Progress log lives in git history and `reports/`; decisions with reasoning in
`docs/design_notes.md`. State: **Phase 0 complete, Phase 1 LEAR complete.**

Done: data layer (ENTSO-E 5 datasets + SMARD cross-check + fuels, 2015->today,
QA'd: prices cross-validated between portals at corr 1.0); weather archiver
live (ICON-EU-EPS, daily cron); `pep build-dataset` (101k hourly rows,
leakage-safe); LEAR reimplemented and validated on the Lago 2021 EPEX-DE
benchmark (short windows within 1-2%; long windows BEAT published numbers by
4-7% due to sklearn's corrected AIC — see reports/lear_reproduction/); first
baseline on own data (`pep run lear-de`, window 364, 2019-2026: overall rMAE
0.492; worst relative year 2021 rMAE 0.588 = measured cost of implicit drift;
see reports/lear_de/). RunPod CPU workflow validated (template auto-pulls repo;
32-core run = 15 min, see deploy/runpod/).

### Next steps (agreed 2026-07-30)

1. **ECMWF ENS backfill probe (first thing next session):** ECMWF open-data
   forecasts (51-member ENS) on a public AWS bucket since ~2023. If usable,
   backfill a Germany-level ensemble-spread series and run the
   spread-vs-LEAR-error correlation immediately (the Phase 3 go/no-go signal)
   instead of waiting for our own archive to mature. Resolution (0.4/0.25 deg
   vs ICON 13 km) is acceptable because spread features aggregate to country
   scale anyway.
2. Literature sweep on ensemble-weather-in-EPF before claiming novelty:
   demand (Taylor & Buizza 2003) and wind/PV power are mature; direct
   "spread conditions the price distribution's scale" appears under-grazed.
3. Window ablation on own data (~15 pod-min per window) to fill the
   lear_de results table.
4. Distributional model phase: small MLP + Johnson's SU head (softplus links),
   NLL training, per docs/design_notes.md priors. Two-stage design for the
   spread feature (main model without, tiny recalibration layer with) so it
   trains on months, not years, of archive.

### Caveats and open items collected on day 1

- **UTC-day convention:** lear-de uses UTC day blocks; German delivery days
  are local-midnight aligned (1-2 h shift). Harmonize when building the
  evaluation harness; matters for hour-index interpretation.
- **Leakage audit pending:** TSO day-ahead forecasts are formally published
  D-1 evening (after the 12:00 gate) on ENTSO-E/SMARD; using them is the
  benchmark convention but needs an explicit leakage note in any write-up.
  Fuel settlements enter with a 2-day lag (settlement of D-2 for delivery D).
- **Point-forecast peak blindness is structural:** MAE-optimal = conditional
  median, so LEAR/DNN point models systematically under-call spikes. Add a
  dedicated spike-day scorecard (tail calibration, CRPS conditional on
  price > quantile) to the harness; this is where the probabilistic model
  must earn its keep.
- **sMAPE/MAPE are dying metrics** on negative-price data (visible post-2023);
  rely on rMAE + proper scores. rMAE convention: weekly-persistence naive
  (matches published Lago tables; the mixed naive is the harder anchor, also
  logged).
- **Neural phase priors are pre-registered** in docs/design_notes.md:
  fixed ReLU (merit-order piecewise-linearity argument; SwiGLU swap w/
  warm-start when embedding in a larger differentiable system + GAM/A-B test
  receipts); width plateau expected >=64-128 with narrow-net seed variance;
  input is low-rank (~20-40 effective dims of 391) so first layer compresses
  — the anti-superposition regime (dense few factors, surplus width). Run
  effective-rank (participation ratio) diagnostics per test year on trained
  nets; a crisis-period rank rise = capacity recruitment (report-worthy).
  Sparse rare-event detectors (holidays, scarcity, negative-price regimes)
  are the one place superposition-like structure could appear.
- **Seed variance protocol:** seed ensembles (4-8) are the model; report
  variance across full-pipeline reruns; LEAR is the deterministic control.
  Re-run hyperopt at test-year boundaries (hyperparameter staleness).
- **LEAR n<p note:** window 364 on own data = 357 samples vs 391 features;
  lives on lasso sparsity; sklearn needs explicit noise_variance there
  (we use per-hour target variance — deviation from paper, documented).
- **Data watch-list:** EUA proxy starts 2021-10 (stitch Ember for earlier);
  API2 coal ticker stale since 2025-12; pandas 3 vs entsoe-py pin risk;
  ENTSO-E load_forecast gaps (Sep-Dec 2018 + 2022 outage days) are patched
  from SMARD in build-dataset.
- **Infra notes:** repo pulls to pods via deploy/runpod/ template (secret
  GITHUB_PAT, image runpod/base:0.6.3-cpu for CPU work); artifacts come home
  via scp into runs/ (gitignored); reports/ is the public face — every
  analysis becomes a page, no notebooks.

## Status addendum (2026-07-31, day 2)

**ECMWF ENS backfill probe: GO.** The open-data archive on AWS
(`ecmwf-forecasts` S3 bucket, anonymous access) is usable for backfilling a
Germany-level ensemble-spread series:

- **Coverage:** daily 00z ENS runs (leakage-safe: published ~07-08 CET on
  D-1) from **2023-01-18** to today; 3-hourly steps. Layout changed twice
  (`0p4-beta` 0.4 deg → `0p25` Feb 2024 → `ifs/0p25`); handled.
- **Variables:** 10u/10v/2t over the whole archive; **ssrd (solar) only from
  2024-03-10** (plus 100u/100v from ~2024-03-20). So wind/temp spread has
  ~3.5 years of history, solar spread ~2.4 years.
- **Members:** 50 perturbed + control until early files; current `-ef` files
  carry 50 perturbed only (control absent). Spread from 50 members is fine;
  member-0 presence differs across eras (documented in module).
- **Access pattern:** each step is one global ~2.5 GB GRIB, but the `.index`
  sidecar enables HTTP range requests for just our surface fields:
  ~0.8 GB/date (2023, 3 vars) to ~2.7 GB/date (current, 6 vars). S3 throttles
  bursts with 503 Slow Down — pacing + patient backoff required (implemented).
- **Implementation:** `src/pred_el_prices/pipeline/ecmwf.py` +
  `pep backfill-ecmwf --start ... --end ...`. Output schema mirrors the ICON
  archiver (per-member 1-degree-cell means, one Parquet per run) so
  downstream features treat both archives identically. Validated end-to-end
  on 2023-03-15 (3 vars x 51 members) and 2026-07-28 (6 vars x 50 members);
  country-mean spreads physically sane. ssrd is accumulated-since-start;
  de-accumulate downstream.
- **Cost/decision:** ~15 min/date locally (bandwidth + throttling) — the full
  ~1,290-date backfill (~2 TB download, tiny Parquet output) is a **RunPod
  job with parallel date workers**, not a laptop job.

### Next steps (agreed 2026-07-31)

1. Pilot backfill (2-4 weeks of dates) on a pod: verify throughput, tune
   worker count against S3 throttling, sanity-QA the spread series.
2. Full 2023->now backfill on the pod; scp Parquets home into the archive.
3. Spread-vs-LEAR-error correlation (the Phase 4 go/no-go signal): join
   daily wind/temp spread against lear-de per-day absolute error.
4. Then resume prior list: literature sweep on ensemble-weather-in-EPF,
   window ablation, distributional model.

## Status addendum (2026-08-11, day 3)

**Same-horizon comparison vs the academic baseline: done** — see
`reports/lear_same_horizon/`. Test horizon fixed to the epftoolbox 728-day
period (2016-01-04..2017-12-31); windows 56/84/364 x {academic 2-exog,
extended 4-exog} on our data vs published + reproduced numbers. Findings:
(1) our data pipeline beats the benchmark data at matched windows (rMAE
0.482/0.468 vs published 0.506/0.499); (2) best config overall is window
364 + academic exog (MAE 3.614, rMAE 0.396); (3) the extended RES split
*hurts* on this pre-2018 horizon at every window — pre-registered
expectation: it pays off post-2021 (test in the 2019-2026 ablation).

**ECMWF pilot backfill on a 32-core pod: done, QA green.** 28 dates
(2023-02) + QA date archived. Learnings: throughput ~4 min/date/worker in
the 3-var era (~3.5x laptop), zero S3 503s at 4 workers; pods need system
`libeccodes-dev` (now in bootstrap.sh — Linux wheel has no binary); the
pod's old system ecCodes 2.16 decodes byte-identically to the modern
laptop version (validated on 2023-03-15, max diff 0).

**Full 2023->now backfill: running on a cheap pod** (4 vCPU / 8 GB,
~$0.12/h; the job is bandwidth-bound — est. ~2-2.5 days, <10 EUR).
Data-safety: hourly incremental pull to the laptop via
`deploy/runpod/pull_archive.ps1` (Task Scheduler job `pep-pull-ecmwf`);
loss window <=1 h; if the pod dies, push the local archive up and re-run
the same command — per-date Parquets make it fully resumable. Worker
count is disk-capped: each worker holds up to ~2.7 GB GRIB temp (current
era), so 4 workers on a 20 GB container disk.

### Next steps (agreed 2026-08-11)

1. Monitor the backfill via the pull log; when the 2023 era is home, start
   the spread-vs-LEAR-error correlation (overlap 2023-01..2026-07 with
   lear-de daily errors) — no need to wait for the full archive.
   **2023-era result (same day, reports/spread_vs_error): pre-registered
   prediction (Spearman 0.2-0.4) NOT supported** — raw wind-spread vs
   daily-MAE Spearman only +0.13, and *zero* partial correlation once
   plain windiness (ensemble-mean wind) is controlled for; t2m spread
   nothing. Caveat: 10 m wind only in this era. Registered re-test on the
   post-2024-03 span (100 m wind + ssrd spread) once the full archive
   lands; prediction to beat: partial Spearman > +0.15.
2. Window/feature ablation on 2019-2026 (tests the pre-registered
   "extended split pays off post-2021" hypothesis from the day-3 report).
   **Done same day — hypothesis REFUTED** (reports/lear_feature_ablation):
   the academic aggregate wins nearly every year at every window, widest
   gap in 2021-2022 at window 364. New point-forecast floor: **LEAR(364,
   academic exog) rMAE 0.407** (was 0.492 with the split). RES split
   shelved for linear models; revisit only as a neural-phase ablation.
3. Then: literature sweep, distributional model phase.

### Registered experiment (2026-08-11): window-546 ablation rerun

**Motivation.** The day-3 ablation compared exog sets at windows where the
extended model is structurally handicapped: at window 364 it sits at n < p
(391 weights, 357 training rows) while the academic model does not (247).
A selection-churn probe (15 days, June 2024, `_scratch/selection_churn.py`)
showed the mechanism is not LASSO flip-instability (day-to-day support
churn ~0.20 Jaccard in *both* configs) but admitted signal: AIC's alpha
prunes the extended exog block to ~10.6 nonzero columns vs ~29.0 for the
academic. The window-364 verdict may therefore reflect the fitting regime,
not the feature set.

**Design.** `pep run lear-de --set window=546 --set exog={academic,extended}`
(546 days = 1.5 y, 78 weeks; 539 training rows > 391 weights, so both
configs are n > p and `LassoLarsIC` needs no noise_variance fallback).
Same test span 2019-01-01..today, same per-year metrics. Pod job
(`_scratch/pod_run_ablation_w546.sh`) — NOT to be run on the laptop.
Secondary endpoint: rerun the churn probe at 546 for both configs.

**Pre-registered predictions (before any fit):**
1. The extended-vs-academic gap narrows substantially at 546 (from
   0.085 overall rMAE at 364 to under 0.03), because the shrinkage-tax
   asymmetry disappears.
2. Academic still wins or ties overall (rMAE_546_ext >= rMAE_546_acad -
   0.005): collinearity variance and the fixed linear readout remain even
   at n > p.
3. Academic-546 lands within ~0.01 rMAE of academic-364 overall, but is
   worse in the 2021-2022 regime-break years and better in calm years
   (long windows adapt slower).
4. Churn probe at 546: extended's admitted exog columns at least double
   (from ~10.6 toward the academic's ~29).

**Results (2026-08-13, runs lear-de-20260813-{065639 acad, 070141 ext}):**
academic-546 rMAE 0.408 / MAE 13.95; extended-546 rMAE 0.410 / MAE 14.04.
Prediction scorecard: (1) CONFIRMED, gap collapsed 0.085 -> 0.002 (even
below the predicted <0.03) — the window-364 verdict was indeed mostly
shrinkage regime, not feature content; (2) CONFIRMED, academic still
wins by 0.002; (3) HALF-WRONG instructively: academic-546 is within
0.001 of 364 overall as predicted, but the per-year pattern inverts the
story — 546 is *better* in 2021 (-0.019) and 2022 (-0.002) and pays its
whole cost in 2023 (+0.043): long windows hurt on regime *exit* (stale
crisis data), not regime entry; (4) CONFIRMED, extended's admitted exog
went 10.6 -> 46.8 columns (academic stable 29 -> 27.3), churn Jaccard
similar (0.180 vs 0.162). **Verdict: the extended RES split is now
fairly tested at n > p and still doesn't pay — academic exog stays the
config (on-par, simpler). Point-forecast floor unchanged: LEAR(364,
academic) rMAE 0.407.**

### Registered experiment (2026-08-13): benchmark-gap decomposition

**Finding to explain** (analysis in `_scratch/compare_epf_data.py`): our
data beats the epftoolbox DE dataset at matched config on their own
horizon (rMAE 0.482 vs published 0.506) with *bit-identical prices* —
so the gap must come from exog scope: their load forecast is
Amprion-only (34% of zone load, corr 0.983 to ours), their RES forecast
~11% short (consistent with missing offshore; corr 0.994), plus
flattened DST hours. No later data corrections, no alignment bug
(best cross-corr lag 0).

**Design.** Same-horizon LEAR (2016-01-04..2017-12-31, window 364,
academic exog structure), three data variants: (a) ours as-is [done,
0.482]; (b) ours with load replaced by Amprion control-area forecast
(ENTSO-E per-control-area query); (c) ours with RES minus offshore.
Optionally (d) = b+c, expected to approach the published 0.506.

**Pre-registered predictions (before any fit):**
1. Amprion-only load explains the larger share: variant (b) loses at
   least +0.012 rMAE vs (a).
2. Offshore removal is secondary but nonzero: (c) loses +0.003..0.010.
3. Combined (d) lands within 0.008 of the published 0.506 — i.e. exog
   scope (plus the small DST flattening we do not replicate) accounts
   for essentially the whole gap.

## Status addendum (2026-08-13, day 5): ECMWF backfill COMPLETE

**Full archive home and validated: 1,295 of 1,301 dates (2023-01-18..
2026-08-10), integrity QA green** (all files readable, >=50 members,
clean variable-era progression 3 vars -> 4 vars @2024-03-06 (ssrd) ->
6 vars @2024-03-19 (100 m wind)). The 6 missing dates 2023-04-27..
2023-05-02 are an **upstream hole in ECMWF's open-data S3 archive** (no
ENS index in any path layout; failed identically on two independent
runs) — unrecoverable, downstream joins must tolerate the gap. Lesson
for future backfills: per-date failures can be silent; always run a
calendar completeness check before declaring done (the first "done"
state was quietly missing 14 scattered dates beyond the known block).
Pod cost ~52 h x $0.12 ~= $6.30, within the <10 EUR estimate.

**Full-span spread-vs-error re-test: registered prediction REFUTED
(final).** n=862 days (2024-03-20..2026-07-29): partial Spearman(100 m
wind spread, LEAR MAE | wind mean) = **-0.002** (p=0.95) vs the +0.15
bar; even the raw Spearman collapsed (+0.164 interim -> +0.026 final) —
the interim look's p=5.5e-04 did not survive, a textbook
optional-stopping exhibit now documented in
`reports/spread_vs_error_retest/`. Ensemble spread at daily national
aggregation carries no incremental error signal in either era;
hypothesis closed for the linear phase. If ensemble info enters the
neural phase: hourly/regional spread or member-level residual-load
features, scored with CRPS.

## Status addendum (2026-08-12, day 4)

**Spread-vs-error re-test (6-var era): interim run, prediction NOT met.**
Backfill reached 2025-06-03 (~60% of the 6-var era), enough to run the
registered re-test early — see `reports/spread_vs_error_retest/`
(n=441 days, 2024-03-20..2025-06-04). Partial Spearman(100 m wind spread,
LEAR MAE | wind mean) = **+0.088** (p=0.065) vs the registered bar of
+0.15. Better than the 2023-era zero, but weak; ssrd-spread partial is
*negative* (-0.143), read as seasonal confounding, not signal. Final
verdict deferred to the full-span re-run when the backfill lands
(threshold unchanged). Full backfill ETA ~Aug 13 morning (throughput
halved in the 6-var era: ~16-19 dates/h, bandwidth-bound + shared-host
contention).

**Rhine level (Pegel Kaub) data acquired — drought/cooling-water feature
candidate** (motivated by the 2026 drought + 2026-08-12 eclipse price
spike). PEGELONLINE serves only a rolling ~31-day raw window, so:
(1) history 2007-11..2025-12 parsed from the BfG DGJ yearbook PDF into
`data/archive/water/pegel-kaub-dgj.parquet` (daily W+Q, 0 gaps, 0
cross-edition mismatches, golden extremes verified; parser
`_scratch/parse_dgj_kaub.py`); (2) 15-min W+Q now archived daily via
`pep archive-pegel` in the archive-weather GitHub workflow (self-healing
within the 31-day window), plus the current window pulled locally.
Known hole: 2026-01-01..2026-07-12 (no public source until the 2026
yearbook, ~mid-2027) — any rolling-window model using the feature must
handle it. Notable: current level ~11 cm is below the post-1880 record
low (25 cm, Oct 2018). Leakage note: DGJ values are after-the-fact
validated daily means; production would use the raw D-1 morning reading.

**2026 bridge via neighbor-gauge regression: attempted and REJECTED
(2026-08-13).** Düsseldorf open data publishes annually (no 2026);
sole open machine-readable 2026 source found was GKD Bayern (Main at
Kleinheubach, fetcher `_scratch/fetch_gkd_kleinheubach.py`). Best
regression (log-Q lags/rollings + seasonal harmonics) reached only
R2~0.60 on 2023-2025 holdout, and the decisive out-of-sample check on
the Jul/Aug 2026 PEGELONLINE overlap failed: +152 cm bias — the Main
cannot see the Alpine/Upper-Rhine deficit driving the record low, so
the bridge is wrong exactly in the regime the feature targets. Bridge
parquet trashed. Sources checked and capped: PEGELONLINE (31 d), WSV
file share (55 d), Undine (images only), RLP portal API (days), Wayback
(1 snapshot), HVZ BW (no API surface). Recommended fix: one email to
the BfG/WSV Datenstelle (Datenstelle-M1@bafg.de) requesting raw Kaub W
for Jan-Jul 2026; meanwhile drought-feature studies proceed on
2019-2025 (complete, includes 2018 + 2022 droughts).

## Status addendum (2026-08-13, day 5)

**Window-546 ablation rerun: DONE** (32-core pod; artifacts
`runs/lear-de-20260813-{065639,070141,071933}`, job log archived next to
the extended run). Scorecard vs the four pre-registered predictions
(registered 2026-08-11, before any fit):

1. **Gap narrows below 0.03 — MET, emphatically.** Extended-vs-academic
   overall rMAE gap collapses from 0.085 at window 364 to **0.002** at
   546 (academic 0.408, extended 0.410).
2. **Academic still wins or ties — MET.** Academic wins by 0.002, inside
   the registered 0.005 tie band.
3. **Academic-546 ~ academic-364 overall, worse in 2021-22 — PARTIAL.**
   Overall within 0.001 (0.408 vs 0.407), but the predicted year pattern
   is *inverted*: 546 is slightly better in the regime-break years
   (2021: 0.423 vs 0.442; 2022: 0.376 vs 0.378) and worse in 2023
   (0.541 vs 0.498) and 2024 (0.398 vs 0.392). The "long windows adapt
   slower" intuition did not show up at this window scale.
4. **Extended's admitted exog columns at least double — MET, 4.4x.**
   Churn probe at 546: extended admits **46.8** nonzero exog cols/hour
   (was ~10.6 at 364; academic stable at 27.3 vs ~29.0). Day-to-day
   Jaccard churn 0.18/0.16 — selection stability unchanged.

**Interpretation.** Mechanism confirmed: the RES split's loss at 364 was
the n < p shrinkage tax, not the features themselves. At 546 the extended
block is fully admitted — and still doesn't win. That is the strongest
form of the refutation: the features got their fair shot. Point-forecast
floor unchanged: LEAR(364, academic) rMAE 0.407, with academic-546 an
effective tie (0.408).

**Convergence footnote.** The coordinate-descent Lasso stage hits its
2500-epoch cap unconverged in a minority of hour-fits (sampled via
`_scratch/check_convergence.py`: 16% ext-546, 11% acad-364, 7% acad-546,
0.6% ext-364; the LassoLarsIC/LARS stage always converges, max ~1.4k of
2500 steps). Sensitivity rerun of extended-546 with max_iter=10000
(run 071933): overall MAE 14.036 vs 14.037, rMAE identical (0.410),
per-year identical to 3 dp — truncation is immaterial. Cap stays at 2500
(epftoolbox-faithful).
