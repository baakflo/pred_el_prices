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

## Status addendum (2026-08-31): ENTSO-E outage; load-surrogate registered

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
