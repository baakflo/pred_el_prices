"""Shadow forecast of the zero-shot foundation model t0-beta: logged, never published.

An out-of-sample forward test of t0 against the live nets (P65-P69 arm B, same
convention). Rows go to `t0_log/YYYY-MM.parquet` next to the nets log and appear in no
site JSON. The step is fail-soft: nothing here may cost LEAR or the nets the day.

Per delivery UTC day D:
- origin = Berlin-local midnight starting local day D (UTC 23:00 of D-1 in winter,
  22:00 in summer), the first hour not yet cleared at the D-1 12:00 CET gate;
- context = the CONTEXT hourly prices before the origin; horizon = 26 h from the
  origin, of which the 24 UTC hours of D are kept.
Covariates, in the order [residual_load, holidays], span context + horizon:
- context hours: TSO load - TSO RES (all of local day <= D-1, public before the gate);
- horizon load: the run's own DE load input for the UTC hours of D (TSO forecast with the
  boundary hours from 24 h earlier, or the surrogate); UTC 22-23 of D-1 (local D) take
  the TSO forecast, else the value 24 h earlier; UTC 00 of D+1 (winter) 24 h earlier;
- horizon RES: own RES only, never the TSO RES (published 18:00 D-1, after the gate).
  UTC hours of D: own RES of D. UTC 22-23 of D-1: own RES of delivery D-1 as made at
  the D-2 gate (fallback: own RES of D 24 h later). UTC 00 of D+1: own RES of D 24 h
  earlier;
- holidays: federal public holiday flag of the Berlin-local day.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import RES_TARGETS, own_res_parts, own_res_total
from pred_el_prices.features.holidays import holiday_share
from pred_el_prices.production.site import _write_partitions

MODEL_ID = "theforecastingcompany/t0-beta"
# pinned: the weights the P65-P69 backtest ran on (HF main as of 2026-09-24); an upstream
# update must not silently swap the model mid forward test
MODEL_REVISION = "c8885416fab935d604749a90cdcbf9b54fffcaeb"
LEVELS = [0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99]
Q_NAMES = [f"q{round(q * 100):02d}" for q in LEVELS]
CONTEXT = 8192
HORIZON = 26
LOG_DIR = "t0_log"  # monthly partitions, like the nets log
RES_COLS = list(RES_TARGETS)
H = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def origin_of(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(day.date()).tz_localize("Europe/Berlin").tz_convert("UTC")


def horizon_load(load_tso: pd.Series, load_d: pd.Series, hz: pd.DatetimeIndex, day) -> np.ndarray:
    """Load over the horizon: `load_d` on the UTC hours of D, the TSO forecast of local
    day D elsewhere, local day D+1 (not published pre-gate) from 24 h earlier."""
    local = hz.tz_convert("Europe/Berlin").date
    src = hz.where(local == day.date(), hz - DAY)
    load = load_tso.reindex(src).to_numpy(dtype=float, copy=True)
    for k in (1, 2):  # remaining gaps: the same hour k days earlier
        gap = np.isnan(load)
        if gap.any():
            load[gap] = load_tso.reindex(src[gap] - k * DAY).to_numpy()
    in_d = hz.normalize() == day
    load[in_d] = load_d.reindex(hz[in_d]).to_numpy()
    return load


def horizon_res(res_d: pd.Series, res_prev: pd.Series | None, hz: pd.DatetimeIndex, day):
    """Own RES over the horizon (`res_d`: own RES of D; `res_prev`: of D-1, or None)."""
    res = np.full(len(hz), np.nan)
    for i, h in enumerate(hz):
        if h.normalize() == day:
            cands = [(res_d, h)]
        elif h < day:  # UTC 22-23 of D-1
            cands = [(res_prev, h), (res_d, h + DAY)]
        else:  # UTC 00 of D+1
            cands = [(res_d, h - DAY)]
        for s, t in cands:
            v = np.nan if s is None else s.get(t, np.nan)
            if np.isfinite(v):
                res[i] = v
                break
    return res


def build_inputs(delivery, prices, dataset, load_d, res_d, res_prev, context=CONTEXT):
    """Model inputs for `delivery`: (context prices, covariates [2, context + HORIZON],
    kept horizon hours, keep mask). Raises when a horizon covariate is missing (t0 would
    read a NaN over the horizon as 0)."""
    o = origin_of(delivery)
    ctx = pd.date_range(o - context * H, periods=context, freq="1h")
    hz = pd.date_range(o, periods=HORIZON, freq="1h")
    keep = hz.normalize() == delivery
    ctx_prices = prices.reindex(ctx).to_numpy(np.float32)
    load_c = dataset["load_forecast_mw"].reindex(ctx).to_numpy(dtype=float)
    res_c = dataset[RES_COLS].sum(axis=1, min_count=len(RES_COLS)).reindex(ctx).to_numpy()
    load_h = horizon_load(dataset["load_forecast_mw"], load_d, hz, delivery)
    res_h = horizon_res(res_d, res_prev, hz, delivery)
    if np.isnan(load_h).any() or np.isnan(res_h).any():
        raise RuntimeError(f"horizon covariates incomplete for {delivery:%Y-%m-%d}")
    rl = np.concatenate([load_c - res_c, load_h - res_h])
    hol = (holiday_share(ctx.append(hz), 0).to_numpy() >= 1.0).astype(float)
    cov = np.stack([rl, hol]).astype(np.float32)
    return ctx_prices, cov, hz[keep], keep


def load_model():
    import torch
    from t0 import T0Forecaster

    return T0Forecaster.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(torch.device("cpu")).eval()


def forecast_day(model, delivery, prices, dataset, load_d, res_d, res_prev) -> pd.DataFrame:
    """24 UTC hourly rows for `delivery`: t0's 21 quantile levels plus the median."""
    ctx, cov, hours, keep = build_inputs(delivery, prices, dataset, load_d, res_d, res_prev)
    fc = model.predict(
        ctx[None], horizon=HORIZON, quantile_levels=LEVELS, future_covariates=cov[None]
    )
    q = fc.quantiles.float().cpu().numpy()[0]  # [HORIZON, 21]
    out = pd.DataFrame(q[keep], index=hours, columns=Q_NAMES)
    out["median"] = out["q50"]
    return out


def t0_step(
    out_dir: Path,
    cache_dir: Path,
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    delivery: pd.Timestamp,
    prices: pd.Series,
    load_d: pd.Series,
    res_parts: pd.DataFrame,
    generated_utc: str,
    weather_vintage: str,
    load_surrogate: bool,
    model=None,
) -> bool:
    """Shadow t0 forecast for `delivery`, appended to the t0 log. Never raises."""
    try:
        t = time.perf_counter()
        prev = delivery - DAY
        try:  # own RES of D-1 as the D-2 morning run made it (TSO RES cut at its gate)
            seen = dataset[dataset.index < prev - 2 * H]
            res_prev = own_res_total(own_res_parts(features, seen, cache_dir, prev))
        except Exception as e:  # noqa: BLE001 - a fallback exists for these two hours
            print(f"t0: own RES of {prev:%Y-%m-%d} unavailable ({e}); D's own RES + 24 h used")
            res_prev = None
        if model is None:
            model = load_model()
        rows = forecast_day(
            model, delivery, prices, dataset, load_d, own_res_total(res_parts), res_prev
        ).astype("float32")
        rows.insert(0, "generated_utc", generated_utc)
        rows.insert(1, "weather_vintage", weather_vintage)
        rows.insert(2, "load_surrogate", bool(load_surrogate))
        rows.index.name = "t"
        _write_partitions(Path(out_dir) / LOG_DIR, rows)
        print(f"t0 shadow: {delivery:%Y-%m-%d} logged in {time.perf_counter() - t:.0f} s")
        return True
    except Exception as e:  # noqa: BLE001 - shadow path: nothing here may cost LEAR the day
        print(f"::warning::t0 shadow step failed ({type(e).__name__}: {e})")
        return False
