"""Gradient-boosted correction of LEAR's out-of-sample error (`lear-gbm-de`).

LEAR is linear per hour; its residuals against the actual price can still
carry a systematic, nonlinear pattern (e.g. under-forecasting scarcity hours).
This trains a HistGradientBoostingRegressor on LEAR's own rolling-backtest
errors (resid = actual - lear_pred, genuinely out-of-sample) and adds the
predicted correction back onto the LEAR forecast.

Evaluation mirrors res-de/load-de: expanding window, monthly refits, each
calendar month predicted by a model trained strictly on earlier LEAR-error
rows.

Leakage: every feature must be knowable before the day-ahead auction gate
closes at 12:00 CET on D-1, for delivery day D.
- hour, dayofweek, sin/cos(day-of-year): calendar, always known.
- load_forecast_mw, res_forecast_mw, residual_load_forecast_mw,
  ttf_gas_eur_mwh, eua_proxy_usd at hour t of day D: TSO day-ahead forecasts
  (pre-auction, per the EPF benchmark convention) and fuel/carbon settlements
  already lagged 2 days in the dataset build (see features/dataset.py).
- day D's residual_load_forecast_mw max/min: from the same day-ahead
  forecast, so known for all 24 hours of day D before the gate.
- lear_pred: the LEAR forecast for hour t of day D, produced pre-gate.
- resid at the same hour on D-1 (lag 24) and D-7 (lag 168), and mean |resid|
  over day D-1: LEAR's D-1 forecast was made pre-gate on D-2, and local day
  D-1 cleared in the D-2 auction. But UTC days are not delivery days: UTC
  22-23 of D-1 belong to local day D, i.e. to the auction being forecast, so
  the D-1 features exclude those two hours.

Shifts are computed on a continuous hourly index (the base run is reindexed
to a full hourly range first) so lags are by elapsed time, not row position.
HistGradientBoostingRegressor handles NaN features natively (fuel columns
start NaN, lags are NaN at the start of history) — rows are only dropped
from training where the *target* (resid) is NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pred_el_prices.eval.metrics import mae
from pred_el_prices.eval.scorecard import load_forecast

RES_COLS = ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]


def fuel_cost_scale(dataset: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Rough gas-plant marginal cost in EUR/MWh, floored at 20: 2 x TTF + 0.37 x EUA.

    Both inputs carry the dataset's 2-day settlement lag. The EUA proxy starts
    2021-10 and counts as 0 before (no back-fill: that would read the future).
    """
    ds = dataset.reindex(index)
    srmc = 2.0 * ds["ttf_gas_eur_mwh"].fillna(0.0) + 0.37 * ds["eua_proxy_usd"].fillna(0.0)
    return srmc.clip(lower=20.0)


def build_features(
    fc: pd.DataFrame,
    dataset: pd.DataFrame,
    groups: tuple[str, ...] = (),
    direct: bool = False,
) -> pd.DataFrame:
    """Design matrix for correcting LEAR's error, indexed like `fc`.

    `fc` has columns `pred` (LEAR forecast) and `actual`, e.g. from
    eval.scorecard.load_forecast. `dataset` is the hourly feature/target
    table (data/dataset/hourly.parquet); only day-ahead-known columns are
    used. See the module docstring for the leakage argument per feature.

    `groups` adds optional feature groups: "neighbours" (day-ahead residual
    load of the neighbouring zones, same TSO-forecast convention as DE's).
    `direct` drops everything derived from LEAR and adds price lags instead,
    for a tree that forecasts the price itself; the D-1 lag skips UTC 22-23
    for the same reason the D-1 error features do.
    """
    idx = fc.index
    full_idx = pd.date_range(idx.min(), idx.max(), freq="1h", tz=idx.tz)
    fc_full = fc.reindex(full_idx)
    resid = fc_full["actual"] - fc_full["pred"]

    x = pd.DataFrame(index=full_idx)
    x["hour"] = full_idx.hour
    x["dayofweek"] = full_idx.dayofweek
    doy = full_idx.dayofyear.to_numpy()
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    ds = dataset.reindex(full_idx)
    x["load_forecast_mw"] = ds["load_forecast_mw"]
    x["res_forecast_mw"] = ds[RES_COLS].sum(axis=1, min_count=1)
    x["residual_load_forecast_mw"] = ds["residual_load_forecast_mw"]
    x["ttf_gas_eur_mwh"] = ds["ttf_gas_eur_mwh"]
    x["eua_proxy_usd"] = ds["eua_proxy_usd"]

    day_key = full_idx.normalize()
    rl_by_day = ds["residual_load_forecast_mw"].groupby(day_key)
    x["rl_day_max"] = rl_by_day.transform("max")
    x["rl_day_min"] = rl_by_day.transform("min")

    if "neighbours" in groups:
        nb = ds[[c for c in ds.columns if c.startswith("rl_") and c.endswith("_mw")]]
        x["rl_fr_mw"] = ds["rl_fr_mw"]
        x["rl_neighbours_mw"] = nb.sum(axis=1, skipna=False)
        x["rl_region_mw"] = x["rl_neighbours_mw"] + ds["residual_load_forecast_mw"]
        x["rl_region_day_max"] = x["rl_region_mw"].groupby(day_key).transform("max")

    if direct:
        price = dataset["price_eur_mwh"].reindex(full_idx)
        price_d1 = price.where(~full_idx.hour.isin([22, 23]))
        x["price_lag24"] = price_d1.shift(24)
        x["price_lag168"] = price.shift(168)
        d1 = price_d1.groupby(day_key)
        for stat in ("mean", "max", "min"):
            x[f"price_d1_{stat}"] = d1.agg(stat).shift(1).reindex(day_key).to_numpy()
        return x.reindex(idx)

    x["lear_pred"] = fc_full["pred"]

    # UTC 22-23 of D-1 are local 00:00-01:00 of delivery day D (CEST; 23:00 in
    # CET) — cleared in the very auction being forecast, so unknown at the gate.
    # D-1 features only see hours 0-21 UTC; the week-old lag is safe as is.
    resid_d1 = resid.where(~full_idx.hour.isin([22, 23]))
    x["resid_lag24"] = resid_d1.shift(24)
    x["resid_lag168"] = resid.shift(168)

    daily_abs = resid_d1.abs().groupby(day_key).mean()
    prev_day_abs = daily_abs.shift(1)
    x["resid_mean_abs_d1"] = prev_day_abs.reindex(day_key).to_numpy()

    return x.reindex(idx)


def run(
    out_dir: Path,
    base_run: str,
    first_fit: str = "2020-01-01",
    dataset_path: str = "data/dataset/hourly.parquet",
    max_iter: int = 300,
    learning_rate: float = 0.05,
    features: list[str] | None = None,
    scale_target: bool = False,
    direct: bool = False,
) -> dict:
    """`features`: optional groups for build_features. `scale_target`: learn the
    target in units of fuel_cost_scale and scale back, so a pattern learned at
    cheap gas carries over to expensive gas. `direct`: the tree forecasts the
    price itself; the base run only supplies the evaluation index and actuals.
    """
    fc = load_forecast(Path(base_run))
    dataset = pd.read_parquet(dataset_path)
    x = build_features(fc, dataset, tuple(features or ()), direct)
    scale = fuel_cost_scale(dataset, x.index) if scale_target else pd.Series(1.0, index=x.index)
    base = pd.Series(0.0, index=fc.index) if direct else fc["pred"]
    resid = (fc["actual"] - base) / scale
    resid_valid = resid.notna().to_numpy()

    month_starts = pd.date_range(pd.Timestamp(first_fit, tz="UTC"), x.index.max(), freq="MS")
    parts = []
    for month in month_starts:
        train = (x.index < month) & resid_valid
        test = (x.index >= month) & (x.index < month + pd.offsets.MonthBegin(1))
        if not test.any():
            continue
        # a column with no variation in the window (e.g. EUA, NaN before 2021-10)
        # breaks HGB's binning; it carries no information there anyway
        cols = [c for c in x.columns if x.loc[train, c].nunique() > 1]
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=0,
        )
        model.fit(x.loc[train, cols], resid[train])
        parts.append(pd.Series(model.predict(x.loc[test, cols]), index=x.index[test]))
    correction = pd.concat(parts)
    correction = correction * scale.reindex(correction.index)
    print(f"{len(correction)} OOS hours corrected", flush=True)

    eval_index = correction.index
    lear_pred = fc["pred"].reindex(eval_index)
    actual = fc["actual"].reindex(eval_index)
    corrected = base.reindex(eval_index) + correction

    out = corrected.rename("lear_gbm_forecast").to_frame().assign(actual=actual)
    out.to_parquet(out_dir / "forecast.parquet")

    metrics: dict = {
        "base_run": base_run,
        "first_fit": first_fit,
        "params": {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "features": features or [],
            "scale_target": scale_target,
            "direct": direct,
        },
        "n_hours": len(eval_index),
        "MAE_lear": round(mae(actual.values, lear_pred.values), 3),
        "MAE_lear_gbm": round(mae(actual.values, corrected.values), 3),
        "by_year": {},
    }
    for year in sorted(set(eval_index.year)):
        mask = eval_index.year == year
        metrics["by_year"][int(year)] = {
            "MAE_lear": round(mae(actual[mask].values, lear_pred[mask].values), 3),
            "MAE_lear_gbm": round(mae(actual[mask].values, corrected[mask].values), 3),
        }
    return metrics
