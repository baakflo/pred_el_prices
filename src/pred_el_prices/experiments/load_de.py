"""Surrogate for the TSO day-ahead load forecast (`load-de`).

The 2026-08-30/31 ENTSO-E outage showed the TSO load forecast is the daily
price forecast's only hard delivery-time dependency on ENTSO-E (see the
2026-08-31 plan addendum). This trains a tree to imitate that series from
inputs that survive such an outage: the D-7 load forecast (the naive copy,
demoted to a feature), ENS temperature/radiation stats from the independent
ECMWF chain, and calendar features including a population-weighted holiday
share. Target is the TSO *forecast*, not actual load — LEAR calibrated
against that series including its biases.

Evaluation is expanding-window with monthly refits, mirroring `res-de`:
each calendar month predicted by a model trained strictly on earlier data.
Baseline: the plain D-7 copy.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pred_el_prices.features.holidays import holiday_share
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import resample_hourly

TARGET = "load_forecast_mw"
WEATHER_PREFIXES = ("t2m_", "ssrd_")


def _design_matrix(features: pd.DataFrame, lag_w: pd.Series) -> pd.DataFrame:
    weather = [c for c in features.columns if c.startswith(WEATHER_PREFIXES)]
    x = features[weather].copy()
    x["load_d7"] = lag_w
    x["hour"] = features.index.hour
    x["weekday"] = features.index.dayofweek
    x["doy"] = features.index.dayofyear
    for off in (-1, 0, 1):
        x[f"holiday_{off:+d}"] = holiday_share(features.index, off)
    return x


def _metrics(err: pd.Series, mean_load: float) -> dict:
    return {
        "MAE_mw": round(float(err.mean()), 1),
        "nMAE_pct": round(100 * float(err.mean()) / mean_load, 2),
    }


def run(
    out_dir: Path,
    features_path: str = "data/dataset/ens_features.parquet",
    cache_dir: str = "data/cache",
    first_fit: str = "2024-10-01",
    res_predictions_path: str | None = None,
) -> dict:
    features = pd.read_parquet(features_path)
    load_fc = resample_hourly(cache.load(Path(cache_dir), "entsoe/load_forecast"))[
        "Forecasted Load"
    ].rename(TARGET)

    lag_w = load_fc.copy()
    lag_w.index = lag_w.index + pd.Timedelta(days=7)

    joined = features.index.intersection(load_fc.dropna().index)
    features = features.loc[joined]
    target = load_fc.loc[joined]
    x = _design_matrix(features, lag_w.reindex(joined))

    month_starts = pd.date_range(pd.Timestamp(first_fit, tz="UTC"), joined.max(), freq="MS")
    parts = []
    for month in month_starts:
        train = joined < month
        test = (joined >= month) & (joined < month + pd.offsets.MonthBegin(1))
        if not test.any():
            continue
        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(x[train], target[train])
        parts.append(pd.Series(model.predict(x[test]), index=joined[test]))
    pred = pd.concat(parts).clip(lower=0.0)
    print(f"{len(pred)} OOS hours predicted", flush=True)

    eval_index = pred.index
    actual = target.reindex(eval_index)
    d7 = lag_w.reindex(eval_index)
    mean_load = float(actual.mean())
    holiday_touched = (
        sum(holiday_share(eval_index, off) for off in (-1, 0, 1)) > 0
    )

    metrics: dict = {
        "n_hours": len(eval_index),
        "n_days": int(eval_index.normalize().nunique()),
        "mean_load_mw": round(mean_load, 1),
        "surrogate": _metrics((pred - actual).abs(), mean_load),
        "naive_d7": _metrics((d7 - actual).abs(), mean_load),
        "holiday_slice": {
            "n_hours": int(holiday_touched.sum()),
            "surrogate": _metrics((pred - actual).abs()[holiday_touched], mean_load),
            "naive_d7": _metrics((d7 - actual).abs()[holiday_touched], mean_load),
        },
    }

    out = pred.rename(f"own_{TARGET}").to_frame()
    out.to_parquet(out_dir / "predictions.parquet")

    # production-mode predict-exog for lear-de arm B: surrogate load + own RES
    if res_predictions_path is not None:
        res = pd.read_parquet(res_predictions_path)
        combined = res.join(out, how="inner")
        combined.to_parquet(out_dir / "predictions_with_res.parquet")
        metrics["combined_hours"] = len(combined)

    return metrics
