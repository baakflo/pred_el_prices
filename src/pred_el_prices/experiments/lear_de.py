"""LEAR on our own DE-LU dataset (data/dataset/hourly.parquet).

Convention note: days are UTC blocks (24 rows each; German delivery days are
local-midnight-aligned, i.e. shifted by 1-2 h — acceptable for v1, revisit
with the evaluation harness). No published numbers exist for this data; the
naive baselines in the metrics are the honest anchors.
"""

from pathlib import Path

import pandas as pd

from pred_el_prices.eval.metrics import mae, naive_forecast, rmse, smape
from pred_el_prices.models.lear import rolling_forecast

EXOG_COLS = [
    "load_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "solar_forecast_mw",
]
PRICE_COL = "price_eur_mwh"


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)[[PRICE_COL, *EXOG_COLS]].dropna()
    # trim partial UTC days at the edges (DE delivery starts 23:00 UTC, and
    # the current day is still incomplete)
    day_sizes = df.groupby(df.index.normalize()).size()
    full_days = day_sizes[day_sizes == 24].index
    df = df[pd.DatetimeIndex(df.index.normalize()).isin(full_days)]
    daily = pd.DatetimeIndex(sorted(set(df.index.normalize())))
    gaps = (daily[1:] - daily[:-1]) != pd.Timedelta(days=1)
    if gaps.any():
        raise ValueError(f"non-contiguous days in dataset: {daily[1:][gaps][:5].tolist()}")
    return df


def _slice_metrics(prices_all: pd.Series, actual: pd.Series, pred: pd.Series) -> dict:
    naive_w = naive_forecast(prices_all, "weekly").reindex(actual.index)
    naive_m = naive_forecast(prices_all, "mixed").reindex(actual.index)
    return {
        "n_hours": len(actual),
        "MAE": round(mae(actual.values, pred.values), 3),
        "rMAE_weekly": round(
            mae(actual.values, pred.values) / mae(actual.values, naive_w.values), 3
        ),
        "sMAPE": round(smape(actual.values, pred.values), 2),
        "RMSE": round(rmse(actual.values, pred.values), 3),
        "naive_weekly_MAE": round(mae(actual.values, naive_w.values), 3),
        "naive_mixed_MAE": round(mae(actual.values, naive_m.values), 3),
    }


def run(
    out_dir: Path,
    window: int = 364,
    test_start: str = "2019-01-01",
    test_end: str | None = None,
    exog: str = "extended",
    n_jobs: int = -1,
    dataset_path: str = "data/dataset/hourly.parquet",
) -> dict:
    df = _load_dataset(dataset_path)
    start = pd.Timestamp(test_start, tz="UTC")
    if test_end is not None:
        df = df[df.index < pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)]
    if exog == "extended":
        exog_cols = EXOG_COLS
    elif exog == "academic":
        # Lago-style pair: load forecast + one aggregate RES forecast
        df = df.assign(
            res_forecast_mw=df[
                ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
            ].sum(axis=1)
        )
        exog_cols = ["load_forecast_mw", "res_forecast_mw"]
    else:
        raise ValueError(f"unknown exog mode {exog!r}; use 'extended' or 'academic'")

    pred = rolling_forecast(
        df, PRICE_COL, exog_cols, start, calibration_window=window, n_jobs=n_jobs
    )
    actual = df[PRICE_COL].loc[pred.index]
    pred.to_frame().assign(actual=actual).to_parquet(out_dir / "forecast.parquet")

    prices_all = df[PRICE_COL]
    metrics = {
        "window": window,
        "test_start": test_start,
        "test_end": test_end,
        "exog": exog,
        "overall": _slice_metrics(prices_all, actual, pred),
        "by_year": {},
    }
    for year in sorted(set(actual.index.year)):
        mask = actual.index.year == year
        metrics["by_year"][int(year)] = _slice_metrics(prices_all, actual[mask], pred[mask])
    return metrics
