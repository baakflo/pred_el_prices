"""Daily pre-gate production forecast for the public site.

Runs on D-1 before the 12:00 CET/CEST auction gate: refreshes the input
caches, archives today's 00Z ECMWF ENS run, generates the own pre-gate RES
forecast for the next UTC day (no public wind/solar forecast exists
pre-gate — see the 2026-08-15 plan addendum), feeds it to LEAR(364,
academic exog) via the predict-day-only substitution, and emits the site
JSON contract (latest.json, history.json) plus an append-only forecast log.

Delivery days are UTC blocks, matching the backtest convention. The load
forecast for the delivery day comes from ENTSO-E (published >= 2 h before
gate closure by law); if it is not yet available the run aborts loudly so a
later cron slot can retry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pred_el_prices.features.dataset import build_dataset
from pred_el_prices.features.ens_weather import run_features
from pred_el_prices.models.lear import forecast_day
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.capacity import hourly_capacity
from pred_el_prices.pipeline.entsoe import resample_hourly

MODEL_LABEL = "LEAR(364, academic exog) + own-RES v2"
NOTE = (
    "Generated before the 12:00 CET/CEST auction gate. Renewables input is our "
    "own forecast from the ECMWF ENS 00Z run (the official TSO forecast is only "
    "published at 18:00, after the auction)."
)
RES_TARGETS = {
    "wind_onshore_forecast_mw": "wind_onshore_capacity_mw",
    "wind_offshore_forecast_mw": "wind_offshore_capacity_mw",
    "solar_forecast_mw": "solar_capacity_mw",
}
WINDOW = 364
LAG_BURN_IN_DAYS = 7


def update_features(features_path: Path, archive_dir: Path, through_run_date) -> pd.DataFrame:
    """Append hourly ENS features for archived runs missing from the table."""
    features = pd.read_parquet(features_path)
    have = set(pd.to_datetime(features["run_date"]).dt.date)
    day = max(have) + timedelta(days=1)
    frames = [features]
    while day <= through_run_date:
        path = archive_dir / "ecmwf-ens" / f"{day:%Y}" / f"ecmwf-ens_{day:%Y%m%d}00.parquet"
        if path.exists():
            frames.append(run_features(path))
            print(f"features appended for run {day}")
        else:
            print(f"no ECMWF archive for run {day}")
        day += timedelta(days=1)
    if len(frames) > 1:
        features = pd.concat(frames)
        features.to_parquet(features_path)
    return features


def own_res_forecast(
    features: pd.DataFrame, dataset: pd.DataFrame, cache_dir: Path, delivery: pd.Timestamp
) -> pd.Series:
    """Train on all history before `delivery`, predict its 24 hours (aggregate MW)."""
    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    if not delivery_hours.isin(features.index).all():
        raise RuntimeError(f"ENS features for delivery day {delivery:%Y-%m-%d} missing")

    targets = dataset[list(RES_TARGETS)].dropna()
    train_index = features.index.intersection(targets.index)
    train_index = train_index[train_index < delivery]
    capacity = hourly_capacity(cache_dir, train_index.union(delivery_hours))

    def design(idx: pd.DatetimeIndex) -> pd.DataFrame:
        x = features.loc[idx].drop(columns=["run_date"])
        x["hour"] = idx.hour
        x["doy"] = idx.dayofyear
        return x

    total = pd.Series(0.0, index=delivery_hours)
    for target, cap_col in RES_TARGETS.items():
        cf = targets[target].loc[train_index] / capacity[cap_col].loc[train_index].values
        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(design(train_index), cf)
        cf_pred = np.clip(model.predict(design(delivery_hours)), 0.0, None)
        total += cf_pred * capacity[cap_col].loc[delivery_hours].values
    return total


def lear_forecast(
    dataset: pd.DataFrame,
    delivery: pd.Timestamp,
    load_forecast: pd.Series,
    own_res: pd.Series,
) -> pd.Series:
    """One production LEAR day: calibrate on trailing complete days, predict `delivery`."""
    cols = {
        "price_eur_mwh": dataset["price_eur_mwh"],
        "load_forecast_mw": dataset["load_forecast_mw"],
        "res_forecast_mw": dataset[
            ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
        ].sum(axis=1),
    }
    hist = pd.DataFrame(cols).dropna()
    day_sizes = hist.groupby(hist.index.normalize()).size()
    full_days = day_sizes[day_sizes == 24].index.sort_values()
    full_days = full_days[full_days < delivery]
    expected_last = delivery - pd.Timedelta(days=1)
    if full_days[-1] != expected_last:
        raise RuntimeError(
            f"dataset ends {full_days[-1]:%Y-%m-%d}, need complete {expected_last:%Y-%m-%d} "
            "(prices publish ~12:45 CET on the prior day; refresh the cache)"
        )
    window_days = full_days[-(WINDOW + LAG_BURN_IN_DAYS) :]
    hours = hist.index[pd.DatetimeIndex(hist.index.normalize()).isin(window_days)]
    frame = hist.loc[hours]

    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    target_row = pd.DataFrame(
        {
            "price_eur_mwh": np.nan,
            "load_forecast_mw": load_forecast.reindex(delivery_hours).values,
            "res_forecast_mw": own_res.reindex(delivery_hours).values,
        },
        index=delivery_hours,
    )
    if target_row[["load_forecast_mw", "res_forecast_mw"]].isna().any().any():
        raise RuntimeError(f"incomplete exog for delivery day {delivery:%Y-%m-%d}")
    frame = pd.concat([frame, target_row])

    n_days = len(window_days) + 1
    prices = frame["price_eur_mwh"].to_numpy().reshape(n_days, 24)
    exog = np.stack(
        [
            frame["load_forecast_mw"].to_numpy().reshape(n_days, 24),
            frame["res_forecast_mw"].to_numpy().reshape(n_days, 24),
        ],
        axis=2,
    )
    dow = np.array([d.dayofweek for d in [*window_days, delivery]])
    return pd.Series(forecast_day(prices, exog, dow), index=delivery_hours, name="forecast")


def write_site_json(out_dir: Path, log_path: Path, prices: pd.Series) -> None:
    """Derive latest.json and history.json from the forecast log + known prices."""
    log = pd.read_parquet(log_path)
    latest_day = log.index.normalize().max()
    hours = log.loc[log.index.normalize() == latest_day]
    actuals = prices.reindex(hours.index)
    latest = {
        "generated_utc": hours["generated_utc"].iloc[0],
        "delivery_day": f"{latest_day:%Y-%m-%d}",
        "model": MODEL_LABEL,
        "note": NOTE,
        "hours": [
            {
                "t": t.isoformat(),
                "forecast": round(float(f), 2),
                "actual": None if pd.isna(a) else round(float(a), 2),
            }
            for t, f, a in zip(hours.index, hours["forecast"], actuals, strict=True)
        ],
    }

    err = (log["forecast"] - prices.reindex(log.index)).abs()
    daily = err.groupby(err.index.normalize()).agg(["mean", "count"])
    scored = daily[(daily["count"] == 24) & daily["mean"].notna()].tail(60)
    history = {
        "days": [
            {"day": f"{day:%Y-%m-%d}", "mae": round(float(row["mean"]), 2)}
            for day, row in scored.iterrows()
        ]
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(latest, indent=1), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


def run_daily(
    cache_dir: Path,
    archive_dir: Path,
    features_path: Path,
    out_dir: Path,
    delivery_day=None,
    skip_fetch: bool = False,
) -> Path | None:
    """Produce and publish the forecast for the next UTC day. Idempotent per day."""
    now = datetime.now(UTC)
    delivery = (
        pd.Timestamp(delivery_day, tz="UTC")
        if delivery_day
        else pd.Timestamp(now.date() + timedelta(days=1), tz="UTC")
    )
    run_date = (delivery - pd.Timedelta(days=1)).date()
    log_path = out_dir / "forecast_log.parquet"

    if not skip_fetch:
        from entsoe import EntsoePandasClient

        from pred_el_prices.config import entsoe_api_key
        from pred_el_prices.pipeline.capacity import update_cache
        from pred_el_prices.pipeline.ecmwf import archive_run
        from pred_el_prices.pipeline.entsoe import backfill

        client = EntsoePandasClient(api_key=entsoe_api_key())
        backfill(
            client,
            ["day_ahead_prices", "load_forecast", "wind_solar_forecast"],
            pd.Timestamp("2015-01-01", tz="UTC"),
            delivery + pd.Timedelta(days=1),
            cache_dir,
        )
        update_cache(cache_dir)
        archive_run(run_date, archive_dir)

    features = update_features(features_path, archive_dir, run_date)
    dataset, _ = build_dataset(cache_dir)
    prices = resample_hourly(cache.load(cache_dir, "entsoe/day_ahead_prices"))["price_eur_mwh"]

    if log_path.exists() and (pd.read_parquet(log_path).index.normalize() == delivery).any():
        print(f"forecast for {delivery:%Y-%m-%d} already logged; refreshing site JSON only")
        write_site_json(out_dir, log_path, prices)
        return None

    load_fc = resample_hourly(cache.load(cache_dir, "entsoe/load_forecast"))["Forecasted Load"]
    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    if load_fc.reindex(delivery_hours).isna().any():
        raise RuntimeError(
            f"ENTSO-E load forecast for {delivery:%Y-%m-%d} not yet published; retry later"
        )

    own_res = own_res_forecast(features, dataset, cache_dir, delivery)
    forecast = lear_forecast(dataset, delivery, load_fc, own_res)

    entry = forecast.to_frame()
    entry["generated_utc"] = now.isoformat(timespec="seconds")
    if log_path.exists():
        entry = pd.concat([pd.read_parquet(log_path), entry])
    out_dir.mkdir(parents=True, exist_ok=True)
    entry.to_parquet(log_path)

    write_site_json(out_dir, log_path, prices)
    print(f"forecast for {delivery:%Y-%m-%d} written to {out_dir}")
    return out_dir / "latest.json"
