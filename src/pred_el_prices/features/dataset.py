"""Build the hourly benchmark feature/target table from the local caches.

Leakage rules (auction gate 12:00 CET on D-1):
- TSO day-ahead forecasts (load, wind, solar) for delivery day D are the
  pre-auction weather proxies — used as-is, per the EPF benchmark convention.
- Fuel/carbon settlements of day D-2 are the freshest surely-known prices at
  the gate, so daily series are shifted by 2 days before forward-filling.
- Actuals (load, generation) stay out of this table entirely.

ENTSO-E is the primary source (15-min native, resampled to hourly means);
SMARD (hourly) patches the ENTSO-E gaps — chiefly the Sep-Dec 2018 zone-split
teething and two 2022 outage days. The target index is defined by the price
series. Fuel columns keep NaN before their sources begin (TTF 2017-10,
EUA proxy 2021-10); models must handle or drop those spans.
"""

import json
from pathlib import Path

import pandas as pd

from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import resample_hourly

# output column -> (entsoe dataset, entsoe column, smard dataset, smard column)
FORECAST_SOURCES = {
    "load_forecast_mw": (
        "entsoe/load_forecast",
        "Forecasted Load",
        "smard_load",
        "load_forecast_mw",
    ),
    "wind_onshore_forecast_mw": (
        "entsoe/wind_solar_forecast",
        "Wind Onshore",
        "smard_wind_solar",
        "wind_onshore_forecast_mw",
    ),
    "wind_offshore_forecast_mw": (
        "entsoe/wind_solar_forecast",
        "Wind Offshore",
        "smard_wind_solar",
        "wind_offshore_forecast_mw",
    ),
    "solar_forecast_mw": (
        "entsoe/wind_solar_forecast",
        "Solar",
        "smard_wind_solar",
        "solar_forecast_mw",
    ),
}

FUEL_COLUMNS = ["ttf_gas_eur_mwh", "api2_coal_usd_t", "eua_proxy_usd"]
FUEL_SETTLEMENT_LAG_DAYS = 2


def build_dataset(cache_root: Path) -> tuple[pd.DataFrame, dict]:
    """Hourly UTC table: target price + leakage-safe features, plus a build summary."""
    prices = resample_hourly(cache.load(cache_root, "entsoe/day_ahead_prices"))["price_eur_mwh"]
    prices = prices.dropna()
    # The target gets the same SMARD fallback as the forecast columns: both
    # outlets publish the identical EPEX auction result at the same moment,
    # so patching costs no leakage — during the 2026-08-30+ platform outage
    # SMARD was the only source still extending the price history.
    smard_prices = cache.load(cache_root, "smard_day_ahead_prices")
    price_patch = 0
    if not smard_prices.empty and "price_eur_mwh" in smard_prices.columns:
        merged = prices.combine_first(smard_prices["price_eur_mwh"].dropna())
        price_patch = len(merged) - len(prices)
        prices = merged
    index = prices.index
    out = pd.DataFrame({"price_eur_mwh": prices})
    summary: dict = {
        "rows": len(out),
        "first": str(index.min()),
        "last": str(index.max()),
        "price_hours_from_smard": price_patch,
    }

    patches: dict = {}
    for col, (e_ds, e_col, s_ds, s_col) in FORECAST_SOURCES.items():
        primary = resample_hourly(cache.load(cache_root, e_ds)[[e_col]])[e_col].reindex(index)
        fallback = cache.load(cache_root, s_ds)[s_col].reindex(index)
        gaps = primary.isna() & fallback.notna()
        merged = primary.where(~gaps, fallback)
        out[col] = merged
        patches[col] = {
            "patched_from_smard": int(gaps.sum()),
            "still_missing": int(merged.isna().sum()),
        }
    summary["forecast_patches"] = patches

    out["residual_load_forecast_mw"] = (
        out["load_forecast_mw"]
        - out["wind_onshore_forecast_mw"]
        - out["wind_offshore_forecast_mw"]
        - out["solar_forecast_mw"]
    )

    fuels = cache.load(cache_root, "fuels_daily")
    if not fuels.empty:
        lagged = fuels.copy()
        lagged.index = lagged.index + pd.Timedelta(days=FUEL_SETTLEMENT_LAG_DAYS)
        for col in FUEL_COLUMNS:
            if col in lagged.columns:
                out[col] = lagged[col].reindex(index, method="ffill")
        summary["fuel_lag_days"] = FUEL_SETTLEMENT_LAG_DAYS

    summary["columns"] = list(out.columns)
    summary["complete_rows"] = int(out.notna().all(axis=1).sum())
    return out, summary


def write_dataset(cache_root: Path, out_path: Path) -> dict:
    df, summary = build_dataset(cache_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
