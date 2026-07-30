"""Offline tests for the benchmark dataset builder (synthetic caches)."""

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.features.dataset import build_dataset
from pred_el_prices.pipeline import cache


@pytest.fixture
def synthetic_cache(tmp_path):
    idx = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
    cache.upsert(
        tmp_path, "entsoe/day_ahead_prices", pd.DataFrame({"price_eur_mwh": 80.0}, index=idx)
    )

    load = pd.DataFrame({"Forecasted Load": 50000.0}, index=idx)
    load.iloc[10:14] = np.nan  # gap that SMARD should patch
    cache.upsert(tmp_path, "entsoe/load_forecast", load.dropna())

    ws = pd.DataFrame(
        {"Solar": 5000.0, "Wind Offshore": 2000.0, "Wind Onshore": 10000.0}, index=idx
    )
    cache.upsert(tmp_path, "entsoe/wind_solar_forecast", ws)

    cache.upsert(tmp_path, "smard_load", pd.DataFrame({"load_forecast_mw": 49000.0}, index=idx))
    cache.upsert(
        tmp_path,
        "smard_wind_solar",
        pd.DataFrame(
            {
                "wind_onshore_forecast_mw": 9900.0,
                "wind_offshore_forecast_mw": 1900.0,
                "solar_forecast_mw": 4900.0,
            },
            index=idx,
        ),
    )

    fuel_idx = pd.date_range("2023-12-28", periods=8, freq="1D", tz="UTC")
    cache.upsert(
        tmp_path,
        "fuels_daily",
        pd.DataFrame({"ttf_gas_eur_mwh": np.arange(8, dtype=float) + 30.0}, index=fuel_idx),
    )
    return tmp_path


def test_target_defines_index_and_columns(synthetic_cache):
    df, summary = build_dataset(synthetic_cache)
    assert len(df) == 96
    assert summary["rows"] == 96
    assert "residual_load_forecast_mw" in df.columns
    assert "ttf_gas_eur_mwh" in df.columns


def test_smard_patches_entsoe_gap(synthetic_cache):
    df, summary = build_dataset(synthetic_cache)
    assert summary["forecast_patches"]["load_forecast_mw"]["patched_from_smard"] == 4
    assert summary["forecast_patches"]["load_forecast_mw"]["still_missing"] == 0
    # patched hours carry the SMARD value, the rest the ENTSO-E value
    assert df["load_forecast_mw"].iloc[10] == 49000.0
    assert df["load_forecast_mw"].iloc[9] == 50000.0


def test_residual_load_subtraction(synthetic_cache):
    df, _ = build_dataset(synthetic_cache)
    assert df["residual_load_forecast_mw"].iloc[0] == 50000.0 - 10000.0 - 2000.0 - 5000.0


def test_fuel_settlement_lagged_two_days(synthetic_cache):
    df, summary = build_dataset(synthetic_cache)
    assert summary["fuel_lag_days"] == 2
    # settlement of Dec 30 (value 32.0) is the freshest usable on Jan 1
    assert df["ttf_gas_eur_mwh"].iloc[0] == 32.0
    # Jan 2 hours use the Dec 31 settlement (33.0)
    jan2 = df.loc[df.index.normalize() == pd.Timestamp("2024-01-02", tz="UTC")]
    assert (jan2["ttf_gas_eur_mwh"] == 33.0).all()
