"""Offline test for the Yahoo download normalization."""

import numpy as np
import pandas as pd

from pred_el_prices.pipeline import cache, fuels
from pred_el_prices.pipeline.fuels import _normalize_download


def test_update_never_overwrites_a_settlement_with_nan(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    cache.upsert(
        tmp_path,
        fuels.DATASET,
        pd.DataFrame({"ttf_gas_eur_mwh": [30.0, 31.0, 32.0], "eua_proxy_usd": 70.0}, index=idx),
    )
    # a later fetch: day 2 ragged (TTF NaN), day 3 revised, day 4 new with EUA NaN
    fetched = pd.DataFrame(
        {"ttf_gas_eur_mwh": [np.nan, 32.5, 33.0], "eua_proxy_usd": [71.0, 72.0, np.nan]},
        index=idx[1:].append(pd.DatetimeIndex([pd.Timestamp("2024-01-04", tz="UTC")])),
    )
    monkeypatch.setattr(fuels, "fetch_daily", lambda *a, **k: fetched)
    fuels.update_cache(tmp_path, pd.Timestamp("2024-01-01", tz="UTC"))
    got = cache.load(tmp_path, fuels.DATASET)
    np.testing.assert_allclose(got["ttf_gas_eur_mwh"], [30.0, 31.0, 32.5, 33.0])
    np.testing.assert_allclose(got["eua_proxy_usd"].iloc[:3], [70.0, 71.0, 72.0])
    assert np.isnan(got["eua_proxy_usd"].iloc[3])  # nothing cached to keep


def test_normalize_multiticker_download():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    raw = pd.DataFrame(
        np.arange(9, dtype=float).reshape(3, 3),
        index=idx,
        columns=pd.MultiIndex.from_product([["Close"], ["TTF=F", "MTF=F", "CO2.L"]]),
    )
    df = _normalize_download(raw)
    assert list(df.columns) == ["ttf_gas_eur_mwh", "api2_coal_usd_t", "eua_proxy_usd"]
    assert str(df.index.tz) == "UTC"
    assert len(df) == 3


def test_milan_listing_fills_the_days_london_skips():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D")
    raw = pd.DataFrame(
        [[30.0, 80.0, 79.9], [31.0, np.nan, 81.7]],
        index=idx,
        columns=pd.MultiIndex.from_product([["Close"], ["TTF=F", "CO2.L", "CO2.MI"]]),
    )
    df = _normalize_download(raw)
    assert list(df.columns) == ["ttf_gas_eur_mwh", "eua_proxy_usd"]
    np.testing.assert_allclose(df["eua_proxy_usd"], [80.0, 81.7])  # London wins when set


def test_normalize_drops_all_nan_rows_and_unknown_tickers():
    idx = pd.date_range("2024-01-01", periods=2, freq="1D")
    raw = pd.DataFrame(
        [[1.0, np.nan], [np.nan, np.nan]],
        index=idx,
        columns=pd.MultiIndex.from_product([["Close"], ["TTF=F", "XX=F"]]),
    )
    df = _normalize_download(raw)
    assert list(df.columns) == ["ttf_gas_eur_mwh"]
    assert len(df) == 1
