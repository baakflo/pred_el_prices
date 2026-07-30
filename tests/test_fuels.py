"""Offline test for the Yahoo download normalization."""

import numpy as np
import pandas as pd

from pred_el_prices.pipeline.fuels import _normalize_download


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
