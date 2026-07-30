"""Daily fuel/carbon price series via Yahoo Finance proxies.

Coverage gaps (probed 2026-07): TTF=F starts 2017-10; MTF=F (coal) went stale
2025-12 (continuous contract stopped updating); CO2.L (EUA proxy, USD) starts
2021-10. Pre-2021 EUA levels need stitching from Ember's free carbon data —
tracked as a follow-up. Fuel features are Phase 2, so partial history is
acceptable for now.

Leakage rule: the settlement of day D is only knowable after D's market close,
so features for the auction at 12:00 CET on D-1 may use settlements up to D-2.
That shift happens at feature-build time; the cache stores raw settlements.
"""

import pandas as pd
import yfinance as yf

from pred_el_prices.pipeline import cache

DATASET = "fuels_daily"

TICKERS = {
    "TTF=F": "ttf_gas_eur_mwh",
    "MTF=F": "api2_coal_usd_t",
    "CO2.L": "eua_proxy_usd",
}


def _normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    """Close prices only, friendly column names, UTC-midnight index."""
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    df = close.rename(columns=TICKERS)
    df = df[[c for c in TICKERS.values() if c in df.columns]]
    df.columns.name = None
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    return df.dropna(how="all").sort_index()


def fetch_daily(start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    raw = yf.download(
        list(TICKERS),
        start=start.tz_convert(None),
        end=end.tz_convert(None) if end is not None else None,
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    return _normalize_download(raw)


def update_cache(cache_root, start: pd.Timestamp, end: pd.Timestamp | None = None) -> int:
    """Fetch and upsert; returns number of rows fetched. Resumes from the cache tail."""
    resume = cache.last_timestamp(cache_root, DATASET)
    if resume is not None:
        start = max(start, resume - pd.Timedelta(days=7))
    df = fetch_daily(start, end)
    cache.upsert(cache_root, DATASET, df)
    return len(df)
