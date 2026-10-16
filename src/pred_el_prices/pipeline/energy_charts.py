"""Day-ahead price fallback from energy-charts.info (Fraunhofer ISE, CC BY 4.0).

Third outlet for the same EPEX auction, behind ENTSO-E and SMARD. Two
outlets proved not to be independent: on 2026-09-07 the Transparency
Platform went dark mid-afternoon (503, then 599, then 404 with a valid
key) AND SMARD never ingested the day's auction — the site sat
actual-less all evening and the evening edition starved on an incomplete
price day. energy-charts carries its own exchange feed and had the full
day. Same leakage argument as SMARD: identical auction result, identical
publication moment.

The API serves native-resolution points (15-min since the 2025-10 MTU
switch), averaged here to hourly means to match the benchmark convention;
the native points are cached too (DATASET_QH), as the quarter-hour actuals'
fallback: on 2026-10-16 ENTSO-E had not published the auction by the 12:15
refresh, energy-charts had, and the site showed hourly actuals without the
quarters. This cache only needs to cover recent gaps — ENTSO-E and SMARD own
the deep history — so refreshes fetch a short trailing window.
"""

from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.energy-charts.info/price"

DATASET = "energy_charts_prices"
DATASET_QH = "energy_charts_prices_qh"


def _payload_to_native(payload: dict) -> pd.Series:
    """energy-charts price payload -> native-resolution UTC series, nulls dropped."""
    idx = pd.to_datetime(payload.get("unix_seconds", []), unit="s", utc=True)
    return pd.Series(payload.get("price", []), index=idx, dtype="float64").dropna()


def _payload_to_hourly(payload: dict) -> pd.DataFrame:
    """energy-charts price payload -> hourly-mean UTC frame, nulls dropped."""
    series = _payload_to_native(payload)
    if series.empty:
        return pd.DataFrame(columns=["price_eur_mwh"])
    return series.resample("1h").mean().dropna().to_frame("price_eur_mwh")


def _fetch_payload(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    resp = requests.get(
        API_URL,
        params={
            "bzn": "DE-LU",
            "start": f"{start.tz_convert('UTC'):%Y-%m-%dT%H:%M}Z",
            "end": f"{end.tz_convert('UTC'):%Y-%m-%dT%H:%M}Z",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """DE-LU clearing prices over [start, end) UTC as hourly means."""
    df = _payload_to_hourly(_fetch_payload(start, end))
    return df[(df.index >= start) & (df.index < end)]


def update_cache(cache_root: Path, start: pd.Timestamp, end: pd.Timestamp | None = None) -> int:
    """Fetch and upsert hourly means and the native quarter-hours; resumes from the
    hourly cache tail (refetches the last 2 days). Returns the hourly row count."""
    from pred_el_prices.pipeline import cache

    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    resume = cache.last_timestamp(cache_root, DATASET)
    if resume is not None:
        start = max(start, resume - pd.Timedelta(days=2))
    payload = _fetch_payload(start, end)
    df = _payload_to_hourly(payload)
    df = df[(df.index >= start) & (df.index < end)]
    native = _payload_to_native(payload)
    qh = native[(native.index >= start) & (native.index < end) & (native.index.minute % 15 == 0)]
    cache.upsert(cache_root, DATASET, df)
    cache.upsert(cache_root, DATASET_QH, qh.to_frame("price_eur_mwh"))
    return len(df)
