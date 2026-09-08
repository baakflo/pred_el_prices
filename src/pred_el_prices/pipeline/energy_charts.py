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
switch), averaged here to hourly means to match the benchmark convention.
This cache only needs to cover recent gaps — ENTSO-E and SMARD own the
deep history — so refreshes fetch a short trailing window.
"""

from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.energy-charts.info/price"

DATASET = "energy_charts_prices"


def _payload_to_hourly(payload: dict) -> pd.DataFrame:
    """energy-charts price payload -> hourly-mean UTC frame, nulls dropped."""
    idx = pd.to_datetime(payload.get("unix_seconds", []), unit="s", utc=True)
    series = pd.Series(payload.get("price", []), index=idx, dtype="float64").dropna()
    if series.empty:
        return pd.DataFrame(columns=["price_eur_mwh"])
    return series.resample("1h").mean().dropna().to_frame("price_eur_mwh")


def fetch_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """DE-LU clearing prices over [start, end) UTC as hourly means."""
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
    df = _payload_to_hourly(resp.json())
    return df[(df.index >= start) & (df.index < end)]


def update_cache(cache_root: Path, start: pd.Timestamp, end: pd.Timestamp | None = None) -> int:
    """Fetch and upsert; resumes from the cache tail (refetches the last 2 days)."""
    from pred_el_prices.pipeline import cache

    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    resume = cache.last_timestamp(cache_root, DATASET)
    if resume is not None:
        start = max(start, resume - pd.Timedelta(days=2))
    df = fetch_prices(start, end)
    cache.upsert(cache_root, DATASET, df)
    return len(df)
