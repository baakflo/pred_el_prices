"""ENTSO-E Transparency Platform downloaders for the German bidding zone.

Germany was part of the joint DE-AT-LU zone until the split on 2018-10-01
(00:00 CEST = 2018-09-30 22:00 UTC); queries spanning the split are issued
against both zone codes and concatenated. Data is cached at native resolution
(hourly, and 15-min after the 2025 MTU switch); `resample_hourly` produces the
hourly benchmark view.
"""

import time
from collections.abc import Iterator

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pred_el_prices.pipeline import cache

# DE-AT-LU -> DE-LU bidding zone split: first DE-LU delivery hour
ZONE_SPLIT_UTC = pd.Timestamp("2018-09-30 22:00", tz="UTC")

DATASETS = {
    "day_ahead_prices": "query_day_ahead_prices",
    "load_forecast": "query_load_forecast",
    "load_actual": "query_load",
    "wind_solar_forecast": "query_wind_and_solar_forecast",
    "generation": "query_generation",
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (408, 429, 500, 502, 503, 504)
    return False


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=5, max=300),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _call(client: EntsoePandasClient, method: str, zone: str, start, end):
    return getattr(client, method)(zone, start=start, end=end)


def _normalize(result: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """UTC-indexed wide DataFrame with flat string column names."""
    if isinstance(result, pd.Series):
        result = result.to_frame("price_eur_mwh")
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [" / ".join(str(p) for p in tup if p) for tup in result.columns]
    result.index = result.index.tz_convert("UTC")
    return result.sort_index()


def _zone_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if start < ZONE_SPLIT_UTC:
        yield "DE_AT_LU", start, min(end, ZONE_SPLIT_UTC)
    if end > ZONE_SPLIT_UTC:
        yield "DE_LU", max(start, ZONE_SPLIT_UTC), end


def fetch(
    client: EntsoePandasClient, dataset: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """One dataset over [start, end) UTC, zone split handled; empty frame if no data."""
    method = DATASETS[dataset]
    parts = []
    for zone, s, e in _zone_windows(start, end):
        try:
            parts.append(_normalize(_call(client, method, zone, s, e)))
        except NoMatchingDataError:
            pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).sort_index()
    return df[(df.index >= start) & (df.index < end)]


def month_ranges(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Calendar-month chunks (UTC) covering [start, end)."""
    current = start.tz_convert("UTC")
    while current < end:
        next_month = current.normalize().replace(day=1) + pd.DateOffset(months=1)
        yield current, min(next_month, end)
        current = next_month


def backfill(
    client: EntsoePandasClient,
    datasets: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_root,
    sleep_s: float = 0.5,
) -> None:
    """Fetch month by month into the cache; resumes from the last cached month."""
    for dataset in datasets:
        resume = cache.last_timestamp(cache_root, f"entsoe/{dataset}")
        ds_start = start
        if resume is not None:
            # refetch the last cached month in full: it may be partial
            ds_start = max(start, resume.normalize().replace(day=1))
        for m_start, m_end in month_ranges(ds_start, end):
            df = fetch(client, dataset, m_start, m_end)
            cache.upsert(cache_root, f"entsoe/{dataset}", df)
            print(f"{dataset} {m_start:%Y-%m}: {len(df)} rows", flush=True)
            time.sleep(sleep_s)


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly benchmark view: mean over sub-hourly values (15-min MTU post-2025).

    Mean (not sum) is correct for prices (EUR/MWh) and power (MW) alike.
    """
    return df.resample("1h").mean()
