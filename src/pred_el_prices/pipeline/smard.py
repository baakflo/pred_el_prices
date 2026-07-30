"""SMARD (Bundesnetzagentur) day-ahead prices — keyless cross-check for ENTSO-E.

SMARD serves weekly JSON files of hourly values. Two filters cover our range:
251 = DE-AT-LU (2015 to the zone split), 4169 = DE-LU (from 2018-09-30 22:00
UTC). Timestamps are unix ms, prices EUR/MWh, hourly resolution throughout.
"""

import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from pred_el_prices.pipeline import cache

BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_DE_AT_LU = 251
FILTER_DE_LU = 4169
DATASET = "smard_day_ahead_prices"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60), reraise=True)
def _get_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _week_index(filter_id: int) -> list[int]:
    return _get_json(f"{BASE_URL}/{filter_id}/DE/index_hour.json")["timestamps"]


def _series_to_frame(series: list[list]) -> pd.DataFrame:
    """SMARD [ms, value] pairs -> UTC-indexed frame, nulls dropped."""
    idx = pd.to_datetime([ms for ms, _ in series], unit="ms", utc=True)
    df = pd.DataFrame({"price_eur_mwh": [v for _, v in series]}, index=idx)
    return df.dropna().sort_index()


def _weeks_overlapping(timestamps: list[int], start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    week = pd.Timedelta(days=7)
    return [
        ts
        for ts in timestamps
        if pd.Timestamp(ts, unit="ms", tz="UTC") < end
        and pd.Timestamp(ts, unit="ms", tz="UTC") + week > start
    ]


def fetch(start: pd.Timestamp, end: pd.Timestamp, sleep_s: float = 0.2) -> pd.DataFrame:
    """Hourly day-ahead prices over [start, end) UTC, both zone eras."""
    parts = []
    for filter_id in (FILTER_DE_AT_LU, FILTER_DE_LU):
        for ts in _weeks_overlapping(_week_index(filter_id), start, end):
            data = _get_json(f"{BASE_URL}/{filter_id}/DE/{filter_id}_DE_hour_{ts}.json")
            parts.append(_series_to_frame(data["series"]))
            time.sleep(sleep_s)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df.index >= start) & (df.index < end)]


def update_cache(cache_root, start: pd.Timestamp, end: pd.Timestamp | None = None) -> int:
    """Fetch and upsert; resumes from the cache tail (refetches the last week)."""
    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    resume = cache.last_timestamp(cache_root, DATASET)
    if resume is not None:
        start = max(start, resume - pd.Timedelta(days=7))
    df = fetch(start, end)
    cache.upsert(cache_root, DATASET, df)
    return len(df)
