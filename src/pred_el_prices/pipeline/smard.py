"""SMARD (Bundesnetzagentur) scrapers — keyless German market data.

SMARD serves weekly JSON files of hourly values per "filter" (one series
each). Filters used here (probed live, all reach back to 2014-12-28):

- prices: 251 = DE-AT-LU (until the Oct 2018 zone split), 4169 = DE-LU
- load: 411 forecast, 410 actual (MW)
- wind/solar: 123/3791/125 day-ahead forecasts (onshore/offshore/PV),
  4067/1225/4068 the matching actuals (MW)

Timestamps are unix ms UTC. Forecast series are the TSO day-ahead forecasts —
the standard pre-auction features in the EPF literature (Lago et al. 2021).
Caveat carried from the plan: ENTSO-E/SMARD publication is D-1 evening for
some series, i.e. formally after the 12:00 gate; the benchmark convention is
to use them anyway since market participants have the underlying TSO
forecasts pre-auction. Revisit when we build the leakage audit.
"""

import time

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from pred_el_prices.pipeline import cache

BASE_URL = "https://www.smard.de/app/chart_data"

# dataset -> column -> filter ids stitched in order (multiple = zone eras)
DATASETS: dict[str, dict[str, list[int]]] = {
    "smard_day_ahead_prices": {"price_eur_mwh": [251, 4169]},
    "smard_load": {"load_forecast_mw": [411], "load_actual_mw": [410]},
    "smard_wind_solar": {
        "wind_onshore_forecast_mw": [123],
        "wind_offshore_forecast_mw": [3791],
        "solar_forecast_mw": [125],
        "wind_onshore_actual_mw": [4067],
        "wind_offshore_actual_mw": [1225],
        "solar_actual_mw": [4068],
    },
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60), reraise=True)
def _get_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _week_index(filter_id: int) -> list[int]:
    return _get_json(f"{BASE_URL}/{filter_id}/DE/index_hour.json")["timestamps"]


def _series_to_frame(series: list[list], column: str) -> pd.DataFrame:
    """SMARD [ms, value] pairs -> single-column UTC frame, nulls dropped."""
    idx = pd.to_datetime([ms for ms, _ in series], unit="ms", utc=True)
    df = pd.DataFrame({column: [v for _, v in series]}, index=idx)
    return df.dropna().sort_index()


def _weeks_overlapping(timestamps: list[int], start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    week = pd.Timedelta(days=7)
    return [
        ts
        for ts in timestamps
        if pd.Timestamp(ts, unit="ms", tz="UTC") < end
        and pd.Timestamp(ts, unit="ms", tz="UTC") + week > start
    ]


def _fetch_filter(
    filter_id: int, column: str, start: pd.Timestamp, end: pd.Timestamp, sleep_s: float
) -> pd.DataFrame:
    parts = []
    for ts in _weeks_overlapping(_week_index(filter_id), start, end):
        data = _get_json(f"{BASE_URL}/{filter_id}/DE/{filter_id}_DE_hour_{ts}.json")
        parts.append(_series_to_frame(data["series"], column))
        time.sleep(sleep_s)
    if not parts:
        return pd.DataFrame(columns=[column])
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_dataset(
    dataset: str, start: pd.Timestamp, end: pd.Timestamp, sleep_s: float = 0.2
) -> pd.DataFrame:
    """All columns of one dataset over [start, end) UTC, outer-joined on time."""
    columns = []
    for column, filter_ids in DATASETS[dataset].items():
        eras = [_fetch_filter(f, column, start, end, sleep_s) for f in filter_ids]
        merged = pd.concat(eras)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        columns.append(merged)
    df = pd.concat(columns, axis=1)
    return df[(df.index >= start) & (df.index < end)]


def update_cache(
    cache_root, dataset: str, start: pd.Timestamp, end: pd.Timestamp | None = None
) -> int:
    """Fetch and upsert; resumes from the cache tail (refetches the last week)."""
    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    resume = cache.last_timestamp(cache_root, dataset)
    if resume is not None:
        start = max(start, resume - pd.Timedelta(days=7))
    df = fetch_dataset(dataset, start, end)
    cache.upsert(cache_root, dataset, df)
    return len(df)
