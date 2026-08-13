"""Daily archiver for PEGELONLINE river-gauge readings (Rhine at Kaub).

PEGELONLINE serves only a rolling ~31-day window of raw 15-min values
(no historical access), so this must run regularly. Each run writes one
Parquet per complete UTC day and backfills every day still inside the
window, so a missed run self-heals for a month. History before the
window comes from the DGJ yearbook backfill (validated daily values;
see _scratch/parse_dgj_kaub.py).

Raw 15-min values are archived (not daily aggregates) so features can
use leakage-safe intraday snapshots, e.g. the level known at 09:00 on
D-1 before the auction gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
STATION = "KAUB"
TIMESERIES = ("W", "Q")  # water level [cm], discharge [m^3/s]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60), reraise=True)
def _fetch_measurements(timeseries: str) -> list[dict]:
    url = f"{BASE_URL}/stations/{STATION}/{timeseries}/measurements.json?start=P31D"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _window_frame() -> pd.DataFrame:
    """All raw values currently served by the API, long format, UTC."""
    frames = []
    for ts in TIMESERIES:
        df = pd.DataFrame(_fetch_measurements(ts))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timeseries"] = ts
        frames.append(df[["timestamp", "timeseries", "value"]])
    return pd.concat(frames, ignore_index=True)


def archive_window(archive_dir: Path, now: datetime | None = None) -> list[Path]:
    """Write one Parquet per complete UTC day missing from the archive.

    Returns the paths written. Days already on disk are skipped, so the
    call is idempotent and any gap inside the API's ~31-day window is
    refilled automatically on the next run.
    """
    now = now or datetime.now(UTC)
    data = _window_frame()
    data = data[data.timestamp.dt.date < now.date()]
    written = []
    for day, day_rows in data.groupby(data.timestamp.dt.date):
        dest = archive_dir / f"pegel-kaub/{day:%Y}/pegel-kaub_{day:%Y%m%d}.parquet"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        day_rows.sort_values(["timeseries", "timestamp"]).reset_index(drop=True).to_parquet(dest)
        written.append(dest)
    return written
