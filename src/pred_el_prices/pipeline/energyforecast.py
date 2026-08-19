"""Daily archiver for the energyforecast.de day-ahead price forecast (benchmark).

energyforecast.de publishes an ML forecast of the EPEX day-ahead price,
recalculated twice daily (03:10 / 15:10 UTC) with no historical access, so a
missed pull is unrecoverable. Only the morning snapshot is a clean benchmark:
after the auction (~12:45 CET) the API replaces "forecasts" for tomorrow with
the real clearing prices. One pre-auction pull per UTC day is archived so the
public site can show this benchmark alongside our own pre-auction forecast.

The raw JSON responses are stored verbatim plus fetch metadata: the API is
third-party and its schema may drift, so parsing is deferred to read time.
With fixed_cost_cent=0 and vat=0 the prices are raw market prices in ct/kWh
(x10 = EUR/MWh).

Two snapshots per day make the benchmark honest: the early one (~06:00 UTC)
may predate their morning weather refresh, so a second `_late` snapshot is
taken as close to the auction gate as the workflow crons allow (09:45 UTC
slot). The late snapshot hard-refuses to write past the 12:00 Europe/Berlin
gate — a drifted cron must not sneak a post-auction curve into a "pre-gate"
benchmark file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://www.energyforecast.de/api/v1/predictions/prices_for_ha"
MARKET_ZONE = "DE-LU"
RESOLUTIONS = ("HOURLY", "QUARTER_HOURLY")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60), reraise=True)
def _fetch(token: str, resolution: str) -> dict:
    params = {
        "token": token,
        "fixed_cost_cent": 0,
        "vat": 0,
        "resolution": resolution,
        "market_zone": MARKET_ZONE,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def archive_snapshot(
    archive_dir: Path, token: str, now: datetime | None = None, late: bool = False
) -> Path | None:
    """Write today's snapshot (both resolutions, one JSON); skip if it exists.

    Returns the path written, or None if today's file was already there, so
    the workflow's best-effort retry crons stay idempotent. With `late=True`
    the file gets a `_late` suffix and is refused at/after the day-ahead
    auction gate (12:00 Europe/Berlin).
    """
    now = now or datetime.now(UTC)
    suffix = "_late" if late else ""
    dest = archive_dir / f"energyforecast/{now:%Y}/energyforecast_{now:%Y%m%d}{suffix}.json"
    if dest.exists():
        return None
    if late:
        gate = now.astimezone(ZoneInfo("Europe/Berlin")).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        if now >= gate:
            print(f"WARN late snapshot refused: {now:%H:%M} UTC is at/past the auction gate")
            return None
    payload = {
        "fetched_utc": now.isoformat(timespec="seconds"),
        "source": BASE_URL,
        "market_zone": MARKET_ZONE,
        "responses": {resolution: _fetch(token, resolution) for resolution in RESOLUTIONS},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return dest
