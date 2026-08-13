"""Daily pre-auction snapshot of ENTSO-E day-ahead forecasts, as published.

The Transparency Platform serves only the latest submitted version of a
forecast series; TSOs occasionally resubmit, so the vintage that was
actually visible before the 12:00 CET auction gate is unrecoverable
later. This archives the day-ahead load and wind/solar forecasts for the
next delivery day each morning. All three workflow cron slots are
pre-gate; the snapshot is written only once both series are published,
and the first complete snapshot per delivery day wins.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

from pred_el_prices.pipeline.entsoe import fetch

SNAPSHOT_DATASETS = ("load_forecast", "wind_solar_forecast")


def archive_snapshot(
    archive_dir: Path, client: EntsoePandasClient, now: datetime | None = None
) -> Path | None:
    """Snapshot tomorrow's (Europe/Berlin) forecasts; None if done or not yet published."""
    now = now or datetime.now(UTC)
    delivery = (pd.Timestamp(now).tz_convert("Europe/Berlin") + pd.Timedelta(days=1)).normalize()
    dest = (
        archive_dir / f"entsoe-forecasts/{delivery:%Y}/entsoe-forecasts_{delivery:%Y%m%d}.parquet"
    )
    if dest.exists():
        return None

    start = delivery.tz_convert("UTC")
    end = (delivery + pd.Timedelta(days=1)).tz_convert("UTC")
    parts = []
    for dataset in SNAPSHOT_DATASETS:
        df = fetch(client, dataset, start, end)
        if df.empty:
            print(f"{dataset} for {delivery:%Y-%m-%d} not published yet; skipping snapshot")
            return None
        parts.append(df.add_prefix(f"{dataset} / "))
    snapshot = pd.concat(parts, axis=1)
    snapshot["fetched_at"] = pd.Timestamp(now)
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(dest)
    return dest
