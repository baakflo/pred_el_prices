"""Daily snapshot of ENTSO-E day-ahead forecasts, as first published.

The Transparency Platform serves only the latest submitted version of a
forecast series; TSOs occasionally resubmit, so the first-published
vintage is unrecoverable later. The wind/solar day-ahead forecast is only
published at 18:00 CET/CEST D-1 (EEV Par. 3) - six hours AFTER the gate -
so this is an evening capture of the vintage the backtests use, not a
pre-gate snapshot (nothing pre-gate exists publicly; see the 2026-08-15
plan addendum). The snapshot is written only once both series are
published, and the first complete snapshot per delivery day wins;
`fetched_at` records the capture time.
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
