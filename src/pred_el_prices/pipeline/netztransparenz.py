"""Daily snapshot of the TSOs' EEG marketing forecast (netztransparenz.de).

The four German TSOs sell the feed-in-tariff share of EEG wind/solar in the
day-ahead auction themselves, so they hold this forecast before the 12:00
gate. Whether the public API serves it pre-gate is undocumented; the
workflow polls it at every slot and the first complete snapshot per delivery
day wins, with `fetched_at` recording when it first existed. Solar is ~half
of German PV; wind is only ~1-2 % of German wind (most is direct-marketed).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
DATA_URL = "https://ds.netztransparenz.de/api/v1/data/vermarktung/Vermarktungs{kind}/{start}/{end}"
KINDS = ("Solar", "Wind")


def access_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def parse(text: str) -> pd.DataFrame:
    """Quarter-hour CSV (one MW column per TSO) -> frame indexed by UTC start."""
    df = pd.read_csv(io.StringIO(text), sep=";", decimal=",")
    idx = pd.to_datetime(df["Datum"] + " " + df["von"], utc=True)
    return df.filter(like="(MW)").set_axis(idx).rename_axis("time_utc")


def fetch(token: str, kind: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = DATA_URL.format(
        kind=kind, start=f"{start:%Y-%m-%dT%H:%M:%S}", end=f"{end:%Y-%m-%dT%H:%M:%S}"
    )
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return parse(r.text)


def archive_snapshot(archive_dir: Path, token: str, now: datetime | None = None) -> Path | None:
    """Snapshot tomorrow's (Europe/Berlin) forecast; None if done or not yet complete."""
    now = now or datetime.now(UTC)
    delivery = (pd.Timestamp(now).tz_convert("Europe/Berlin") + pd.Timedelta(days=1)).normalize()
    dest = (
        archive_dir
        / f"netztransparenz-vermarktung/{delivery:%Y}/netztransparenz-vermarktung_{delivery:%Y%m%d}.parquet"
    )
    if dest.exists():
        return None

    start = delivery.tz_convert("UTC")
    end = (delivery + pd.Timedelta(days=1)).tz_convert("UTC")
    expected = pd.date_range(start, end, freq="15min", inclusive="left")
    parts = []
    for kind in KINDS:
        df = fetch(token, kind, start, end)
        df = df[(df.index >= start) & (df.index < end)]
        # a partial day counts as unpublished (DST days have 92/100 quarters)
        if not expected.isin(df.index).all():
            print(f"{kind} for {delivery:%Y-%m-%d} not published yet; skipping snapshot")
            return None
        parts.append(df.add_prefix(f"{kind} / "))
    snapshot = pd.concat(parts, axis=1)
    snapshot["fetched_at"] = pd.Timestamp(now)
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(dest)
    return dest
