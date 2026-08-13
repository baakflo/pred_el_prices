from datetime import UTC, datetime

import pandas as pd

from pred_el_prices.pipeline import pegel


def _fake_measurements(timeseries: str) -> list[dict]:
    # two full days + a partial "today" that must not be written
    stamps = pd.date_range("2026-08-11 00:00", "2026-08-13 06:00", freq="6h", tz="Europe/Berlin")
    base = 100.0 if timeseries == "W" else 500.0
    return [{"timestamp": t.isoformat(), "value": base + i} for i, t in enumerate(stamps)]


def test_archive_window_writes_complete_days_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pegel, "_fetch_measurements", _fake_measurements)
    now = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)

    written = pegel.archive_window(tmp_path, now=now)

    names = sorted(p.name for p in written)
    assert names == [
        "pegel-kaub_20260810.parquet",
        "pegel-kaub_20260811.parquet",
        "pegel-kaub_20260812.parquet",
    ]
    df = pd.read_parquet(tmp_path / "pegel-kaub/2026/pegel-kaub_20260812.parquet")
    assert set(df.timeseries) == {"W", "Q"}
    assert df.timestamp.dt.tz is not None  # stored in UTC
    assert (df.timestamp.dt.date == pd.Timestamp("2026-08-12").date()).all()

    assert pegel.archive_window(tmp_path, now=now) == []  # second run: nothing new
