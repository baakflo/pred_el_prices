from datetime import UTC, datetime

import pandas as pd

from pred_el_prices.pipeline import entsoe_snapshot


def _frame(cols):
    idx = pd.date_range("2026-08-13 22:00", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({c: range(24) for c in cols}, index=idx)


def test_snapshot_writes_once_and_skips_when_unpublished(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 8, 13, 7, 0, tzinfo=UTC)  # Berlin Aug 13 -> delivery Aug 14

    calls = []

    def fake_fetch(client, dataset, start, end):
        calls.append((dataset, start, end))
        return _frame(["v"])

    monkeypatch.setattr(entsoe_snapshot, "fetch", fake_fetch)
    dest = entsoe_snapshot.archive_snapshot(tmp_path, client=None, now=now)

    assert dest == tmp_path / "entsoe-forecasts/2026/entsoe-forecasts_20260814.parquet"
    # delivery day in UTC: Aug 13 22:00 .. Aug 14 22:00 (CEST day)
    assert calls[0][1] == pd.Timestamp("2026-08-13 22:00", tz="UTC")
    assert calls[0][2] == pd.Timestamp("2026-08-14 22:00", tz="UTC")
    df = pd.read_parquet(dest)
    assert "fetched_at" in df.columns
    assert "load_forecast / v" in df.columns and "wind_solar_forecast / v" in df.columns

    assert entsoe_snapshot.archive_snapshot(tmp_path, client=None, now=now) is None  # idempotent

    # unpublished (empty) series -> no file for the next day
    monkeypatch.setattr(
        entsoe_snapshot, "fetch", lambda client, dataset, start, end: pd.DataFrame()
    )
    later = datetime(2026, 8, 14, 7, 0, tzinfo=UTC)
    assert entsoe_snapshot.archive_snapshot(tmp_path, client=None, now=later) is None
    assert not (tmp_path / "entsoe-forecasts/2026/entsoe-forecasts_20260815.parquet").exists()
