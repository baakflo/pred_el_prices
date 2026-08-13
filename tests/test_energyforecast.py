import json
from datetime import UTC, datetime

from pred_el_prices.pipeline import energyforecast


def _fake_fetch(token: str, resolution: str) -> dict:
    n = 24 if resolution == "HOURLY" else 96
    return {"forecast": {"data": [{"start": i, "end": i + 1, "price": 8.5} for i in range(n)]}}


def test_archive_snapshot_writes_raw_json_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(energyforecast, "_fetch", _fake_fetch)
    now = datetime(2026, 8, 13, 5, 45, tzinfo=UTC)

    written = energyforecast.archive_snapshot(tmp_path, "tok", now=now)

    assert written == tmp_path / "energyforecast/2026/energyforecast_20260813.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["fetched_utc"] == "2026-08-13T05:45:00+00:00"
    assert payload["market_zone"] == "DE-LU"
    assert set(payload["responses"]) == {"HOURLY", "QUARTER_HOURLY"}
    assert len(payload["responses"]["QUARTER_HOURLY"]["forecast"]["data"]) == 96

    assert energyforecast.archive_snapshot(tmp_path, "tok", now=now) is None  # second run: skip
