"""The _late benchmark snapshot must never contain a post-gate curve."""

from datetime import UTC, datetime
from pathlib import Path

from pred_el_prices.pipeline import energyforecast


def _patch_fetch(monkeypatch):
    monkeypatch.setattr(energyforecast, "_fetch", lambda token, resolution: {"data": resolution})


def test_late_snapshot_written_pre_gate(tmp_path: Path, monkeypatch):
    _patch_fetch(monkeypatch)
    now = datetime(2026, 8, 19, 9, 45, tzinfo=UTC)  # CEST: gate is 10:00 UTC
    dest = energyforecast.archive_snapshot(tmp_path, "tok", now=now, late=True)
    assert dest is not None
    assert dest.name == "energyforecast_20260819_late.json"


def test_late_snapshot_refused_at_gate(tmp_path: Path, monkeypatch):
    _patch_fetch(monkeypatch)
    now = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)  # exactly the CEST gate
    assert energyforecast.archive_snapshot(tmp_path, "tok", now=now, late=True) is None
    assert not list(tmp_path.rglob("*.json"))


def test_late_gate_follows_winter_time(tmp_path: Path, monkeypatch):
    _patch_fetch(monkeypatch)
    now = datetime(2026, 12, 1, 10, 30, tzinfo=UTC)  # CET: gate is 11:00 UTC
    dest = energyforecast.archive_snapshot(tmp_path, "tok", now=now, late=True)
    assert dest is not None


def test_early_and_late_files_coexist(tmp_path: Path, monkeypatch):
    _patch_fetch(monkeypatch)
    now = datetime(2026, 8, 19, 9, 45, tzinfo=UTC)
    early = energyforecast.archive_snapshot(tmp_path, "tok", now=now)
    late = energyforecast.archive_snapshot(tmp_path, "tok", now=now, late=True)
    assert early is not None and late is not None
    assert early != late
    # each is idempotent on retry
    assert energyforecast.archive_snapshot(tmp_path, "tok", now=now) is None
    assert energyforecast.archive_snapshot(tmp_path, "tok", now=now, late=True) is None
