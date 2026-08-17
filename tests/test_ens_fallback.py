"""Offline tests for the 12Z ENS fallback path (synthetic archive parquets)."""

from datetime import date

import pandas as pd

from pred_el_prices.daily_forecast import update_features
from pred_el_prices.features.ens_weather import run_features
from pred_el_prices.pipeline.ecmwf import STEPS, _step_url

VARIABLES = [
    ("u_100m", 8.0),
    ("v_100m", 0.0),
    ("u_10m", 4.0),
    ("v_10m", 0.0),
    ("t_2m", 290.0),
]


def _write_run(archive_dir, run_day: date, run_hour: int) -> None:
    """Synthetic archived ENS run: one cell, two members, all six variables."""
    run_time = pd.Timestamp(run_day, tz="UTC") + pd.Timedelta(hours=run_hour)
    rows = []
    for member in (0, 1):
        for step in STEPS[run_hour]:
            t = run_time + pd.Timedelta(hours=step)
            values = [*VARIABLES, ("ssrd", step * 3600 * 100.0)]
            for variable, value in values:
                rows.append(
                    {
                        "cell_lat": 50.0,
                        "cell_lon": 9.0,
                        "member": member,
                        "variable": variable,
                        "valid_time": t,
                        "value": value,
                    }
                )
    df = pd.DataFrame(rows)
    df["run_time"] = run_time
    out = archive_dir / "ecmwf-ens" / f"{run_day:%Y}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"ecmwf-ens_{run_day:%Y%m%d}{run_hour:02d}.parquet", index=False)


def _run_path(archive_dir, run_day: date, run_hour: int):
    return (
        archive_dir
        / "ecmwf-ens"
        / f"{run_day:%Y}"
        / (f"ecmwf-ens_{run_day:%Y%m%d}{run_hour:02d}.parquet")
    )


def test_step_url_encodes_the_run_hour():
    url = _step_url(date(2026, 8, 15), "ifs/0p25", 33, "grib2", run_hour=12)
    assert url.endswith("/20260815/12z/ifs/0p25/enfo/20260815120000-33h-enfo-ef.grib2")
    assert list(STEPS[12]) == list(range(33, 61, 3))
    assert list(STEPS[0]) == list(range(21, 49, 3))


def test_run_features_12z_targets_the_day_after_next(tmp_path):
    """A 12Z run of D-2 must yield 24 hourly rows for delivery day D."""
    _write_run(tmp_path, date(2026, 8, 10), 12)
    features = run_features(_run_path(tmp_path, date(2026, 8, 10), 12))
    expected = pd.date_range("2026-08-12", periods=24, freq="1h", tz="UTC")
    assert list(features.index) == list(expected)
    assert (features["run_date"] == date(2026, 8, 10)).all()
    assert features["ws100_nat"].notna().all()


def _seed_table(tmp_path) -> tuple:
    """Features table primary-backed through delivery 2026-08-11."""
    _write_run(tmp_path, date(2026, 8, 10), 0)
    table = run_features(_run_path(tmp_path, date(2026, 8, 10), 0))
    features_path = tmp_path / "ens_features.parquet"
    table.to_parquet(features_path)
    return features_path, tmp_path


def test_update_features_without_fallback_leaves_the_day_missing(tmp_path):
    features_path, archive_dir = _seed_table(tmp_path)
    _write_run(archive_dir, date(2026, 8, 10), 12)  # fallback exists but is not allowed

    features = update_features(features_path, archive_dir, date(2026, 8, 11))

    days = set(features.index.normalize().date)
    assert days == {date(2026, 8, 11)}


def test_update_features_uses_the_12z_fallback_when_allowed(tmp_path):
    features_path, archive_dir = _seed_table(tmp_path)
    _write_run(archive_dir, date(2026, 8, 10), 12)  # no 00Z of 08-11 -> fallback

    features = update_features(features_path, archive_dir, date(2026, 8, 11), allow_fallback=True)

    day = features.index.normalize().date == date(2026, 8, 12)
    assert day.sum() == 24
    assert (features.loc[day, "run_date"] == date(2026, 8, 10)).all()


def test_update_features_upgrades_fallback_rows_to_the_00z_run(tmp_path):
    features_path, archive_dir = _seed_table(tmp_path)
    _write_run(archive_dir, date(2026, 8, 10), 12)
    update_features(features_path, archive_dir, date(2026, 8, 11), allow_fallback=True)

    _write_run(archive_dir, date(2026, 8, 11), 0)  # 00Z backfilled later
    features = update_features(features_path, archive_dir, date(2026, 8, 11))

    day = features.index.normalize().date == date(2026, 8, 12)
    assert day.sum() == 24  # replaced, not duplicated
    assert (features.loc[day, "run_date"] == date(2026, 8, 11)).all()
