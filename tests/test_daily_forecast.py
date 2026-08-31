"""Offline tests for the daily production forecast (synthetic log/prices)."""

import json

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import (
    backfill_history,
    lear_forecast,
    run_daily,
    write_site_json,
)
from pred_el_prices.pipeline import cache


def test_lear_forecast_heals_the_local_day_boundary():
    """The last 2 UTC hours of D-1 (next local day: unauctioned, no TSO
    forecast pre-gate) must not block the production run."""
    idx = pd.date_range("2025-07-01", periods=380 * 24, freq="1h", tz="UTC")
    hours = idx.hour.to_numpy()
    dataset = pd.DataFrame(
        {
            "price_eur_mwh": 80 + 30 * np.sin(2 * np.pi * hours / 24),
            "load_forecast_mw": 55_000 + 8_000 * np.sin(2 * np.pi * hours / 24),
            "wind_onshore_forecast_mw": 12_000.0,
            "wind_offshore_forecast_mw": 2_000.0,
            "solar_forecast_mw": np.where((hours > 5) & (hours < 20), 20_000.0, 0.0),
        },
        index=idx,
    )
    # production reality: the last 2 hours of the pre-delivery day are unknown
    dataset = dataset.iloc[:-2]
    delivery = pd.Timestamp("2026-07-16", tz="UTC")
    assert idx[-1].normalize() == delivery - pd.Timedelta(days=1)

    delivery_hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    load = pd.Series(55_000.0, index=delivery_hours)
    res = pd.Series(30_000.0, index=delivery_hours)

    forecast = lear_forecast(dataset, delivery, load, res)
    assert list(forecast.index) == list(delivery_hours)
    assert forecast.notna().all()


def _log(days, start="2026-08-01", generated="2026-07-31T09:00:00+00:00"):
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    df = pd.DataFrame({"forecast": np.linspace(50, 150, len(idx))}, index=idx)
    df["generated_utc"] = generated
    return df


def test_write_site_json_contract(tmp_path):
    # generated 2026-08-02T09:00Z for delivery 2026-08-03: before the gate
    # (12:00 CEST on D-1 = 10:00 UTC), no vintage column (legacy log = 00Z)
    log = _log(days=3, generated="2026-08-02T09:00:00+00:00")
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    # actual prices known for the first two days only (third is tomorrow)
    prices = (log["forecast"] + 5.0).iloc[:48]

    write_site_json(tmp_path, log_path, prices)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["delivery_day"] == "2026-08-03"
    assert len(latest["hours"]) == 24
    assert all(h["actual"] is None for h in latest["hours"])
    assert latest["generated_utc"] == "2026-08-02T09:00:00+00:00"
    assert latest["pre_gate"] is True
    assert latest["weather_vintage"] == "00Z"
    assert latest["note"].startswith("Generated before")

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [d["day"] for d in history["days"]] == ["2026-08-01", "2026-08-02"]
    assert all(d["mae"] == 5.0 for d in history["days"])
    # scored days carry their full hourly curve, forecast and actual
    for d in history["days"]:
        assert len(d["hours"]) == 24
        assert all(round(h["actual"] - h["forecast"], 2) == 5.0 for h in d["hours"])


def test_write_site_json_preserves_history_across_log_reseed(tmp_path):
    """Scored days already published must survive a forecast-log reseed;
    freshly scored days win over previously published values."""
    (tmp_path / "history.json").write_text(
        json.dumps(
            {"days": [{"day": "2026-07-28", "mae": 11.7}, {"day": "2026-08-16", "mae": 99.0}]}
        ),
        encoding="utf-8",
    )
    log = _log(days=1, start="2026-08-16")  # reseeded log: no pre-Aug-16 rows
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    prices = log["forecast"] - 2.0

    write_site_json(tmp_path, log_path, prices)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    # the pre-reseed day survives MAE-only (no curve recoverable); the
    # freshly scored day wins over the stale published value and gains hours
    assert [(d["day"], d["mae"]) for d in history["days"]] == [
        ("2026-07-28", 11.7),
        ("2026-08-16", 2.0),
    ]
    assert "hours" not in history["days"][0]
    assert len(history["days"][1]["hours"]) == 24


def test_write_site_json_preserves_the_post_gate_flag(tmp_path):
    """Backfilled reconstructions exist only in the published file; a
    rewrite must keep their post_gate flag or the site would present them
    as pre-gate forecasts."""
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "days": [
                    {
                        "day": "2026-07-28",
                        "mae": 11.7,
                        "post_gate": True,
                        "hours": [{"t": "2026-07-28T00:00:00+00:00", "forecast": 1.0, "actual": 2.0}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    log = _log(days=1, start="2026-08-16")
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)

    write_site_json(tmp_path, log_path, log["forecast"] - 2.0)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    recovered, fresh = history["days"]
    assert recovered["post_gate"] is True
    assert len(recovered["hours"]) == 1
    assert "post_gate" not in fresh  # the live day stays unflagged


def _run_dir(tmp_path, name, start, days, err=3.0):
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    run = tmp_path / name
    run.mkdir()
    forecast = pd.Series(np.linspace(40, 160, len(idx)), index=idx, name="lear_forecast")
    forecast.to_frame().assign(actual=forecast + err).to_parquet(run / "forecast.parquet")
    return run


def test_backfill_history_fills_curveless_days_from_backtest_runs(tmp_path):
    """MAE-only and absent days gain a post_gate curve with a consistent
    recomputed MAE; days that already carry a live curve stay untouched."""
    live_hours = [
        {"t": f"2026-08-03T{h:02d}:00:00+00:00", "forecast": 50.0, "actual": 51.0}
        for h in range(24)
    ]
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "days": [
                    {"day": "2026-08-01", "mae": 9.99},  # seeded MAE, curve lost
                    {"day": "2026-08-03", "mae": 1.0, "hours": live_hours},
                ]
            }
        ),
        encoding="utf-8",
    )
    run = _run_dir(tmp_path, "run-a", "2026-08-01", days=2)  # covers 08-01 + 08-02

    filled = backfill_history(tmp_path, [run], "2026-08-01", "2026-08-04")

    assert filled == 2  # 08-01 recovered, 08-02 added; 08-03 live, 08-04 uncovered
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [(d["day"], d.get("post_gate")) for d in history["days"]] == [
        ("2026-08-01", True),
        ("2026-08-02", True),
        ("2026-08-03", None),
    ]
    recovered = history["days"][0]
    assert recovered["mae"] == 3.0  # recomputed from the curve, seed replaced
    assert len(recovered["hours"]) == 24
    assert all(round(h["actual"] - h["forecast"], 2) == 3.0 for h in recovered["hours"])
    assert history["days"][2]["hours"] == live_hours


def test_write_site_json_publishes_the_current_day_as_provisional(tmp_path):
    """Under UTC delivery blocks the current day's last 1-2 hours clear only
    in today's auction, so the morning run must publish it flagged as
    provisional (partial actuals) instead of leaving a hole between
    yesterday's score and tomorrow's forecast."""
    log = _log(days=2)  # delivery 2026-08-01 (today) + 2026-08-02 (tomorrow)
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    # today's last 2 UTC hours belong to tomorrow's local day: unauctioned
    prices = (log["forecast"] + 4.0).iloc[:22]

    write_site_json(tmp_path, log_path, prices)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [(d["day"], d.get("partial")) for d in history["days"]] == [("2026-08-01", 22)]
    day = history["days"][0]
    assert day["mae"] == 4.0
    assert len(day["hours"]) == 24
    assert sum(h["actual"] is None for h in day["hours"]) == 2

    # the post-auction refresh sees all 24 prices and finalizes the day
    write_site_json(tmp_path, log_path, log["forecast"].iloc[:24] + 4.0)
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    day = history["days"][0]
    assert (day["day"], day["mae"]) == ("2026-08-01", 4.0)
    assert "partial" not in day
    assert all(h["actual"] is not None for h in day["hours"])


def test_write_site_json_flags_post_gate_and_fallback_vintage(tmp_path):
    """A run that missed the gate or used the 12Z weather fallback must say
    so in latest.json instead of repeating the pre-gate/00Z claim."""
    # gate for delivery 2026-08-01 is 2026-07-31 12:00 CEST = 10:00 UTC
    log = _log(days=1, generated="2026-07-31T10:01:12+00:00")
    log["weather_vintage"] = "12Z"
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)

    write_site_json(tmp_path, log_path, log["forecast"] - 2.0)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["pre_gate"] is False
    assert latest["weather_vintage"] == "12Z"
    assert "AFTER" in latest["note"]
    assert "12Z" in latest["note"]


def test_write_site_json_fills_actuals_once_known(tmp_path):
    log = _log(days=1)
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    prices = log["forecast"] - 2.0  # all 24 actuals known

    write_site_json(tmp_path, log_path, prices)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert all(h["actual"] is not None for h in latest["hours"])
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [(d["day"], d["mae"]) for d in history["days"]] == [("2026-08-01", 2.0)]
    assert len(history["days"][0]["hours"]) == 24


def test_write_site_json_evening_edition_carries_its_own_flags(tmp_path):
    """The evening edition (12Z + load surrogate, generated the night before)
    is pre-gate but must be flagged as its own vintage, not pass as the
    regular morning forecast."""
    # delivery 2026-08-02; generated 2026-07-31 21:40 UTC — evening of D-2
    log = _log(days=1, start="2026-08-02", generated="2026-07-31T21:40:00+00:00")
    log["weather_vintage"] = "12Z"
    log["evening"] = True
    log["load_surrogate"] = True
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)

    write_site_json(tmp_path, log_path, log["forecast"].iloc[:0])

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["pre_gate"] is True
    assert latest["evening"] is True
    assert latest["load_surrogate"] is True
    assert "Evening edition" in latest["note"]


def test_morning_run_replaces_the_evening_edition(tmp_path):
    """Evening and morning rows for the same delivery day coexist in the
    append-only log; the morning rows (appended last) are the standing
    forecast — site JSON and day flags follow them."""
    evening = _log(days=1, start="2026-08-02", generated="2026-07-31T21:40:00+00:00")
    evening["forecast"] = 99.0
    evening["weather_vintage"] = "12Z"
    evening["evening"] = True
    evening["load_surrogate"] = True
    morning = _log(days=1, start="2026-08-02", generated="2026-08-01T09:20:00+00:00")
    morning["weather_vintage"] = "00Z"
    morning["evening"] = False
    morning["load_surrogate"] = False
    log_path = tmp_path / "forecast_log.parquet"
    pd.concat([evening, morning]).to_parquet(log_path)
    prices = morning["forecast"] + 3.0

    write_site_json(tmp_path, log_path, prices)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert "evening" not in latest and "load_surrogate" not in latest
    assert latest["generated_utc"] == "2026-08-01T09:20:00+00:00"
    assert latest["weather_vintage"] == "00Z"
    # the standing forecast is the morning one, scored against its values
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [(d["day"], d["mae"]) for d in history["days"]] == [("2026-08-02", 3.0)]
    assert "evening" not in history["days"][0]


def test_write_site_json_flags_the_standing_evening_day_in_history(tmp_path):
    """A scored day whose standing forecast stayed the evening edition (the
    morning never replaced it) carries the flags into history so the site
    can hold it out of the headline mean."""
    log = _log(days=1, start="2026-08-02", generated="2026-07-31T21:40:00+00:00")
    log["evening"] = True
    log["load_surrogate"] = True
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)

    write_site_json(tmp_path, log_path, log["forecast"] - 1.0)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    day = history["days"][0]
    assert (day["day"], day["mae"]) == ("2026-08-02", 1.0)
    assert day["evening"] is True
    assert day["load_surrogate"] is True

    # and a log reseed must not strip the published flags
    reseed = _log(days=1, start="2026-08-20")
    reseed.to_parquet(log_path)
    write_site_json(tmp_path, log_path, reseed["forecast"] - 2.0)
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    kept = next(d for d in history["days"] if d["day"] == "2026-08-02")
    assert kept["evening"] is True and kept["load_surrogate"] is True


def test_write_site_json_flags_post_gate_days_in_history(tmp_path):
    """A recovery run generated past its own gate must carry post_gate into
    history (not just latest.json) so the site holds it out of the mean."""
    # gate for delivery 2026-08-01 is 2026-07-31 10:00 UTC; generated 12:30
    log = _log(days=1, generated="2026-07-31T12:30:00+00:00")
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)

    write_site_json(tmp_path, log_path, log["forecast"] - 2.0)

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["days"][0]["post_gate"] is True
    # a pre-gate day stays unflagged
    log2 = _log(days=1, generated="2026-07-31T09:00:00+00:00")
    log2.to_parquet(log_path)
    write_site_json(tmp_path, log_path, log2["forecast"] - 2.0)
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert "post_gate" not in history["days"][0]


def test_run_daily_refresh_only_fills_actuals_without_forecasting(tmp_path):
    """The post-auction slot rewrites the site JSON from fresh prices but
    must never generate a forecast, even when today's run is missing."""
    out_dir = tmp_path / "site"
    out_dir.mkdir()
    log = _log(days=1)  # delivery 2026-08-01 only; "tomorrow" never logged
    log.to_parquet(out_dir / "forecast_log.parquet")
    prices = (log["forecast"] - 2.0).to_frame("price_eur_mwh")
    cache.upsert(tmp_path / "cache", "entsoe/day_ahead_prices", prices)

    result = run_daily(
        cache_dir=tmp_path / "cache",
        archive_dir=tmp_path,
        features_path=tmp_path / "features.parquet",
        out_dir=out_dir,
        skip_fetch=True,
        refresh_only=True,
    )

    assert result == out_dir / "latest.json"
    latest = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["delivery_day"] == "2026-08-01"  # no new forecast appeared
    assert all(h["actual"] is not None for h in latest["hours"])


def test_run_daily_refresh_only_without_log_is_a_noop(tmp_path):
    result = run_daily(
        cache_dir=tmp_path / "cache",
        archive_dir=tmp_path,
        features_path=tmp_path / "features.parquet",
        out_dir=tmp_path / "site",
        skip_fetch=True,
        refresh_only=True,
    )
    assert result is None
    assert not (tmp_path / "site" / "latest.json").exists()
