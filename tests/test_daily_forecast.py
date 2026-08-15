"""Offline tests for the daily production forecast (synthetic log/prices)."""

import json

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import write_site_json


def _log(days, start="2026-08-01"):
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    df = pd.DataFrame({"forecast": np.linspace(50, 150, len(idx))}, index=idx)
    df["generated_utc"] = "2026-08-15T09:00:00+00:00"
    return df


def test_write_site_json_contract(tmp_path):
    log = _log(days=3)
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    # actual prices known for the first two days only (third is tomorrow)
    prices = (log["forecast"] + 5.0).iloc[:48]

    write_site_json(tmp_path, log_path, prices)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["delivery_day"] == "2026-08-03"
    assert len(latest["hours"]) == 24
    assert all(h["actual"] is None for h in latest["hours"])
    assert latest["generated_utc"] == "2026-08-15T09:00:00+00:00"

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [d["day"] for d in history["days"]] == ["2026-08-01", "2026-08-02"]
    assert all(d["mae"] == 5.0 for d in history["days"])


def test_write_site_json_fills_actuals_once_known(tmp_path):
    log = _log(days=1)
    log_path = tmp_path / "forecast_log.parquet"
    log.to_parquet(log_path)
    prices = log["forecast"] - 2.0  # all 24 actuals known

    write_site_json(tmp_path, log_path, prices)

    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert all(h["actual"] is not None for h in latest["hours"])
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["days"] == [{"day": "2026-08-01", "mae": 2.0}]
