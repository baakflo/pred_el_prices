"""Offline tests for the energy-charts price payload conversion."""

import pandas as pd

from pred_el_prices.pipeline import cache, energy_charts
from pred_el_prices.pipeline.energy_charts import _payload_to_hourly
from pred_el_prices.production.site import quarter_prices


def unix(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def _qh_payload(day: str, n: int = 8, base: float = 100.0) -> dict:
    idx = pd.date_range(day, periods=n, freq="15min", tz="UTC")
    return {"unix_seconds": [int(t.timestamp()) for t in idx], "price": [base + i for i in range(n)]}


def test_update_cache_keeps_the_native_quarter_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(energy_charts, "_fetch_payload", lambda s, e: _qh_payload("2026-10-17"))
    n = energy_charts.update_cache(
        tmp_path, pd.Timestamp("2026-10-17", tz="UTC"), pd.Timestamp("2026-10-18", tz="UTC")
    )
    assert n == 2  # hourly rows
    qh = cache.load(tmp_path, energy_charts.DATASET_QH)["price_eur_mwh"]
    assert len(qh) == 8
    assert qh.iloc[5] == 105.0
    hourly = cache.load(tmp_path, energy_charts.DATASET)["price_eur_mwh"]
    assert hourly.iloc[0] == 101.5  # mean of 100..103


def test_quarter_prices_fill_entsoe_gaps_from_energy_charts(tmp_path):
    # 2026-10-16: ENTSO-E had not published D+1 past the 12:15 refresh; energy-charts had
    idx = pd.date_range("2026-10-16", periods=8, freq="15min", tz="UTC")
    cache.upsert(tmp_path, "entsoe/day_ahead_prices", pd.DataFrame({"price_eur_mwh": 50.0}, idx))
    ec = pd.date_range("2026-10-16 01:00", periods=8, freq="15min", tz="UTC")
    cache.upsert(tmp_path, energy_charts.DATASET_QH, pd.DataFrame({"price_eur_mwh": 60.0}, ec))
    p = quarter_prices(tmp_path)
    assert len(p) == 12  # 00:00-01:45 ENTSO-E, 02:00-02:45 energy-charts
    assert (p.loc[:"2026-10-16 01:45"] == 50.0).all()  # ENTSO-E wins on overlap
    assert (p.loc["2026-10-16 02:00":] == 60.0).all()


class TestPayloadToHourly:
    def test_quarter_hours_average_to_hourly_means(self):
        payload = {
            "unix_seconds": [
                unix("2026-09-08 00:00"),
                unix("2026-09-08 00:15"),
                unix("2026-09-08 00:30"),
                unix("2026-09-08 00:45"),
                unix("2026-09-08 01:00"),
            ],
            "price": [100.0, 110.0, 90.0, 100.0, 50.0],
        }
        df = _payload_to_hourly(payload)
        assert len(df) == 2
        assert df["price_eur_mwh"].iloc[0] == 100.0
        assert df["price_eur_mwh"].iloc[1] == 50.0
        assert df.index[0] == pd.Timestamp("2026-09-08 00:00", tz="UTC")

    def test_nulls_dropped(self):
        payload = {
            "unix_seconds": [unix("2026-09-08 00:00"), unix("2026-09-08 01:00")],
            "price": [None, 42.0],
        }
        df = _payload_to_hourly(payload)
        assert len(df) == 1
        assert df["price_eur_mwh"].iloc[0] == 42.0

    def test_empty_payload_yields_empty_frame(self):
        df = _payload_to_hourly({"unix_seconds": [], "price": []})
        assert df.empty
        assert list(df.columns) == ["price_eur_mwh"]
