"""Offline tests for the energy-charts price payload conversion."""

import pandas as pd

from pred_el_prices.pipeline.energy_charts import _payload_to_hourly


def unix(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


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
