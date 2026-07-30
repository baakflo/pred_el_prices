"""Offline tests for SMARD payload parsing and week selection."""

import pandas as pd

from pred_el_prices.pipeline.smard import _series_to_frame, _weeks_overlapping


def ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


class TestSeriesToFrame:
    def test_parses_and_drops_nulls(self):
        series = [[ms("2020-01-01 00:00"), 30.5], [ms("2020-01-01 01:00"), None]]
        df = _series_to_frame(series, "price_eur_mwh")
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2020-01-01 00:00", tz="UTC")
        assert df["price_eur_mwh"].iloc[0] == 30.5


class TestWeeksOverlapping:
    def test_selects_only_overlapping_weeks(self):
        weeks = [ms("2020-01-06"), ms("2020-01-13"), ms("2020-01-20")]
        start, end = pd.Timestamp("2020-01-14", tz="UTC"), pd.Timestamp("2020-01-16", tz="UTC")
        assert _weeks_overlapping(weeks, start, end) == [ms("2020-01-13")]

    def test_boundary_week_included_when_range_touches_it(self):
        weeks = [ms("2020-01-06"), ms("2020-01-13")]
        start, end = (
            pd.Timestamp("2020-01-12 23:00", tz="UTC"),
            pd.Timestamp("2020-01-13 01:00", tz="UTC"),
        )
        assert _weeks_overlapping(weeks, start, end) == [ms("2020-01-06"), ms("2020-01-13")]
