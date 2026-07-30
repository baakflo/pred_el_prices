"""Tests for the data-QA checks (synthetic frames, offline)."""

import pandas as pd

from pred_el_prices.reporting import qa


def utc_frame(start: str, periods: int, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"v": 1.0}, index=idx)


class TestCoverage:
    def test_full_year_slice(self):
        df = utc_frame("2020-01-01", 48)
        (row,) = qa.coverage_by_year(df)
        assert row == {
            "year": 2020,
            "hours_present": 48,
            "hours_in_span": 48,
            "coverage_pct": 100.0,
        }

    def test_gap_reduces_coverage(self):
        df = utc_frame("2020-01-01", 48).drop(pd.Timestamp("2020-01-01 10:00", tz="UTC"))
        (row,) = qa.coverage_by_year(df)
        assert row["hours_present"] == 47
        assert row["hours_in_span"] == 48


class TestMissingHours:
    def test_reports_the_gap(self):
        df = utc_frame("2020-01-01", 24).drop(pd.Timestamp("2020-01-01 05:00", tz="UTC"))
        result = qa.missing_hours(df)
        assert result["count"] == 1
        assert result["hours"] == ["2020-01-01 05:00:00+00:00"]

    def test_no_gap(self):
        assert qa.missing_hours(utc_frame("2020-01-01", 24))["count"] == 0


class TestResolutionSwitches:
    def test_hourly_to_quarter_hourly(self):
        hourly = utc_frame("2025-01-01", 3 * 24)
        quarter = utc_frame("2025-01-04", 3 * 96, freq="15min")
        switches = qa.resolution_switches(pd.concat([hourly, quarter]))
        assert len(switches) == 1
        assert switches[0]["date"] == "2025-01-04"

    def test_uniform_resolution_no_switch(self):
        assert qa.resolution_switches(utc_frame("2020-01-01", 72)) == []


class TestPriceStats:
    def test_negative_and_spike_counts(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="1h", tz="UTC")
        prices = pd.Series([-10.0, 50.0, 600.0, 30.0], index=idx)
        (row,) = qa.price_stats_by_year(prices)
        assert row["n"] == 4
        assert row["n_negative"] == 1
        assert row["n_above_500"] == 1
        assert row["min"] == -10.0
        assert row["max"] == 600.0


class TestDstDayCheck:
    def test_spring_and_fall_days_flagged_with_correct_counts(self):
        idx = pd.date_range(
            "2021-01-01 00:00",
            "2021-12-31 23:00",
            freq="1h",
            tz="Europe/Berlin",
        ).tz_convert("UTC")
        df = pd.DataFrame({"v": 1.0}, index=idx)
        days = qa.dst_day_check(df, years=[2021])
        assert {d["date"]: d["rows"] for d in days} == {
            "2021-03-28": 23,
            "2021-10-31": 25,
        }
