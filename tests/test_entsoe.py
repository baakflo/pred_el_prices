"""Offline tests for the ENTSO-E fetch layer and the Parquet cache."""

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.entsoe import (
    ZONE_SPLIT_UTC,
    _normalize,
    _zone_windows,
    month_ranges,
    resample_hourly,
)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class TestZoneWindows:
    def test_entirely_before_split(self):
        windows = list(_zone_windows(ts("2016-01-01"), ts("2016-02-01")))
        assert windows == [("DE_AT_LU", ts("2016-01-01"), ts("2016-02-01"))]

    def test_entirely_after_split(self):
        windows = list(_zone_windows(ts("2020-01-01"), ts("2020-02-01")))
        assert windows == [("DE_LU", ts("2020-01-01"), ts("2020-02-01"))]

    def test_spanning_split(self):
        windows = list(_zone_windows(ts("2018-09-01"), ts("2018-11-01")))
        assert windows == [
            ("DE_AT_LU", ts("2018-09-01"), ZONE_SPLIT_UTC),
            ("DE_LU", ZONE_SPLIT_UTC, ts("2018-11-01")),
        ]


class TestMonthRanges:
    def test_covers_range_without_gaps(self):
        ranges = list(month_ranges(ts("2018-11-15"), ts("2019-02-10")))
        assert ranges == [
            (ts("2018-11-15"), ts("2018-12-01")),
            (ts("2018-12-01"), ts("2019-01-01")),
            (ts("2019-01-01"), ts("2019-02-01")),
            (ts("2019-02-01"), ts("2019-02-10")),
        ]

    def test_empty_when_start_at_end(self):
        assert list(month_ranges(ts("2020-01-01"), ts("2020-01-01"))) == []


class TestNormalize:
    def test_series_becomes_price_frame_in_utc(self):
        idx = pd.date_range("2020-06-01", periods=3, freq="1h", tz="Europe/Berlin")
        result = _normalize(pd.Series([1.0, 2.0, 3.0], index=idx))
        assert list(result.columns) == ["price_eur_mwh"]
        assert str(result.index.tz) == "UTC"
        assert result.index[0] == ts("2020-05-31 22:00")

    def test_multiindex_columns_flattened(self):
        idx = pd.date_range("2020-06-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=idx,
            columns=pd.MultiIndex.from_tuples(
                [("Nuclear", "Actual Aggregated"), ("Solar", "Actual Aggregated")]
            ),
        )
        result = _normalize(df)
        assert list(result.columns) == [
            "Nuclear / Actual Aggregated",
            "Solar / Actual Aggregated",
        ]


class TestResampleHourly:
    def test_quarter_hours_average_to_hour(self):
        idx = pd.date_range("2025-06-01", periods=8, freq="15min", tz="UTC")
        hourly = resample_hourly(
            pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 10, 10, 10, 10]}, index=idx)
        )
        assert len(hourly) == 2
        assert hourly["v"].tolist() == [2.5, 10.0]

    def test_dst_spring_forward_day_has_23_hours(self):
        # Europe/Berlin 2021-03-28: clocks jump 02:00 -> 03:00
        idx = pd.date_range("2021-03-28 00:00", "2021-03-28 23:00", freq="1h", tz="Europe/Berlin")
        assert len(idx) == 23
        hourly = resample_hourly(pd.DataFrame({"v": 1.0}, index=idx.tz_convert("UTC")))
        assert len(hourly) == 23


class TestCache:
    def _frame(self, start: str, hours: int, value: float, col: str = "a") -> pd.DataFrame:
        idx = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
        return pd.DataFrame({col: value}, index=idx)

    def test_roundtrip(self, tmp_path):
        df = self._frame("2020-06-01", 24, 1.0)
        cache.upsert(tmp_path, "prices", df)
        loaded = cache.load(tmp_path, "prices")
        assert len(loaded) == 24
        assert loaded.index.name == "time_utc"
        assert str(loaded.index.tz) == "UTC"

    def test_upsert_overlap_new_wins(self, tmp_path):
        cache.upsert(tmp_path, "d", self._frame("2020-06-01", 24, 1.0))
        cache.upsert(tmp_path, "d", self._frame("2020-06-01 12:00", 24, 2.0))
        loaded = cache.load(tmp_path, "d")
        assert len(loaded) == 36
        assert loaded["a"].iloc[11] == 1.0
        assert loaded["a"].iloc[12] == 2.0

    def test_upsert_splits_across_year_files(self, tmp_path):
        cache.upsert(tmp_path, "d", self._frame("2019-12-31 20:00", 8, 1.0))
        assert (tmp_path / "d" / "2019.parquet").exists()
        assert (tmp_path / "d" / "2020.parquet").exists()
        assert len(cache.load(tmp_path, "d")) == 8

    def test_column_union_on_upsert(self, tmp_path):
        cache.upsert(tmp_path, "d", self._frame("2020-06-01", 2, 1.0, col="x"))
        cache.upsert(tmp_path, "d", self._frame("2020-06-01 02:00", 2, 2.0, col="y"))
        loaded = cache.load(tmp_path, "d")
        assert set(loaded.columns) == {"x", "y"}
        assert np.isnan(loaded["y"].iloc[0])

    def test_last_timestamp(self, tmp_path):
        assert cache.last_timestamp(tmp_path, "d") is None
        cache.upsert(tmp_path, "d", self._frame("2020-06-01", 24, 1.0))
        assert cache.last_timestamp(tmp_path, "d") == ts("2020-06-01 23:00")

    def test_naive_index_rejected(self, tmp_path):
        df = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2020-06-01", periods=1, freq="1h"))
        with pytest.raises(ValueError, match="tz-aware"):
            cache.upsert(tmp_path, "d", df)

    def test_load_slicing(self, tmp_path):
        cache.upsert(tmp_path, "d", self._frame("2020-06-01", 48, 1.0))
        sliced = cache.load(tmp_path, "d", start=ts("2020-06-01 10:00"), end=ts("2020-06-01 12:00"))
        assert len(sliced) == 2
