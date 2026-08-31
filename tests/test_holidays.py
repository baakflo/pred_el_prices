"""The holiday-share feature: computus offsets, regional weights, UTC->Berlin."""

from datetime import date

import pandas as pd

from pred_el_prices.features.holidays import holiday_share, year_shares


def test_easter_derived_holidays_2026():
    # Easter Sunday 2026 is April 5
    shares = year_shares(2026)
    assert shares[date(2026, 4, 3)] == 1.0  # Good Friday
    assert shares[date(2026, 4, 6)] == 1.0  # Easter Monday
    assert shares[date(2026, 5, 14)] == 1.0  # Ascension
    assert shares[date(2026, 5, 25)] == 1.0  # Whit Monday
    assert shares[date(2026, 6, 4)] == 0.64  # Corpus Christi


def test_repentance_day_is_wednesday_before_nov_23():
    assert year_shares(2026)[date(2026, 11, 18)] == 0.05
    assert year_shares(2025)[date(2025, 11, 19)] == 0.05


def test_fixed_and_regional_dates():
    shares = year_shares(2026)
    assert shares[date(2026, 10, 3)] == 1.0
    assert shares[date(2026, 10, 31)] == 0.31
    assert shares[date(2026, 11, 1)] == 0.57


def test_hourly_lookup_uses_berlin_local_day():
    # 2026-10-02 22:00 UTC is already Oct 3 (Unity Day) in Berlin (CEST)
    idx = pd.DatetimeIndex(
        ["2026-10-02 21:00", "2026-10-02 22:00", "2026-10-03 21:00"], tz="UTC"
    )
    s = holiday_share(idx)
    assert s.tolist() == [0.0, 1.0, 1.0]


def test_day_offset_flags_the_eve():
    idx = pd.DatetimeIndex(["2026-10-02 10:00"], tz="UTC")  # Berlin Oct 2, eve of Unity Day
    assert holiday_share(idx, day_offset=1).iloc[0] == 1.0
    assert holiday_share(idx, day_offset=0).iloc[0] == 0.0


def test_normal_week_is_all_zero():
    idx = pd.date_range("2026-07-06", periods=5 * 24, freq="1h", tz="UTC")  # plain July week
    assert (holiday_share(idx) == 0).all()
