"""Offline tests for the own-RES forecast building blocks (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.features.ens_weather import _deaccumulate_ssrd, _speed
from pred_el_prices.pipeline import cache
from pred_el_prices.pipeline.capacity import DATASET, hourly_capacity


def _frame(variable, valid_times, value_by_time, member=0, cell=(50.0, 9.0)):
    rows = []
    for t in valid_times:
        rows.append(
            {
                "cell_lat": cell[0],
                "cell_lon": cell[1],
                "member": member,
                "variable": variable,
                "valid_time": t,
                "value": value_by_time[t],
            }
        )
    return pd.DataFrame(rows)


def test_speed_is_computed_before_averaging():
    """Two members with opposite wind vectors must not cancel to zero speed."""
    t = pd.Timestamp("2026-06-15 12:00", tz="UTC")
    df = pd.concat(
        [
            _frame("u_100m", [t], {t: 10.0}, member=0),
            _frame("v_100m", [t], {t: 0.0}, member=0),
            _frame("u_100m", [t], {t: -10.0}, member=1),
            _frame("v_100m", [t], {t: 0.0}, member=1),
        ]
    )
    speed = _speed(df, "u_100m", "v_100m")
    assert speed["value"].tolist() == [10.0, 10.0]


def test_ssrd_deaccumulation():
    """Accumulated J/m^2 becomes 3-hour mean W/m^2 at the interval midpoint."""
    times = pd.date_range("2026-06-15 09:00", periods=3, freq="3h", tz="UTC")
    accumulated = {times[0]: 0.0, times[1]: 10_800 * 100.0, times[2]: 10_800 * 300.0}
    rates = _deaccumulate_ssrd(_frame("ssrd", times, accumulated))
    assert rates["value"].tolist() == [100.0, 200.0]
    expected_midpoints = [times[1] - pd.Timedelta(hours=1.5), times[2] - pd.Timedelta(hours=1.5)]
    assert rates["valid_time"].tolist() == expected_midpoints


def test_ssrd_deaccumulation_clips_negative_rates():
    times = pd.date_range("2026-06-15 09:00", periods=2, freq="3h", tz="UTC")
    rates = _deaccumulate_ssrd(_frame("ssrd", times, {times[0]: 1000.0, times[1]: 900.0}))
    assert rates["value"].tolist() == [0.0]


def test_hourly_capacity_interpolates_between_months(tmp_path):
    monthly = pd.DataFrame(
        {
            "wind_onshore_capacity_mw": [60_000.0, 61_000.0],
            "wind_offshore_capacity_mw": [9_000.0, 9_000.0],
            "solar_capacity_mw": [100_000.0, 103_000.0],
        },
        index=pd.DatetimeIndex(["2026-05-01", "2026-06-01"], tz="UTC"),
    )
    cache.upsert(tmp_path, DATASET, monthly)
    index = pd.DatetimeIndex(["2026-05-01 00:00", "2026-05-16 12:00", "2026-07-15 00:00"], tz="UTC")
    result = hourly_capacity(tmp_path, index)
    assert result.loc[index[0], "solar_capacity_mw"] == 100_000.0
    assert 101_400.0 < result.loc[index[1], "solar_capacity_mw"] < 101_600.0
    # beyond the last month point: flat extension, not extrapolation
    assert result.loc[index[2], "solar_capacity_mw"] == 103_000.0


def test_hourly_capacity_raises_without_cache(tmp_path):
    with pytest.raises(RuntimeError, match="fetch-capacity"):
        hourly_capacity(tmp_path, pd.DatetimeIndex(["2026-05-01"], tz="UTC"))


def test_lear_predict_exog_overrides_only_the_prediction_day():
    """Override on day d changes day d's forecast and no other day's."""
    from pred_el_prices.models.lear import rolling_forecast

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=40 * 24, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "price": rng.normal(80, 20, len(idx)),
            "exog_a": rng.normal(50_000, 5_000, len(idx)),
            "exog_b": rng.normal(30_000, 8_000, len(idx)),
        },
        index=idx,
    )
    test_start = pd.Timestamp("2026-02-05", tz="UTC")
    base = rolling_forecast(df, "price", ["exog_a", "exog_b"], test_start, calibration_window=28)

    override_day = pd.Timestamp("2026-02-07", tz="UTC")
    hours = pd.date_range(override_day, periods=24, freq="1h", tz="UTC")
    predict_exog = pd.DataFrame({"exog_b": df.loc[hours, "exog_b"] * 1.5}, index=hours)
    changed = rolling_forecast(
        df,
        "price",
        ["exog_a", "exog_b"],
        test_start,
        calibration_window=28,
        predict_exog=predict_exog,
    )

    day_mask = changed.index.normalize() == override_day
    assert not np.allclose(changed[day_mask], base[day_mask])
    assert np.allclose(changed[~day_mask], base[~day_mask])
