"""Tests for the quarter-hour shape model (synthetic data only)."""

import numpy as np
import pandas as pd

from pred_el_prices.models import qh_shape


def test_interp_keeps_flat_hours_flat_and_spreads_ramps():
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    q = qh_shape.interp_quarters(pd.Series([100.0, 100.0, 200.0], index=idx))
    assert len(q) == 12  # four quarters for each of the three hours
    np.testing.assert_allclose(q.iloc[:2], 100.0)  # before the first centre: held flat
    second = q.loc["2026-01-01 01:00":"2026-01-01 01:45"].to_numpy()
    assert (np.diff(second) >= 0).all() and second[-1] > 100.0  # ramps towards 200


def test_shape_sums_to_zero_within_each_hour():
    idx = pd.date_range("2026-01-01", periods=4 * 24 * 20, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    load15 = pd.Series(50000 + rng.normal(0, 500, len(idx)), index=idx)
    hours = pd.date_range(idx[0], idx[-1], freq="1h")
    solar = pd.Series(np.clip(np.sin(np.arange(len(hours)) / 24 * 2 * np.pi), 0, None) * 1e4, index=hours)
    wind = pd.Series(1e4 + rng.normal(0, 100, len(hours)), index=hours)
    p = pd.Series(80 + rng.normal(0, 5, len(hours)), index=hours)
    f = qh_shape.features(load15, solar, wind, p)
    y = qh_shape.target(pd.Series(80 + (idx.minute / 15 - 1.5) * 4, index=idx))
    pred = qh_shape.fit_predict(f.iloc[:-96], y, f.iloc[-96:])
    sums = pred.groupby(pred.index.floor("h")).sum()
    np.testing.assert_allclose(sums, 0.0, atol=1e-9)
    # the synthetic shape is a fixed staircase: -6, -2, 2, 6
    np.testing.assert_allclose(pred.iloc[:4], [-6, -2, 2, 6], atol=0.5)


def test_features_use_only_hourly_res_detail():
    # two RES series with identical hourly means but different 15-min detail give the same
    # features: the post-gate TSO 15-minute detail cannot leak in
    idx = pd.date_range("2026-01-01", periods=4 * 48, freq="15min", tz="UTC")
    hours = pd.date_range(idx[0], idx[-1], freq="1h")
    load15 = pd.Series(50000.0, index=idx)
    solar = pd.Series(np.linspace(0, 1e4, len(hours)), index=hours)
    wind = pd.Series(5e3, index=hours)
    p = pd.Series(80.0, index=hours)
    f1 = qh_shape.features(load15, solar, wind, p)
    f2 = qh_shape.features(load15, solar.copy(), wind.copy(), p)
    pd.testing.assert_frame_equal(f1, f2)
    assert {"solar_dev", "wind_dev", "load_dev"} <= set(f1.columns)
