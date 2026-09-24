"""Tests for rolling PIT recalibration (synthetic data only)."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from pred_el_prices.models.recalibrate import LEVELS, pit, quantile_at, recalibrate

COLS = [f"q{i:02d}" for i in range(1, 100)]


def _normal_forecasts(n_days: int, sigma_model: float, sigma_true: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    mu = 50 + 10 * np.sin(np.arange(len(idx)) / 24)
    q = mu[:, None] + sigma_model * norm.ppf(LEVELS)[None, :]
    y = mu + rng.normal(0, sigma_true, len(idx))
    return pd.DataFrame(q, index=idx, columns=COLS).assign(actual=y)


def test_pit_and_quantile_at_round_trip():
    df = _normal_forecasts(5, 5.0, 5.0)
    q = df[COLS].to_numpy()
    u = pit(q, df["actual"].to_numpy())
    assert ((u > 0) & (u < 1)).all()
    back = np.array([quantile_at(q[i : i + 1], np.array([u[i]]))[0, 0] for i in range(len(u))])
    np.testing.assert_allclose(back, df["actual"].to_numpy(), atol=1e-6)


def test_too_narrow_bands_get_widened():
    df = _normal_forecasts(500, 5.0, 8.0)
    out = recalibrate(df, COLS, first="2024-12-01", window_days=300)
    a = out["actual"]
    before = ((df.loc[out.index, "actual"] >= df.loc[out.index, "q10"])
              & (df.loc[out.index, "actual"] <= df.loc[out.index, "q90"])).mean()
    after = ((a >= out["q10"]) & (a <= out["q90"])).mean()
    assert before < 0.65
    assert abs(after - 0.80) < 0.04


def test_calibrated_forecasts_stay_put():
    df = _normal_forecasts(500, 5.0, 5.0)
    out = recalibrate(df, COLS, first="2024-12-01", window_days=300)
    np.testing.assert_allclose(out["q50"], df.loc[out.index, "q50"], atol=0.5)


def test_only_past_days_are_used():
    df = _normal_forecasts(200, 5.0, 5.0)
    first = "2024-06-03"
    base = recalibrate(df, COLS, first=first, window_days=100)
    bumped = df.copy()
    week1 = (df.index >= pd.Timestamp(first, tz="UTC") - pd.Timedelta(days=1))
    bumped.loc[week1, "actual"] += 100.0  # the day before the week and everything after
    out = recalibrate(bumped, COLS, first=first, window_days=100)
    first_week = out.index < pd.Timestamp(first, tz="UTC") + pd.Timedelta(days=7)
    pd.testing.assert_frame_equal(base.loc[first_week, COLS], out.loc[first_week, COLS])
