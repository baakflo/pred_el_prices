"""Tests for the LEAR reimplementation and EPF metrics (offline, synthetic)."""

import numpy as np
import pandas as pd

from pred_el_prices.eval.metrics import mae, naive_forecast, rmae_with_history, smape
from pred_el_prices.models.lear import InvariantScaler, build_xy, rolling_forecast


class TestInvariantScaler:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        x = rng.normal(50, 10, size=(200, 3))
        s = InvariantScaler().fit(x)
        assert np.allclose(s.inverse_transform(s.transform(x)), x)

    def test_columnwise(self):
        x = np.column_stack([np.full(50, 10.0), np.arange(50, dtype=float)])
        s = InvariantScaler().fit(x)
        assert s.median[0] == 10.0
        assert s.median[1] == 24.5


class TestBuildXy:
    def test_shapes_and_content(self):
        n_days, n_exog = 20, 2
        prices = np.arange(n_days * 24, dtype=float).reshape(n_days, 24)
        exog = np.stack([prices * 10, prices * 100], axis=2)
        dow = np.arange(n_days) % 7
        x, y = build_xy(prices, exog, dow)
        assert x.shape == (n_days - 7, 96 + 72 * n_exog + 7)
        assert y.shape == (n_days - 7, 24)
        # row 0 = day 7: first block is prices of day 6
        assert np.array_equal(x[0, :24], prices[6])
        # 4th price block is day 0 (lag 7)
        assert np.array_equal(x[0, 72:96], prices[0])
        # first exog block is exog[:, :, 0] at day 7 itself (lag 0)
        assert np.array_equal(x[0, 96:120], exog[7, :, 0])
        # dummy one-hot matches day-of-week
        assert x[0, -7:].sum() == 1.0
        assert x[0, -7 + dow[7]] == 1.0


class TestMetrics:
    def test_mae_smape(self):
        assert mae([1, 2, 3], [2, 2, 3]) == pytest_approx(1 / 3)
        assert smape([100, 100], [100, 100]) == 0.0

    def test_naive_forecast_rules(self):
        idx = pd.date_range("2020-01-06", periods=14 * 24, freq="1h", tz="UTC")  # starts Monday
        prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        naive = naive_forecast(prices)
        tuesday_hour = idx[24 * 8]  # second week's Tuesday 00:00
        assert naive[tuesday_hour] == prices[tuesday_hour - pd.Timedelta(days=1)]
        monday_hour = idx[24 * 7]  # second week's Monday 00:00
        assert naive[monday_hour] == prices[monday_hour - pd.Timedelta(days=7)]

    def test_rmae_perfect_forecast_is_zero(self):
        idx = pd.date_range("2020-01-06", periods=21 * 24, freq="1h", tz="UTC")
        prices = pd.Series(np.random.default_rng(0).normal(50, 10, len(idx)), index=idx)
        test = prices[prices.index >= idx[14 * 24]]
        assert rmae_with_history(prices, test, test.values) == 0.0

    def test_weekly_naive_is_pure_lag7(self):
        idx = pd.date_range("2020-01-06", periods=14 * 24, freq="1h", tz="UTC")
        prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)
        naive = naive_forecast(prices, kind="weekly")
        assert naive[idx[24 * 8]] == prices[idx[24 * 8] - pd.Timedelta(days=7)]


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


class TestRollingForecast:
    def test_runs_and_aligns(self):
        n_days = 40
        idx = pd.date_range("2020-01-01", periods=n_days * 24, freq="1h", tz="UTC")
        rng = np.random.default_rng(1)
        base = 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24)
        df = pd.DataFrame(
            {
                "price": 50 + base + rng.normal(0, 2, len(idx)),
                "load": 50000 + 1000 * base + rng.normal(0, 200, len(idx)),
                "wind": 10000 + rng.normal(0, 500, len(idx)),
            },
            index=idx,
        )
        test_start = idx[30 * 24]
        preds = rolling_forecast(
            df, "price", ["load", "wind"], test_start, calibration_window=25, progress_every=0
        )
        assert len(preds) == 10 * 24
        assert preds.index[0] == test_start
        # a sane model beats wild guesses on a strongly periodic series
        assert mae(df["price"].loc[preds.index].values, preds.values) < 5.0
        # parallel execution is bit-identical to serial (deterministic fits)
        preds_par = rolling_forecast(
            df,
            "price",
            ["load", "wind"],
            test_start,
            calibration_window=25,
            progress_every=0,
            n_jobs=2,
        )
        assert np.allclose(preds.values, preds_par.values)
