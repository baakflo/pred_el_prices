"""Tests for the quantile network (offline, synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pred_el_prices.experiments import qnn_de
from pred_el_prices.models.qnn import QUANTILES, QNNConfig, QuantileMLP, fit_predict


def _days(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    prices = 50 + rng.normal(0, 10, (n, 24))
    exog = rng.normal(40000, 3000, (n, 24, 3))
    fuels = np.column_stack([30 + rng.normal(0, 1, n), 70 + rng.normal(0, 1, n)])
    return prices, exog, fuels, days


class TestHead:
    def test_percentiles_never_cross(self):
        model = QuantileMLP(10, [16], 0.0)
        out = model(torch.randn(50, 10))
        assert out.shape == (50, 24, len(QUANTILES))
        assert (out.diff(dim=-1) > 0).all()


class TestDesign:
    def test_delivery_day_prices_do_not_enter_its_own_row(self):
        prices, exog, fuels, days = _days(30)
        x, _, rows = qnn_de.design(prices, exog, fuels, days)
        d = 20
        bumped = prices.copy()
        bumped[d] += 1000.0
        x2, _, _ = qnn_de.design(bumped, exog, fuels, days)
        row = np.where(rows == days[d])[0][0]
        np.testing.assert_array_equal(x[row], x2[row])

    def test_target_auction_hours_of_d_minus_1_do_not_leak(self):
        prices, exog, fuels, days = _days(30)
        x, _, rows = qnn_de.design(prices, exog, fuels, days)
        d = 20
        bumped_p, bumped_e = prices.copy(), exog.copy()
        bumped_p[d - 1, 22:] += 1000.0
        bumped_e[d - 1, 22:] += 1000.0
        x2, _, _ = qnn_de.design(bumped_p, bumped_e, fuels, days)
        row = np.where(rows == days[d])[0][0]
        np.testing.assert_array_equal(x[row], x2[row])

    def test_refit_never_trains_on_the_day_before_the_month(self):
        prices, exog, fuels, days = _days(60)
        x, y, rows = qnn_de.design(prices, exog, fuels, days)
        month = pd.Timestamp("2024-02-01", tz="UTC")
        seen = {}

        def spy(x_train, y_train, x_pred, n_unscaled, config, seed):
            seen["n_train"] = len(x_train)
            return np.zeros((len(x_pred), 24, len(QUANTILES)))

        orig = qnn_de.fit_predict
        qnn_de.fit_predict = spy
        try:
            qnn_de._refit(x, y, rows, month, QNNConfig(), 0)
        finally:
            qnn_de.fit_predict = orig
        assert seen["n_train"] == int((rows <= month - pd.Timedelta(days=2)).sum())
        assert rows[seen["n_train"] - 1] == month - pd.Timedelta(days=2)


class TestFit:
    def test_learns_a_shifted_median_and_sorted_output(self):
        prices, exog, fuels, days = _days(200, seed=1)
        x, y, _ = qnn_de.design(prices, exog, fuels, days)
        cfg = QNNConfig(hidden=[32], max_epochs=40, patience=10, batch_size=32)
        out = fit_predict(x[:-10], y[:-10], x[-10:], 7, cfg, seed=0)
        assert out.shape == (10, 24, len(QUANTILES))
        assert (np.diff(out, axis=-1) >= 0).all()
        assert abs(np.median(out[..., 49]) - 50) < 5
