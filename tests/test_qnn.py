"""Tests for the quantile network (offline, synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pred_el_prices.experiments import qnn_de
from pred_el_prices.models.qnn import (
    QUANTILES,
    QNNConfig,
    QuantileMLP,
    dist_nll,
    dist_quantiles,
    fit,
    fit_predict,
    load_bundle,
    predict,
    save_bundle,
)


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

    def test_jsu_quantiles_match_scipy(self):
        from scipy.stats import johnsonsu

        params = np.array([[[1.5, 2.0, -0.4, 1.3]]])
        got = dist_quantiles(params, "jsu")[0, 0]
        want = johnsonsu.ppf(QUANTILES, a=-0.4, b=1.3, loc=1.5, scale=2.0)
        np.testing.assert_allclose(got, want, rtol=1e-8)

    def test_jsu_nll_matches_scipy(self):
        from scipy.stats import johnsonsu

        params = torch.tensor([[[1.5, 2.0, -0.4, 1.3]] * 24], dtype=torch.float64)
        y = torch.full((1, 24), 3.0, dtype=torch.float64)
        want = -johnsonsu.logpdf(3.0, a=-0.4, b=1.3, loc=1.5, scale=2.0)
        assert abs(dist_nll(params, y, "jsu").item() - want) < 1e-10


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
            end = month + pd.offsets.MonthBegin(1)
            qnn_de._refit(x, y, rows, month, end, QNNConfig(), 0)
            n_expanding = seen["n_train"]
            qnn_de._refit(x, y, rows, month, end, QNNConfig(), 0, window_days=10)
        finally:
            qnn_de.fit_predict = orig
        assert n_expanding == int((rows <= month - pd.Timedelta(days=2)).sum())
        assert rows[n_expanding - 1] == month - pd.Timedelta(days=2)
        assert seen["n_train"] == 10

    def test_fuel_scale_divides_prices_by_that_days_cost(self):
        _, _, fuels, _ = _days(5)
        cost = qnn_de.fuel_cost(fuels)
        np.testing.assert_allclose(cost, np.maximum(20, 2 * fuels[:, 0] + 0.37 * fuels[:, 1]))


class TestSeeds:
    def test_saved_seeds_average_to_the_forecast(self, monkeypatch, tmp_path):
        prices, exog, fuels, days = _days(80)
        hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)])
        monkeypatch.setattr(qnn_de, "load_days", lambda *a: (prices, exog, fuels, days, hours))

        def fake_fit(x_train, y_train, x_pred, n_unscaled, config, seed):
            base = np.broadcast_to(QUANTILES * 100, (len(x_pred), 24, len(QUANTILES)))
            return base + 10.0 * seed

        monkeypatch.setattr(qnn_de, "fit_predict", fake_fit)
        path = tmp_path / "seeds.npz"
        qdf = qnn_de.backtest(
            "unused", "2024-01-01", [], QNNConfig(), "2024-02-15", None, "4weeks",
            None, False, 3, 1, verbose=0, seed_path=path,
        )  # fmt: skip
        saved = np.load(path)
        assert saved["q"].shape == (3, len(qdf), len(QUANTILES))
        np.testing.assert_array_equal(
            pd.DatetimeIndex(saved["index"], tz="UTC"), qdf.index.as_unit("ns")
        )
        np.testing.assert_allclose(saved["q"].mean(axis=0), qdf[qnn_de.Q_COLS].to_numpy())
        assert not np.allclose(saved["q"][0], saved["q"][2])

    def test_first_seed_shifts_the_seed_range(self, monkeypatch):
        prices, exog, fuels, days = _days(80)
        hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)])
        monkeypatch.setattr(qnn_de, "load_days", lambda *a: (prices, exog, fuels, days, hours))
        seen = set()

        def spy(x_train, y_train, x_pred, n_unscaled, config, seed):
            seen.add(seed)
            return np.zeros((len(x_pred), 24, len(QUANTILES)))

        monkeypatch.setattr(qnn_de, "fit_predict", spy)
        qnn_de.backtest(
            "unused", "2024-01-01", [], QNNConfig(), "2024-02-15", None, "4weeks",
            None, False, 3, 1, verbose=0, first_seed=4,
        )  # fmt: skip
        assert seen == {4, 5, 6}

    def test_percentiles_are_clipped_to_the_auction_price_limits(self, monkeypatch):
        prices, exog, fuels, days = _days(80)
        hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)])
        monkeypatch.setattr(qnn_de, "load_days", lambda *a: (prices, exog, fuels, days, hours))

        def wild_fit(x_train, y_train, x_pred, n_unscaled, config, seed):
            spread = np.linspace(-1e5, 1e5, len(QUANTILES))
            return np.broadcast_to(spread, (len(x_pred), 24, len(QUANTILES))).copy()

        monkeypatch.setattr(qnn_de, "fit_predict", wild_fit)
        qdf = qnn_de.backtest(
            "unused", "2024-01-01", [], QNNConfig(), "2024-02-15", None, "4weeks",
            None, True, 1, 1, verbose=0,
        )  # fmt: skip
        q = qdf[qnn_de.Q_COLS].to_numpy()
        assert q.min() == -500.0 and q.max() == 4000.0
        assert (np.diff(q, axis=1) >= 0).all()


class TestFit:
    @pytest.mark.parametrize("head", ["jsu", "normal"])
    def test_distribution_heads_learn_the_median(self, head):
        prices, exog, fuels, days = _days(200, seed=1)
        x, y, _ = qnn_de.design(prices, exog, fuels, days)
        cfg = QNNConfig(hidden=[32], max_epochs=40, patience=10, batch_size=32, head=head)
        out = fit_predict(x[:-10], y[:-10], x[-10:], 7, cfg, seed=0)
        assert out.shape == (10, 24, len(QUANTILES))
        assert (np.diff(out, axis=-1) > 0).all()
        assert abs(np.median(out[..., 49]) - 50) < 5
        # prices are N(50, 10): 10-90 span 25.6; noise inputs on 160 days widen it
        # (the quantile head gives ~56 here)
        assert 15 < np.median(out[..., 89] - out[..., 9]) < 70

    @pytest.mark.parametrize("head", ["quantile", "jsu"])
    def test_fit_save_load_predict_equals_fit_predict(self, head, tmp_path):
        prices, exog, fuels, days = _days(120, seed=2)
        x, y, _ = qnn_de.design(prices, exog, fuels, days)
        cfg = QNNConfig(hidden=[32, 32], max_epochs=15, patience=5, head=head)
        want = fit_predict(x[:-5], y[:-5], x[-5:], 7, cfg, seed=3)
        net = fit(x[:-5], y[:-5], 7, cfg, seed=3)
        save_bundle(tmp_path / "b.pt", [net], {"trained_through": "2024-04-01"})
        nets, meta = load_bundle(tmp_path / "b.pt")
        assert meta == {"trained_through": "2024-04-01"}
        np.testing.assert_array_equal(predict(nets[0], x[-5:]), want)
        with pytest.raises(ValueError):
            predict(nets[0], x[-5:, 1:])

    def test_learns_a_shifted_median_and_sorted_output(self):
        prices, exog, fuels, days = _days(200, seed=1)
        x, y, _ = qnn_de.design(prices, exog, fuels, days)
        cfg = QNNConfig(hidden=[32], max_epochs=40, patience=10, batch_size=32)
        out = fit_predict(x[:-10], y[:-10], x[-10:], 7, cfg, seed=0)
        assert out.shape == (10, 24, len(QUANTILES))
        assert (np.diff(out, axis=-1) >= 0).all()
        assert abs(np.median(out[..., 49]) - 50) < 5
