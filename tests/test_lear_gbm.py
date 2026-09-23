"""Tests for the LEAR-GBM correction experiment (offline, synthetic data only)."""

import numpy as np
import pandas as pd

from pred_el_prices.eval.metrics import mae
from pred_el_prices.experiments import lear_gbm_de


def _synthetic_fc(n_days: int, start: str = "2020-01-01", seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    actual = pd.Series(50 + rng.normal(0, 5, len(idx)), index=idx)
    pred = actual + rng.normal(0, 1, len(idx))
    return pd.DataFrame({"pred": pred, "actual": actual}, index=idx)


def _synthetic_dataset(idx: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(idx)
    onshore = 5000 + rng.normal(0, 500, n)
    offshore = 2000 + rng.normal(0, 200, n)
    solar = np.maximum(0.0, 3000 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 100, n))
    load = 50000 + 5000 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 500, n)
    residual_load = load - onshore - offshore - solar
    return pd.DataFrame(
        {
            "load_forecast_mw": load,
            "wind_onshore_forecast_mw": onshore,
            "wind_offshore_forecast_mw": offshore,
            "solar_forecast_mw": solar,
            "residual_load_forecast_mw": residual_load,
            "ttf_gas_eur_mwh": 30 + rng.normal(0, 2, n),
            "eua_proxy_usd": 80 + rng.normal(0, 3, n),
            "api2_coal_usd_t": 100 + rng.normal(0, 5, n),  # dead ticker, must be ignored
        },
        index=idx,
    )


class TestBuildFeatures:
    def test_lag_columns_match_shifted_resid(self):
        fc = _synthetic_fc(20)
        dataset = _synthetic_dataset(fc.index)
        x = lear_gbm_de.build_features(fc, dataset)
        resid = fc["actual"] - fc["pred"]
        resid_d1 = resid.where(~fc.index.hour.isin([22, 23]))
        pd.testing.assert_series_equal(x["resid_lag24"], resid_d1.shift(24), check_names=False)
        pd.testing.assert_series_equal(x["resid_lag168"], resid.shift(168), check_names=False)

    def test_mean_abs_resid_d1_is_previous_calendar_day_mean(self):
        fc = _synthetic_fc(10)
        dataset = _synthetic_dataset(fc.index)
        x = lear_gbm_de.build_features(fc, dataset)
        resid = fc["actual"] - fc["pred"]
        day3 = fc.index.normalize() == fc.index.normalize()[0] + pd.Timedelta(days=3)
        day2 = fc.index.normalize() == fc.index.normalize()[0] + pd.Timedelta(days=2)
        day2_abs_mean = resid.abs()[day2 & (fc.index.hour < 22)].mean()
        assert np.allclose(x.loc[day3, "resid_mean_abs_d1"].to_numpy(), day2_abs_mean)

    def test_day_rl_max_min_constant_within_day(self):
        fc = _synthetic_fc(6)
        dataset = _synthetic_dataset(fc.index)
        x = lear_gbm_de.build_features(fc, dataset)
        for _, group in x.groupby(fc.index.normalize()):
            assert group["rl_day_max"].nunique() == 1
            assert group["rl_day_min"].nunique() == 1
            assert (group["rl_day_max"] >= group["rl_day_min"]).all()

    def test_perturbing_day_d_does_not_change_that_days_features(self):
        # only `actual` is perturbed: `pred` (the LEAR forecast) is itself a
        # legal same-day feature (lear_pred), known pre-gate. resid = actual -
        # pred is the *target*, not a feature of day D; only its lagged/prior-
        # day-aggregated forms feed into later days' features.
        fc = _synthetic_fc(15)
        dataset = _synthetic_dataset(fc.index)
        x_base = lear_gbm_de.build_features(fc, dataset)

        fc_perturbed = fc.copy()
        target_day = fc.index.normalize()[0] + pd.Timedelta(days=5)
        on_target_day = fc.index.normalize() == target_day
        fc_perturbed.loc[on_target_day, "actual"] += 1000.0
        x_perturbed = lear_gbm_de.build_features(fc_perturbed, dataset)

        pd.testing.assert_frame_equal(x_base[on_target_day], x_perturbed[on_target_day])

    def test_target_auction_hours_of_d_minus_1_do_not_leak(self):
        # UTC 22-23 of D-1 are local 00-01 of delivery day D: same auction as day D
        fc = _synthetic_fc(15)
        dataset = _synthetic_dataset(fc.index)
        x_base = lear_gbm_de.build_features(fc, dataset)
        day = fc.index.normalize()[0] + pd.Timedelta(days=5)
        late = (fc.index.normalize() == day) & fc.index.hour.isin([22, 23])
        fc_perturbed = fc.copy()
        fc_perturbed.loc[late, "actual"] += 1000.0
        x_perturbed = lear_gbm_de.build_features(fc_perturbed, dataset)
        next_day = fc.index.normalize() == day + pd.Timedelta(days=1)
        pd.testing.assert_frame_equal(x_base[next_day], x_perturbed[next_day])

    def test_perturbing_day_d_changes_later_days_lag_features(self):
        fc = _synthetic_fc(15)
        dataset = _synthetic_dataset(fc.index)
        x_base = lear_gbm_de.build_features(fc, dataset)

        fc_perturbed = fc.copy()
        target_day = fc.index.normalize()[0] + pd.Timedelta(days=5)
        on_target_day = fc.index.normalize() == target_day
        fc_perturbed.loc[on_target_day, "actual"] += 1000.0
        x_perturbed = lear_gbm_de.build_features(fc_perturbed, dataset)

        next_day = fc.index.normalize() == target_day + pd.Timedelta(days=1)
        assert not np.allclose(
            x_base.loc[next_day, "resid_lag24"].to_numpy(),
            x_perturbed.loc[next_day, "resid_lag24"].to_numpy(),
        )


class TestDirect:
    def _perturbed_features(self, hours):
        fc = _synthetic_fc(15)
        dataset = _synthetic_dataset(fc.index).assign(price_eur_mwh=fc["actual"])
        x_base = lear_gbm_de.build_features(fc, dataset, direct=True)
        day = fc.index.normalize()[0] + pd.Timedelta(days=5)
        hit = (fc.index.normalize() == day) & fc.index.hour.isin(hours)
        dataset.loc[hit, "price_eur_mwh"] += 1000.0
        x_perturbed = lear_gbm_de.build_features(fc, dataset, direct=True)
        return fc, day, x_base, x_perturbed

    def test_no_lear_columns(self):
        fc = _synthetic_fc(10)
        dataset = _synthetic_dataset(fc.index).assign(price_eur_mwh=fc["actual"])
        x = lear_gbm_de.build_features(fc, dataset, direct=True)
        assert not any(c.startswith(("lear_", "resid_")) for c in x.columns)

    def test_same_day_prices_do_not_leak(self):
        fc, day, x_base, x_perturbed = self._perturbed_features(range(24))
        on_day = fc.index.normalize() == day
        pd.testing.assert_frame_equal(x_base[on_day], x_perturbed[on_day])

    def test_target_auction_hours_of_d_minus_1_do_not_leak(self):
        fc, day, x_base, x_perturbed = self._perturbed_features([22, 23])
        next_day = fc.index.normalize() == day + pd.Timedelta(days=1)
        pd.testing.assert_frame_equal(x_base[next_day], x_perturbed[next_day])

    def test_earlier_hours_of_d_minus_1_are_used(self):
        fc, day, x_base, x_perturbed = self._perturbed_features([12])
        next_day = fc.index.normalize() == day + pd.Timedelta(days=1)
        assert (x_perturbed.loc[next_day, "price_d1_max"] > x_base.loc[next_day, "price_d1_max"]).all()


class TestRun:
    def test_corrected_mae_beats_lear(self, tmp_path):
        n_days = 200
        idx = pd.date_range("2020-01-01", periods=n_days * 24, freq="1h", tz="UTC")
        rng = np.random.default_rng(3)

        onshore = 5000 + rng.normal(0, 500, len(idx))
        offshore = 2000 + rng.normal(0, 200, len(idx))
        solar = np.maximum(
            0.0, 3000 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 100, len(idx))
        )
        load = (
            50000
            + 15000 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24)
            + rng.normal(0, 500, len(idx))
        )
        residual_load = load - onshore - offshore - solar
        dataset = pd.DataFrame(
            {
                "load_forecast_mw": load,
                "wind_onshore_forecast_mw": onshore,
                "wind_offshore_forecast_mw": offshore,
                "solar_forecast_mw": solar,
                "residual_load_forecast_mw": residual_load,
                "ttf_gas_eur_mwh": 30 + rng.normal(0, 2, len(idx)),
                "eua_proxy_usd": 80 + rng.normal(0, 3, len(idx)),
            },
            index=idx,
        )
        dataset_path = tmp_path / "hourly.parquet"
        dataset.to_parquet(dataset_path)

        # true price: rises with residual load; LEAR systematically
        # under-forecasts (misses a chunk of the residual-load-driven mark-up)
        # -- a pattern a GBM correction can learn from residual_load_forecast_mw.
        true_price = 40 + 0.001 * residual_load + rng.normal(0, 1, len(idx))
        lear_pred = 40 + 0.0004 * residual_load + rng.normal(0, 1, len(idx))
        base_dir = tmp_path / "base_run"
        base_dir.mkdir()
        pd.DataFrame({"lear_forecast": lear_pred, "actual": true_price}, index=idx).to_parquet(
            base_dir / "forecast.parquet"
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        metrics = lear_gbm_de.run(
            out_dir=out_dir,
            base_run=str(base_dir),
            first_fit="2020-03-01",
            dataset_path=str(dataset_path),
            max_iter=100,
            learning_rate=0.1,
        )

        assert metrics["MAE_lear_gbm"] < metrics["MAE_lear"]
        assert (out_dir / "forecast.parquet").exists()
        out = pd.read_parquet(out_dir / "forecast.parquet")
        assert set(out.columns) == {"lear_gbm_forecast", "actual"}
        assert out.index.min() >= pd.Timestamp("2020-03-01", tz="UTC")

        recomputed_mae = mae(out["actual"].values, out["lear_gbm_forecast"].values)
        assert round(recomputed_mae, 3) == metrics["MAE_lear_gbm"]

        dataset.assign(price_eur_mwh=true_price).to_parquet(dataset_path)
        for flags in ({"scale_target": True}, {"direct": True, "scale_target": True}):
            flag_dir = tmp_path / "_".join(flags)
            flag_dir.mkdir()
            flagged = lear_gbm_de.run(
                out_dir=flag_dir,
                base_run=str(base_dir),
                first_fit="2020-03-01",
                dataset_path=str(dataset_path),
                max_iter=100,
                learning_rate=0.1,
                **flags,
            )
            assert flagged["n_hours"] == metrics["n_hours"]
            assert flagged["MAE_lear_gbm"] < metrics["MAE_lear"]
