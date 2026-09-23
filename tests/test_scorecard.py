"""Tests for the run scorecard (eval/metrics.py additions + eval/scorecard.py), synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.eval.metrics import dm_test, mae_by_hour, negative_hour_metrics
from pred_el_prices.eval.scorecard import compare, load_forecast, score


def _hourly_index(n_days: int, start: str = "2020-01-06") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_days * 24, freq="1h", tz="UTC")


class TestMaeByHour:
    def test_known_pattern(self):
        idx = _hourly_index(2)
        # error is exactly equal to the hour-of-day, repeated across both days
        actual = pd.Series(0.0, index=idx)
        pred = pd.Series(-(idx.hour.astype(float)), index=idx)
        result = mae_by_hour(actual, pred)
        assert result == {h: float(h) for h in range(24)}


class TestNegativeHourMetrics:
    def test_hand_built_case(self):
        idx = pd.date_range("2020-01-06", periods=6, freq="1h", tz="UTC")
        actual = pd.Series([-10.0, -5.0, 0.0, 5.0, -2.0, 3.0], index=idx)
        pred = pd.Series([-8.0, 1.0, 0.0, 4.0, -4.0, 3.0], index=idx)
        # neg_actual: idx 0,1,4 (3); neg_pred: idx 0,4 (2); joint neg: idx 0,4
        result = negative_hour_metrics(actual, pred)
        assert result["n_neg_actual"] == 3
        assert result["n_neg_pred"] == 2
        assert result["sign_recall"] == pytest.approx(2 / 3)
        assert result["sign_precision"] == pytest.approx(2 / 2)
        assert result["median_pred_joint_neg"] == pytest.approx(np.median([-8.0, -4.0]))
        assert result["median_actual_joint_neg"] == pytest.approx(np.median([-10.0, -2.0]))
        assert result["depth_ratio"] == pytest.approx(
            np.median([-8.0, -4.0]) / np.median([-10.0, -2.0])
        )
        # actual >= 0: idx 2,3,5
        assert result["MAE_nonneg_actual"] == pytest.approx(
            np.mean([abs(0.0 - 0.0), abs(5.0 - 4.0), abs(3.0 - 3.0)])
        )

    def test_zero_negatives_gives_nan_not_exception(self):
        idx = pd.date_range("2020-01-06", periods=4, freq="1h", tz="UTC")
        actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        pred = pd.Series([1.5, 2.5, 3.5, 4.5], index=idx)
        result = negative_hour_metrics(actual, pred)
        assert result["n_neg_actual"] == 0
        assert result["n_neg_pred"] == 0
        assert np.isnan(result["sign_recall"])
        assert np.isnan(result["sign_precision"])
        assert np.isnan(result["median_pred_joint_neg"])
        assert np.isnan(result["median_actual_joint_neg"])
        assert np.isnan(result["depth_ratio"])
        assert not np.isnan(result["MAE_nonneg_actual"])


class TestDmTest:
    def test_b_clearly_better_gives_small_p(self):
        idx = _hourly_index(30)
        rng = np.random.default_rng(0)
        actual = pd.Series(50 + rng.normal(0, 1, len(idx)), index=idx)
        pred_a = actual + 10.0  # consistently bad
        pred_b = actual + rng.normal(0, 0.1, len(idx))  # nearly perfect
        stat, p = dm_test(actual, pred_a, pred_b)
        assert stat > 0
        assert p < 0.01

    def test_identical_preds_handles_zero_variance(self):
        idx = _hourly_index(10)
        actual = pd.Series(np.random.default_rng(1).normal(50, 5, len(idx)), index=idx)
        pred = actual + 3.0
        stat, p = dm_test(actual, pred, pred)
        assert stat == 0.0
        assert p == 0.5

    def test_raises_on_partial_days(self):
        idx = _hourly_index(2)[:30]  # 1.25 days
        actual = pd.Series(0.0, index=idx)
        with pytest.raises(ValueError):
            dm_test(actual, actual, actual)


class TestLoadForecast:
    def test_loads_pred_and_actual(self, tmp_path):
        idx = _hourly_index(2)
        df = pd.DataFrame(
            {"lear_forecast": np.arange(48.0), "actual": np.arange(48.0) + 1}, index=idx
        )
        df.index.name = "time_utc"
        (tmp_path / "forecast.parquet")
        df.to_parquet(tmp_path / "forecast.parquet")
        fc = load_forecast(tmp_path)
        assert list(fc.columns) == ["pred", "actual"]
        assert (fc["pred"] == df["lear_forecast"]).all()
        assert (fc["actual"] == df["actual"]).all()

    def test_raises_on_ambiguous_columns(self, tmp_path):
        idx = _hourly_index(1)
        df = pd.DataFrame(
            {"model_a": np.arange(24.0), "model_b": np.arange(24.0), "actual": np.arange(24.0)},
            index=idx,
        )
        df.to_parquet(tmp_path / "forecast.parquet")
        with pytest.raises(ValueError):
            load_forecast(tmp_path)


class TestScore:
    def test_score_keys_and_slices(self):
        idx = _hourly_index(10)
        rng = np.random.default_rng(2)
        actual = pd.Series(50 + rng.normal(0, 5, len(idx)), index=idx)
        pred = actual + rng.normal(0, 2, len(idx))
        fc = pd.DataFrame({"pred": pred, "actual": actual})
        prices_all = actual.copy()
        result = score(fc, prices_all)
        assert result["n_hours"] == len(idx)
        for key in ("MAE", "RMSE", "rMAE_weekly", "MAE_h16_18", "MAE_h00_06"):
            assert key in result
        assert set(result["mae_by_hour"].keys()) == set(range(24))
        assert "negative" in result


class TestCompare:
    def test_compare_synthetic_frames(self):
        idx = _hourly_index(14)
        rng = np.random.default_rng(3)
        actual = pd.Series(50 + rng.normal(0, 5, len(idx)), index=idx)
        # per-day noise on the bias so daily loss differentials vary (avoids
        # the zero-variance degenerate case) while A stays clearly worse.
        pred_a = actual + 10.0 + rng.normal(0, 0.5, len(idx))
        pred_b = actual + 1.0 + rng.normal(0, 0.5, len(idx))
        fc_a = pd.DataFrame({"pred": pred_a, "actual": actual})
        fc_b = pd.DataFrame({"pred": pred_b, "actual": actual})
        result = compare(fc_a, fc_b)
        assert result["n_days"] == 14
        assert result["MAE_a"] == pytest.approx(10.0, abs=0.5)
        assert result["MAE_b"] == pytest.approx(1.0, abs=0.5)
        assert result["delta"] == pytest.approx(-9.0, abs=0.5)
        assert result["dm_b_better_than_a"]["p"] < 0.01
        assert result["dm_a_better_than_b"]["p"] > 0.99
        assert result["share_days_b_wins"] == 1.0

    def test_compare_restricts_to_common_whole_days(self):
        idx_a = _hourly_index(10)
        idx_b = _hourly_index(8, start=str(idx_a[24].date()))  # overlap from day 1
        actual_a = pd.Series(50.0, index=idx_a)
        actual_b = pd.Series(50.0, index=idx_b)
        fc_a = pd.DataFrame({"pred": actual_a + 1.0, "actual": actual_a})
        fc_b = pd.DataFrame({"pred": actual_b + 2.0, "actual": actual_b})
        result = compare(fc_a, fc_b)
        assert result["n_days"] == 8
