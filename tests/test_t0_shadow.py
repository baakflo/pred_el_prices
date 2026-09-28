"""t0 shadow forecast: gate-safe covariates, fail-soft step, opt-in wiring (fake model only)."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pred_el_prices import daily_forecast
from pred_el_prices.pipeline import cache
from pred_el_prices.production import t0_shadow
from pred_el_prices.production.t0_shadow import LEVELS, Q_NAMES, RES_COLS

TSO_POST_GATE = 1e9  # TSO RES of the horizon: published 18:00 D-1, after the gate


def _dataset(delivery: pd.Timestamp, context: int = 48) -> pd.DataFrame:
    """TSO load 1000 and TSO RES 3 x 100; hours the gate hides carry sentinels."""
    idx = pd.date_range(
        delivery - pd.Timedelta(hours=context + 48), periods=context + 120, freq="1h"
    )
    ds = pd.DataFrame({"load_forecast_mw": 1000.0}, index=idx)
    ds[RES_COLS] = 100.0
    ds.loc[idx >= t0_shadow.origin_of(delivery), RES_COLS] = TSO_POST_GATE
    # TSO load of local day D+1 is published only on D
    local_next = idx.tz_convert("Europe/Berlin").date == (delivery + pd.Timedelta(days=1)).date()
    ds.loc[local_next, "load_forecast_mw"] = TSO_POST_GATE
    return ds


def _day(delivery, value):
    return pd.Series(value, index=pd.date_range(delivery, periods=24, freq="1h"))


class _Tensor:  # the torch calls forecast_day makes, on a numpy array
    def __init__(self, a):
        self.a = a

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeT0:
    def __init__(self):
        self.calls = []

    def predict(self, ctx, horizon, quantile_levels, future_covariates):
        self.calls.append((ctx, future_covariates))
        q = np.tile(np.asarray(quantile_levels, dtype=np.float32) * 100, (len(ctx), horizon, 1))
        return SimpleNamespace(quantiles=_Tensor(q))


@pytest.mark.parametrize("day", ["2026-07-15", "2026-01-15"])  # CEST (origin 22Z), CET (23Z)
def test_horizon_uses_own_res_never_tso_res(day):
    delivery = pd.Timestamp(day, tz="UTC")
    ds = _dataset(delivery)
    prices = pd.Series(50.0, index=ds.index)
    res_prev = _day(delivery - pd.Timedelta(days=1), 70.0)
    ctx, cov, hours, keep = t0_shadow.build_inputs(
        delivery, prices, ds, _day(delivery, 2000.0), _day(delivery, 50.0), res_prev, context=48
    )
    assert len(ctx) == 48 and cov.shape == (2, 48 + t0_shadow.HORIZON)
    assert (hours == pd.date_range(delivery, periods=24, freq="1h")).all() and keep.sum() == 24
    rl = cov[0]
    assert (np.abs(rl) < 1e6).all()  # no post-gate TSO value anywhere
    np.testing.assert_allclose(rl[:48], 1000.0 - 300.0)  # context: TSO load - TSO RES
    hz = rl[48:]
    lead = int((~keep[: np.argmax(keep)]).sum())  # UTC hours of D-1 (local D)
    assert lead == (2 if day.startswith("2026-07") else 1)
    np.testing.assert_allclose(hz[:lead], 1000.0 - 70.0)  # TSO load, own RES of D-1
    np.testing.assert_allclose(hz[keep], 2000.0 - 50.0)  # the run's load input, own RES of D
    np.testing.assert_allclose(hz[lead + 24 :], 1000.0 - 50.0)  # D+1: 24 h earlier


def test_missing_own_res_of_d_minus_1_falls_back_to_d():
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    ds = _dataset(delivery)
    _, cov, _, _ = t0_shadow.build_inputs(
        delivery, pd.Series(50.0, index=ds.index), ds, _day(delivery, 2000.0),
        _day(delivery, 50.0), None, context=48,
    )  # fmt: skip
    np.testing.assert_allclose(cov[0][48:50], 1000.0 - 50.0)


def _step(tmp_path, model):
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    ds = _dataset(delivery)
    return t0_shadow.t0_step(
        tmp_path, tmp_path, ds, pd.DataFrame(), delivery, pd.Series(50.0, index=ds.index),
        _day(delivery, 2000.0), pd.DataFrame({c: _day(delivery, 20.0) for c in RES_COLS}),
        "2026-07-14T09:20:00+00:00", "00Z", False, model=model,
    )  # fmt: skip


def test_t0_step_logs_24_hours(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        t0_shadow,
        "own_res_parts",
        lambda f, d, c, day: pd.DataFrame({c: _day(day, 25.0) for c in RES_COLS}),
    )
    model = FakeT0()
    assert _step(tmp_path, model) is True
    ctx, cov = model.calls[0]
    assert ctx.shape == (1, t0_shadow.CONTEXT) and cov.shape == (1, 2, t0_shadow.CONTEXT + 26)
    log = pd.read_parquet(tmp_path / t0_shadow.LOG_DIR / "2026-07.parquet")
    assert len(log) == 24 and log.index[0] == pd.Timestamp("2026-07-15", tz="UTC")
    assert list(log.columns) == [
        "generated_utc",
        "weather_vintage",
        "load_surrogate",
        *Q_NAMES,
        "median",
    ]
    np.testing.assert_allclose(log["q05"], 5.0)
    np.testing.assert_allclose(log["median"], 50.0)
    assert (log["weather_vintage"] == "00Z").all() and not log["load_surrogate"].any()
    assert len(Q_NAMES) == len(LEVELS) == 21
    assert "t0 shadow" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["model", "inputs"])
def test_t0_step_never_raises(tmp_path, monkeypatch, capsys, failure):
    class Broken:
        def predict(self, *a, **k):
            raise RuntimeError("out of memory")

    if failure == "model":
        ok = _step(tmp_path, Broken())
    else:
        ok = t0_shadow.t0_step(
            tmp_path, tmp_path, pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-07-15", tz="UTC"),
            pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), "x", "00Z", False,
            model=FakeT0(),
        )  # fmt: skip
    assert ok is False
    assert "::warning::t0 shadow step failed" in capsys.readouterr().out
    assert not (tmp_path / t0_shadow.LOG_DIR).exists()


@pytest.mark.parametrize("flag", [False, True])
def test_run_daily_runs_t0_only_on_request(tmp_path, monkeypatch, flag):
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    hours = pd.date_range(delivery, periods=24, freq="1h")
    features = pd.DataFrame({"run_date": pd.Timestamp("2026-07-14")}, index=hours)
    cache.upsert(
        tmp_path / "cache", "entsoe/load_forecast", pd.DataFrame({"Forecasted Load": 1.0}, hours)
    )
    monkeypatch.setattr(daily_forecast, "update_features", lambda *a: features)
    monkeypatch.setattr(daily_forecast, "build_dataset", lambda c: (pd.DataFrame(), None))
    monkeypatch.setattr(daily_forecast, "site_prices", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(daily_forecast, "quarter_prices", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(
        daily_forecast, "own_res_parts", lambda *a: pd.DataFrame({c: 1.0 for c in RES_COLS}, hours)
    )
    monkeypatch.setattr(
        daily_forecast, "lear_forecast", lambda *a: pd.Series(1.0, hours, name="forecast")
    )
    monkeypatch.setattr(daily_forecast, "write_site_json", lambda *a: None)
    calls = []
    monkeypatch.setattr(t0_shadow, "t0_step", lambda *a: calls.append(a))
    monkeypatch.delitem(sys.modules, "t0", raising=False)

    daily_forecast.run_daily(
        tmp_path / "cache", tmp_path, tmp_path / "features.parquet", tmp_path / "site",
        delivery_day="2026-07-15", skip_fetch=True, t0_shadow=flag,
    )  # fmt: skip

    assert "t0" not in sys.modules  # the model package is never imported by the wiring
    assert len(calls) == int(flag)
    if flag:
        args = calls[0]
        assert args[4] == delivery and args[9:] == ("00Z", False)
        assert (args[6] == 1.0).all() and list(args[7].columns) == RES_COLS  # reused inputs
