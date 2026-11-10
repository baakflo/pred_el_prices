"""Fine-tuned t0 shadow: the 13 rows as trained, fail-soft step, opt-in wiring (fake model)."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pred_el_prices import daily_forecast
from pred_el_prices.features.dataset import NEIGHBOUR_ZONES
from pred_el_prices.pipeline import cache
from pred_el_prices.production import t0_all_shadow as T
from pred_el_prices.production import t0_shadow
from pred_el_prices.production.t0_shadow import Q_NAMES, RES_COLS
from test_t0_shadow import TSO_POST_GATE, FakeT0, _dataset, _day, _Tensor

NB_COLS = [f"load_{z.lower()}_mw" for z in NEIGHBOUR_ZONES]
T0_ORIGIN = t0_shadow.origin_of


def _dataset_all(delivery):
    """_dataset plus the RES parts 100 / 110 / 120 and 9 neighbour loads of 10 MW; the
    neighbours' local D+1 hours are not published at the gate (sentinel)."""
    ds = _dataset(delivery)
    gate_hidden = ds[RES_COLS[0]] == TSO_POST_GATE
    for k, c in enumerate(RES_COLS):
        ds[c] = np.where(gate_hidden, TSO_POST_GATE, 100.0 + 10 * k)
    next_day = (delivery + pd.Timedelta(days=1)).date()
    local_next = ds.index.tz_convert("Europe/Berlin").date == next_day
    for c in NB_COLS:
        ds[c] = np.where(local_next, TSO_POST_GATE, 10.0)
    return ds


def _fuels():
    days = pd.date_range("2026-01-01", "2026-12-31", freq="1D", tz="UTC")
    return pd.DataFrame({"ttf_gas_eur_mwh": np.arange(len(days), dtype=float),
                         "eua_proxy_usd": 70.0}, index=days)  # fmt: skip


def _parts(day, values=(20.0, 30.0, 40.0)):
    return pd.DataFrame({c: _day(day, v) for c, v in zip(RES_COLS, values, strict=True)})


@pytest.mark.parametrize("day", ["2026-07-15", "2026-01-15"])  # CEST (origin 22Z), CET (23Z)
def test_rows_follow_the_training_rules(day):
    delivery = pd.Timestamp(day, tz="UTC")
    ds = _dataset_all(delivery)
    prev = _parts(delivery - pd.Timedelta(days=1), (7.0, 8.0, 9.0))
    _, cov, _, keep = T.build_inputs(
        delivery, pd.Series(50.0, index=ds.index), ds, _fuels(), _day(delivery, 2000.0),
        _parts(delivery), prev, context=48,
    )  # fmt: skip
    assert cov.shape == (13, 48 + t0_shadow.HORIZON) and keep.sum() == 24
    assert (np.abs(cov) < 1e6).all()  # no post-gate value in any row
    r = dict(zip(T.ROWS, cov, strict=True))
    lead = int((~keep[: np.argmax(keep)]).sum())
    # rl: context TSO load - TSO RES; horizon the run's load - own RES (D-1's own RES lead)
    np.testing.assert_allclose(r["rl"][:48], 1000.0 - 330.0)
    np.testing.assert_allclose(r["rl"][48:][keep], 2000.0 - 90.0)
    np.testing.assert_allclose(r["rl"][48:][:lead], 1000.0 - 24.0)
    np.testing.assert_allclose(r["load"][48:][keep], 2000.0)
    # RES parts: context TSO per part, horizon own per part
    for name, tso, own, own_prev in [("wind_on", 100, 20, 7), ("wind_off", 110, 30, 8),
                                     ("solar", 120, 40, 9)]:  # fmt: skip
        np.testing.assert_allclose(r[name][:48], tso)
        np.testing.assert_allclose(r[name][48:][keep], own)
        np.testing.assert_allclose(r[name][48:][:lead], own_prev)
    # fuels: settlement of D-2 flat over the horizon (TTF = day-of-year index)
    d2 = (delivery - pd.Timedelta(days=2) - pd.Timestamp("2026-01-01", tz="UTC")).days
    np.testing.assert_allclose(r["ttf"][48:], d2)
    np.testing.assert_allclose(r["eua"], 70.0)
    # neighbour sum: context 90, local D+1 hours from 24 h earlier (never the sentinel)
    np.testing.assert_allclose(r["nb_sum"], 90.0)
    # calendar of the delivery day's Berlin-local hours
    dow = delivery.dayofweek
    assert r["sat"][48:][keep].max() == float(dow == 5)
    assert r["mon"][48:][keep].max() == float(dow == 0)


@pytest.mark.parametrize("day", ["2026-07-15", "2026-01-15"])
def test_horizon_reads_day_d_from_the_cache_when_the_dataset_ends_at_the_origin(day):
    # the morning run: D has not cleared, so `dataset` (price hours only) stops at the
    # origin; D's neighbour and DE load forecasts are in the cache (training saw them)
    delivery = pd.Timestamp(day, tz="UTC")
    ds = _dataset_all(delivery)
    ds = ds[ds.index < T0_ORIGIN(delivery)]
    local_d = pd.date_range(delivery - pd.Timedelta(hours=2), periods=28, freq="1h")
    local_d = local_d[local_d.tz_convert("Europe/Berlin").date == delivery.date()]
    gate = {"load": pd.Series(3000.0, local_d), "nb": pd.Series(50.0, local_d)}
    prev = _parts(delivery - pd.Timedelta(days=1), (7.0, 8.0, 9.0))
    args = (delivery, pd.Series(50.0, index=ds.index), ds, _fuels(), _day(delivery, 2000.0),
            _parts(delivery), prev)  # fmt: skip
    _, cov, _, keep = T.build_inputs(*args, context=48, gate=gate)
    r = dict(zip(T.ROWS, cov, strict=True))
    lead = int((~keep[: np.argmax(keep)]).sum())
    np.testing.assert_allclose(r["nb_sum"][:48], 90.0)  # context: the dataset's
    np.testing.assert_allclose(r["nb_sum"][48:], 50.0)  # local D (+1 from 24 h earlier)
    np.testing.assert_allclose(r["load"][48:][:lead], 3000.0)
    np.testing.assert_allclose(r["rl"][48:][:lead], 3000.0 - 24.0)
    np.testing.assert_allclose(r["load"][48:][keep], 2000.0)
    # without the cache the horizon would quietly fall back to D-1's forecast
    _, cov, _, _ = T.build_inputs(*args, context=48)
    np.testing.assert_allclose(cov[T.ROWS.index("nb_sum")][48:], 90.0)


def test_bridge_day_and_weekday_flags():
    # Ascension 2026-05-14 (Thu): Fri 05-15 is a bridge day; Mon 05-18 is not
    idx = pd.date_range("2026-05-14", "2026-05-19", freq="1h", tz="Europe/Berlin").tz_convert("UTC")
    cal = T.calendar_flags(idx)
    local = idx.tz_convert("Europe/Berlin")
    by_day = cal.groupby(local.date).max()
    assert by_day.loc[pd.Timestamp("2026-05-15").date(), "bridge"] == 1.0
    assert by_day["bridge"].sum() == 1.0
    assert by_day.loc[pd.Timestamp("2026-05-16").date(), "sat"] == 1.0
    assert by_day.loc[pd.Timestamp("2026-05-18").date(), "mon"] == 1.0


def test_fuels_skip_zero_and_ragged_rows():
    days = pd.date_range("2026-10-01", periods=4, freq="1D", tz="UTC")
    raw = pd.DataFrame({"ttf_gas_eur_mwh": [30.0, 31.0, 32.0, 33.0],
                        "eua_proxy_usd": [70.0, 71.0, 0.0, np.nan]}, index=days)  # fmt: skip
    f = T.gate_safe_fuels(raw, pd.date_range("2026-10-06", periods=2, freq="1h", tz="UTC"))
    np.testing.assert_allclose(f["ttf"], 33.0)  # 10-04 + 2 days
    np.testing.assert_allclose(f["eua"], 71.0)  # the zero and the NaN never enter


def test_missing_neighbour_horizon_raises():
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    ds = _dataset_all(delivery)
    ds.loc[ds.index.normalize() >= delivery - pd.Timedelta(days=3), NB_COLS[0]] = np.nan
    with pytest.raises(RuntimeError, match="nb_sum"):
        T.build_inputs(
            delivery, pd.Series(50.0, index=ds.index), ds, _fuels(), _day(delivery, 2000.0),
            _parts(delivery), None, context=48,
        )  # fmt: skip


def _step(tmp_path, monkeypatch, model, evening=False):
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    monkeypatch.setattr(T, "own_res_parts", lambda f, d, c, day: _parts(day))
    monkeypatch.setattr(t0_shadow, "CONTEXT", 48)
    monkeypatch.setattr(T.build_inputs, "__defaults__", (48, None, None))
    cache.upsert(tmp_path / "cache", "fuels_daily", _fuels())
    ds = _dataset_all(delivery)
    return T.t0_all_step(
        tmp_path, tmp_path / "cache", ds, pd.DataFrame(), delivery, pd.Series(50.0, index=ds.index),
        _day(delivery, 2000.0), _parts(delivery), "2026-07-14T09:20:00+00:00", "00Z", False,
        model=model, evening=evening,
    )  # fmt: skip


def test_step_logs_24_hours_with_checkpoint_name(tmp_path, monkeypatch, capsys):
    fake = FakeT0()
    rows = _step(tmp_path, monkeypatch, (fake, "t0_all_seed2"))
    assert rows is not None and len(rows) == 24
    assert fake.calls[0][1].shape == (1, 13, 48 + 26)
    log = pd.read_parquet(tmp_path / T.LOG_DIR / "2026-07.parquet")
    assert len(log) == 24 and (log["checkpoint"] == "t0_all_seed2").all()
    assert list(log.columns) == ["generated_utc", "weather_vintage", "load_surrogate",
                                 "checkpoint", "evening", "neighbour_surrogate", *Q_NAMES,
                                 "median"]  # fmt: skip
    assert not log["evening"].any() and not log["neighbour_surrogate"].any()
    assert "t0-all: 2026-07-15 logged" in capsys.readouterr().out


class _Const:
    """A fake model returning one value at every level (NaN: a broken forecast)."""

    def __init__(self, v):
        self.v = v

    def predict(self, ctx, horizon, quantile_levels, future_covariates):
        q = np.full((1, horizon, len(quantile_levels)), self.v, dtype=np.float32)
        return SimpleNamespace(quantiles=_Tensor(q))


def test_step_never_raises_and_skips_without_weights(tmp_path, monkeypatch, capsys):
    class Broken:
        def predict(self, *a, **k):
            raise RuntimeError("out of memory")

    assert _step(tmp_path, monkeypatch, (Broken(), "x")) is None
    assert "::warning::t0-all step failed" in capsys.readouterr().out
    assert not (tmp_path / T.LOG_DIR).exists()
    assert T.t0_all_step(tmp_path, tmp_path, None, None, None, None, None, None, "x", "00Z",
                         False, checkpoint=tmp_path / "absent.pt") is None  # fmt: skip
    assert "no checkpoint" in capsys.readouterr().out


def test_step_refuses_a_non_finite_forecast(tmp_path, monkeypatch, capsys):
    assert _step(tmp_path, monkeypatch, (_Const(float("nan")), "x")) is None
    assert "not finite" in capsys.readouterr().out
    assert not (tmp_path / T.LOG_DIR).exists()


def test_load_model_refuses_other_rows(tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "t0_all.pt"
    torch.save({"state": {}, "rows": ["rl", "hol"], "model_revision": t0_shadow.MODEL_REVISION}, p)
    with pytest.raises(RuntimeError, match="checkpoint rows"):
        T.load_model(p)
    torch.save({"state": {}, "rows": T.ROWS, "model_revision": "abc"}, p)
    with pytest.raises(RuntimeError, match="checkpoint base"):
        T.load_model(p)


def test_evening_load_rows_come_from_the_substitutes():
    # the evening edition: no TSO forecast of local D (DE or neighbours) may enter, even
    # where a later cache holds one; DE = the surrogate (load_d), neighbours = nb_sub
    delivery = pd.Timestamp("2026-01-15", tz="UTC")
    ds = _dataset_all(delivery)
    gate = {"load": pd.Series(3000.0, ds.index), "nb": pd.Series(50.0, ds.index)}
    prev = _parts(delivery - pd.Timedelta(days=1), (7.0, 8.0, 9.0))
    _, cov, _, keep = T.build_inputs(
        delivery, pd.Series(50.0, index=ds.index), ds, _fuels(), _day(delivery, 2000.0),
        _parts(delivery), prev, context=48, gate=gate, nb_sub=_day(delivery, 77.0),
    )  # fmt: skip
    r = dict(zip(T.ROWS, cov, strict=True))
    np.testing.assert_allclose(r["load"][48:], 2000.0)  # all horizon hours, D-1's too
    np.testing.assert_allclose(r["nb_sum"][48:], 77.0)
    np.testing.assert_allclose(r["nb_sum"][:48], 90.0)  # the context stays the dataset's
    np.testing.assert_allclose(r["rl"][48:][keep], 2000.0 - 90.0)


def test_evening_step_flags_its_substitutes(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # production.nets loads the network code
    from pred_el_prices.production import nets

    seen = []
    monkeypatch.setattr(
        nets, "neighbour_load_surrogate",
        lambda f, c, d: seen.append(d) or _day(d, 77.0),
    )  # fmt: skip
    rows = _step(tmp_path, monkeypatch, (FakeT0(), "t0_all_seed2"), evening=True)
    assert rows is not None and len(seen) == 1
    assert rows["evening"].all() and rows["neighbour_surrogate"].all()
    assert rows["load_surrogate"].all()


def _run_daily_stubs(tmp_path, monkeypatch):
    """run_daily on stubbed inputs for 2026-07-15; t0_all_step and write_site_json record
    their calls (t0_all_step returns the sentinel "T0ALL") into the returned list."""
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    hours = pd.date_range(delivery, periods=24, freq="1h")
    features = pd.DataFrame({"run_date": pd.Timestamp("2026-07-14")}, index=hours)
    cache.upsert(
        tmp_path / "cache", "entsoe/load_forecast", pd.DataFrame({"Forecasted Load": 1.0}, hours)
    )
    empty = pd.DataFrame({"price_eur_mwh": []}, index=pd.DatetimeIndex([], tz="UTC"))
    monkeypatch.setattr(daily_forecast, "update_features", lambda *a: features)
    monkeypatch.setattr(daily_forecast, "build_dataset", lambda c: (empty, None))
    monkeypatch.setattr(daily_forecast, "site_prices", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(daily_forecast, "quarter_prices", lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(
        daily_forecast, "own_res_parts", lambda *a: pd.DataFrame({c: 1.0 for c in RES_COLS}, hours)
    )
    monkeypatch.setattr(
        daily_forecast, "lear_forecast", lambda *a: pd.Series(1.0, hours, name="forecast")
    )
    calls = []
    monkeypatch.setattr(
        daily_forecast, "write_site_json", lambda *a, **k: calls.append(("site", a, k))
    )
    monkeypatch.setattr(T, "t0_all_step", lambda *a, **k: calls.append(("t0_all", a, k)) or "T0ALL")
    return calls


@pytest.mark.parametrize("ckpt", [None, "models/t0_all.pt"])
def test_run_daily_runs_t0_all_only_with_weights(tmp_path, monkeypatch, ckpt):
    delivery = pd.Timestamp("2026-07-15", tz="UTC")
    calls = _run_daily_stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(t0_shadow, "t0_step", lambda *a: None)
    monkeypatch.delitem(sys.modules, "t0", raising=False)

    daily_forecast.run_daily(
        tmp_path / "cache", tmp_path, tmp_path / "features.parquet", tmp_path / "site",
        delivery_day="2026-07-15", skip_fetch=True, t0_shadow=True,
        t0_all_checkpoint=None if ckpt is None else tmp_path / ckpt,
    )  # fmt: skip

    assert "t0" not in sys.modules
    t0_all = [c for c in calls if c[0] == "t0_all"]
    assert len(t0_all) == int(ckpt is not None)
    if ckpt:
        assert t0_all[0][2]["checkpoint"] == tmp_path / ckpt and t0_all[0][1][4] == delivery
        assert t0_all[0][2]["evening"] is False
