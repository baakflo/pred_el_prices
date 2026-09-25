"""Production network path: gate-safe rows, recalibration window, bundles (synthetic only)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from pred_el_prices.experiments import qnn_replay
from pred_el_prices.experiments.qnn_de import Q_COLS, RES_COLS, days_from_frame
from pred_el_prices.features.dataset import NEIGHBOUR_ZONES
from pred_el_prices.models.qnn import QUANTILES, load_bundle, save_bundle
from pred_el_prices.pipeline import cache
from pred_el_prices.production import nets
from pred_el_prices.production.site import R_COLS

NB_COLS = nets.LOAD_NB_COLS


def _dataset(n_days: int = 60, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="1h", tz="UTC")
    n = len(idx)
    ds = pd.DataFrame(
        {
            "price_eur_mwh": 50 + rng.normal(0, 10, n),
            "load_forecast_mw": rng.normal(55000, 3000, n),
            RES_COLS[0]: rng.normal(12000, 2000, n),
            RES_COLS[1]: rng.normal(3000, 500, n),
            RES_COLS[2]: rng.normal(8000, 2000, n),
            "ttf_gas_eur_mwh": np.repeat(30 + rng.normal(0, 1, n_days), 24),
            "eua_proxy_usd": np.repeat(70 + rng.normal(0, 1, n_days), 24),
        },
        index=idx,
    )
    for c in NB_COLS:
        ds[c] = rng.normal(10000, 1000, n)
    return ds


def _at_gate(ds: pd.DataFrame, cols, delivery):
    """Summed columns for the delivery day, UTC 22-23 from 24 h earlier (the replay rule)."""
    s = ds[cols].sum(axis=1) if isinstance(cols, list) else ds[cols]
    day = s.loc[delivery : delivery + pd.Timedelta(hours=23)].copy()
    prev = s.loc[delivery - pd.Timedelta(days=1) : delivery - pd.Timedelta(hours=1)]
    day.iloc[22:24] = prev.iloc[22:24].to_numpy()
    return day


def _prod_row(ds, delivery, own):
    """The production row from what the gate sees: the dataset ends at D-1 21:00 UTC."""
    seen = ds[ds.index < delivery - pd.Timedelta(hours=2)]
    fuel = ds.loc[delivery, ["ttf_gas_eur_mwh", "eua_proxy_usd"]].to_numpy(dtype=float)
    return nets.gate_row(
        seen,
        delivery,
        _at_gate(ds, "load_forecast_mw", delivery),
        _at_gate(ds, NB_COLS, delivery),
        own.reindex(pd.date_range(delivery, periods=24, freq="1h")),
        fuel,
    )


class TestGateRow:
    def test_equals_the_replay_row(self, monkeypatch, tmp_path):
        """Pins the production row builder to experiments/qnn_replay.py's x_prod."""
        ds = _dataset()
        monkeypatch.setattr(
            qnn_replay,
            "load_days",
            lambda *a, **k: days_from_frame(ds, "2000-01-01", [], neighbours="load"),
        )
        seen = []

        def spy(x_train, y_train, x_pred, n_unscaled, config, seed):
            seen.append(x_pred.copy())
            return np.zeros((len(x_pred), 24, len(QUANTILES)))

        monkeypatch.setattr(qnn_replay, "fit_predict", spy)
        own = pd.Series(
            np.linspace(15000, 25000, 24 * 10),
            index=pd.date_range("2024-02-20", periods=24 * 10, freq="1h", tz="UTC"),
        )
        own.rename("own_res_mw").to_frame().to_parquet(tmp_path / "own.parquet")
        qnn_replay.run(
            tmp_path, "2024-02-20", "2024-02-26", str(tmp_path / "own.parquet"),
            n_seeds=1, heads=["quantile"], n_jobs=1,
        )  # fmt: skip
        replay_prod = seen[0][: len(seen[0]) // 2]
        for i, d in enumerate(pd.date_range("2024-02-20", "2024-02-26", tz="UTC")):
            x, scale = _prod_row(ds, d, own)
            np.testing.assert_allclose(x[0], replay_prod[i], rtol=1e-12, atol=0)
            assert scale == pytest.approx(max(20.0, 2 * ds.loc[d, "ttf_gas_eur_mwh"]
                                               + 0.37 * ds.loc[d, "eua_proxy_usd"]))  # fmt: skip

    def test_post_gate_values_never_reach_the_row(self):
        ds = _dataset()
        d = pd.Timestamp("2024-02-22", tz="UTC")
        own = pd.Series(20000.0, index=pd.date_range(d, periods=24, freq="1h"))
        x_a, _ = _prod_row(ds, d, own)
        bumped = ds.copy()
        day = bumped.index.normalize() == d
        bumped.loc[day, "price_eur_mwh"] += 500.0  # the auction being forecast
        bumped.loc[day, RES_COLS] += 5000.0  # TSO wind/solar, published 18:00 D-1
        late = bumped.index >= d + pd.Timedelta(hours=22)
        bumped.loc[late & day, ["load_forecast_mw", *NB_COLS]] += 5000.0  # next local day
        prev_late = (bumped.index >= d - pd.Timedelta(hours=2)) & (bumped.index < d)
        bumped.loc[prev_late, ["price_eur_mwh", *RES_COLS]] += 500.0  # D-1 UTC 22-23
        x_b, _ = _prod_row(bumped, d, own)
        np.testing.assert_array_equal(x_a, x_b)

    def test_own_res_and_the_known_load_do_reach_the_row(self):
        ds = _dataset()
        d = pd.Timestamp("2024-02-22", tz="UTC")
        own = pd.Series(20000.0, index=pd.date_range(d, periods=24, freq="1h"))
        x_a, _ = _prod_row(ds, d, own)
        x_b, _ = _prod_row(ds, d, own + 1000.0)
        assert (x_a != x_b).sum() == 24
        bumped = ds.copy()
        bumped.loc[d : d + pd.Timedelta(hours=21), "load_forecast_mw"] += 1000.0
        x_c, _ = _prod_row(bumped, d, own)
        assert (x_a != x_c).sum() == 22

    def test_missing_history_fails_loudly(self):
        ds = _dataset()
        d = pd.Timestamp("2024-02-22", tz="UTC")
        own = pd.Series(20000.0, index=pd.date_range(d, periods=24, freq="1h"))
        holed = ds.drop(ds.index[(ds.index >= d - pd.Timedelta(days=3))
                                 & (ds.index < d - pd.Timedelta(days=3, hours=-2))])  # fmt: skip
        with pytest.raises(RuntimeError, match="incomplete"):
            _prod_row(holed, d, own)


def _cache(tmp_path, ds: pd.DataFrame):
    q = pd.date_range(ds.index[0], ds.index[-1] + pd.Timedelta(minutes=45), freq="15min")
    load15 = ds["load_forecast_mw"].reindex(q, method="ffill")
    cache.upsert(tmp_path, "entsoe/load_forecast", load15.to_frame("Forecasted Load"))
    for zone, col in zip(NEIGHBOUR_ZONES, NB_COLS, strict=True):
        cache.upsert(tmp_path, f"entsoe/{zone}/load_forecast", ds[[col]].set_axis(
            ["Forecasted Load"], axis=1))  # fmt: skip
    days = pd.date_range(ds.index[0], ds.index[-1], freq="D")
    fuels = pd.DataFrame(
        {"ttf_gas_eur_mwh": np.arange(len(days)) + 30.0, "eua_proxy_usd": 70.0}, index=days
    )
    cache.upsert(tmp_path, "fuels_daily", fuels)
    return tmp_path


class TestGateInputs:
    def test_boundary_hours_come_from_the_day_before(self, tmp_path):
        ds = _dataset(20)
        d = pd.Timestamp("2024-01-15", tz="UTC")
        got = nets.gate_inputs(_cache(tmp_path, ds), d)
        np.testing.assert_allclose(got["load_de"].to_numpy(), _at_gate(ds, "load_forecast_mw", d))
        np.testing.assert_allclose(got["load_nb"].to_numpy(), _at_gate(ds, NB_COLS, d))
        # fuels: the settlement of D-2 (day index 12 -> 30 + 12)
        np.testing.assert_allclose(got["fuel"], [42.0, 70.0])
        assert got["flags"] == {"load_surrogate": False, "neighbour_surrogate": False}

    def test_unpublished_boundary_hours_are_not_needed(self, tmp_path):
        ds = _dataset(20)
        d = pd.Timestamp("2024-01-15", tz="UTC")
        want = nets.gate_inputs(_cache(tmp_path / "a", ds), d)
        gate = ds[ds.index < d + pd.Timedelta(hours=22)]  # at the gate: 22-23 not out yet
        got = nets.gate_inputs(_cache(tmp_path / "b", gate), d)
        pd.testing.assert_series_equal(got["load_de"], want["load_de"])
        pd.testing.assert_series_equal(got["load_nb"], want["load_nb"])

    def test_missing_de_load_needs_the_surrogate(self, tmp_path):
        ds = _dataset(20)
        d = pd.Timestamp("2024-01-15", tz="UTC")
        root = _cache(tmp_path, ds[ds.index < d])
        with pytest.raises(RuntimeError, match="DE load"):
            nets.gate_inputs(root, d)
        # a neighbour gap fails too, so fill DE and check the flag path only
        surrogate = pd.Series(1.0, index=pd.date_range(d, periods=24, freq="1h"))
        with pytest.raises(RuntimeError, match="load forecast"):
            nets.gate_inputs(root, d, surrogate)


class TestRecalibration:
    def _hist(self, n_days=60):
        idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
        base = np.linspace(-10, 10, 99)
        h = pd.DataFrame(np.tile(base, (len(idx), 1)) + 50, index=idx, columns=R_COLS)
        h["q50"] = 50.0
        rng = np.random.default_rng(0)
        actual = pd.Series(50 + rng.normal(0, 15, len(idx)), index=idx)  # wider than the model
        return h, actual

    def test_only_days_up_to_d_minus_2_count(self):
        h, a = self._hist()
        d = pd.Timestamp("2024-02-20", tz="UTC")
        raw = np.tile(np.linspace(-10, 10, 99), (24, 1)) + 50
        q1, n1 = nets.recalibrate_day(raw, h, a, d)
        a2 = a.copy()
        a2[a2.index.normalize() == d - pd.Timedelta(days=1)] += 1000.0
        q2, n2 = nets.recalibrate_day(raw, h, a2, d)
        np.testing.assert_array_equal(q1, q2)
        assert n1 == n2 == 49  # 2024-01-01 .. 2024-02-18
        a3 = a.copy()
        a3[a3.index.normalize() == d - pd.Timedelta(days=2)] += 1000.0
        q3, _ = nets.recalibrate_day(raw, h, a3, d)
        assert not np.array_equal(q1, q3)
        # the model is too narrow for N(50, 15): recalibration widens the band
        assert (q1[:, 89] - q1[:, 9]).mean() > (raw[:, 89] - raw[:, 9]).mean()

    def test_short_history_passes_raw_through(self):
        h, a = self._hist(20)
        raw = np.tile(np.linspace(-10, 10, 99), (24, 1)) + 50
        q, n = nets.recalibrate_day(raw, h, a, pd.Timestamp("2024-01-21", tz="UTC"))
        assert n == 0
        np.testing.assert_array_equal(q, raw)


class TestTraining:
    def test_training_set_ends_two_days_before_monday(self):
        ds = _dataset(80)
        monday = pd.Timestamp("2024-03-11", tz="UTC")
        x, y, rows = nets.training_set(ds, monday, window_days=30)
        assert rows.max() == monday - pd.Timedelta(days=2)
        assert len(rows) == 30
        bumped = ds.copy()
        bumped.loc[bumped.index >= monday - pd.Timedelta(days=1), "price_eur_mwh"] += 1e3
        x2, y2, _ = nets.training_set(bumped, monday, window_days=30)
        # (ulp noise: pandas' 9-column row sum depends on the frame's length)
        np.testing.assert_allclose(x, x2, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(y, y2)

    def test_bundle_round_trip_and_forecast(self, tmp_path):
        ds = _dataset(80)
        monday = pd.Timestamp("2024-03-11", tz="UTC")
        fitted, meta = nets.train_ensemble(
            ds, monday, n_seeds=2, n_jobs=1, window_days=60, min_days=50,
            config_overrides={"hidden": [16], "max_epochs": 3, "patience": 2},
        )  # fmt: skip
        assert meta["trained_through"] == "2024-03-09"
        assert meta["heads"] == ["jsu", "jsu", "quantile", "quantile"]
        save_bundle(tmp_path / "b.pt", fitted, meta)
        loaded, meta2 = load_bundle(tmp_path / "b.pt")
        assert meta2 == meta
        d = monday + pd.Timedelta(days=1)
        own = pd.Series(20000.0, index=pd.date_range(d, periods=24, freq="1h"))
        x, scale = _prod_row(ds, d, own)
        np.testing.assert_array_equal(
            nets.predict_raw(loaded, x, scale), nets.predict_raw(fitted, x, scale)
        )
        q = nets.predict_raw(loaded, x, scale)
        assert q.shape == (24, 99)
        assert (np.diff(q, axis=1) >= 0).all()
        assert q.min() >= -500 and q.max() <= 4000

    def test_a_bundle_newer_than_the_gate_is_refused(self):
        meta = {"trained_through": "2024-03-10"}
        with pytest.raises(RuntimeError, match="too late"):
            nets.forecast_day(
                [], meta, pd.Timestamp("2024-03-11", tz="UTC"), None, None, None, None,
                None, None,
            )  # fmt: skip


class TestQuarters:
    def test_shape_keeps_the_hourly_median_and_sorts(self):
        rng = np.random.default_rng(0)
        q_idx = pd.date_range("2025-10-01", "2025-11-30 23:45", freq="15min", tz="UTC")
        h_idx = pd.date_range("2025-10-01", "2025-11-30 23:00", freq="1h", tz="UTC")
        hour = q_idx.hour.to_numpy()
        load15 = pd.Series(50000 + 5000 * np.sin(np.arange(len(q_idx)) / 20), index=q_idx)
        solar = pd.Series(np.clip(np.sin((h_idx.hour - 6) / 12 * np.pi), 0, None) * 2e4, h_idx)
        wind = pd.Series(1e4 + rng.normal(0, 100, len(h_idx)), index=h_idx)
        price15 = pd.Series(60 + 10 * (q_idx.minute / 15 - 1.5) * (hour > 5)
                            + rng.normal(0, 1, len(q_idx)), index=q_idx)  # fmt: skip
        qh = nets.QHInputs(load15, solar, wind, price15)
        d = pd.Timestamp("2025-11-30", tz="UTC")
        hours = pd.date_range(d, periods=24, freq="1h")
        q_h = pd.DataFrame(np.tile(np.linspace(40, 80, 99), (24, 1)), index=hours, columns=Q_COLS)
        parts = pd.DataFrame(
            {
                "wind_onshore_forecast_mw": 8000.0,
                "wind_offshore_forecast_mw": 2000.0,
                "solar_forecast_mw": solar.reindex(hours).to_numpy(),
            },
            index=hours,
        )
        p_hist = pd.Series(60.0, index=h_idx)
        out, shaped = nets.quarter_forecast(q_h, d, parts, p_hist, qh)
        assert shaped
        assert len(out) == 96 and out.index[0] == d
        assert (np.diff(out.to_numpy(), axis=1) >= 0).all()
        med = out["q50"].groupby(out.index.floor("h")).mean()
        np.testing.assert_allclose(med.to_numpy(), q_h["q50"].to_numpy(), atol=1e-9)
        # the learnt within-hour ramp: first quarter below the last in daytime hours
        noon = out.loc[d + pd.Timedelta(hours=12) : d + pd.Timedelta(hours=12, minutes=45), "q50"]
        assert noon.iloc[0] < noon.iloc[-1]

    def test_without_the_15_min_load_the_hour_is_repeated(self):
        d = pd.Timestamp("2025-11-30", tz="UTC")
        hours = pd.date_range(d, periods=24, freq="1h")
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        qh = nets.QHInputs(empty, empty, empty, empty)
        q_h = pd.DataFrame(np.tile(np.linspace(40, 80, 99), (24, 1)), index=hours, columns=Q_COLS)
        parts = pd.DataFrame(0.0, index=hours, columns=list(nets_res_cols()))
        out, shaped = nets.quarter_forecast(q_h, d, parts, pd.Series(60.0, hours), qh)
        assert not shaped
        np.testing.assert_array_equal(out.to_numpy(), np.repeat(q_h.to_numpy(), 4, axis=0))


class TestEvening:
    def _setup(self, tmp_path, n_days=60):
        ds = _dataset(n_days)
        root = _cache(tmp_path, ds)
        rng = np.random.default_rng(1)
        feats = pd.DataFrame(
            {"t2m_mean": rng.normal(15, 5, len(ds)), "ssrd_mean": rng.normal(200, 50, len(ds))},
            index=ds.index,
        )
        return ds, root, feats

    def test_neighbour_substitute_uses_only_what_exists_at_21_35_on_d_minus_2(self, tmp_path):
        ds, root, feats = self._setup(tmp_path / "a")
        d = pd.Timestamp("2024-02-20", tz="UTC")
        want = nets.neighbour_load_surrogate(feats, root, d)
        assert len(want) == 24 and want.notna().all()
        bumped = ds.copy()
        late = (bumped.index >= d - pd.Timedelta(hours=2)) | (
            bumped.index.normalize() >= d
        )  # D-1 UTC 22-23 (local D) and everything from D on: unpublished at 21:35 D-2
        bumped.loc[late, NB_COLS] += 5000.0
        got = nets.neighbour_load_surrogate(feats, _cache(tmp_path / "b", bumped), d)
        pd.testing.assert_series_equal(got, want)
        known = bumped.copy()
        known.loc[d - pd.Timedelta(hours=12), NB_COLS] += 5000.0  # D-1 12:00: published
        moved = nets.neighbour_load_surrogate(feats, _cache(tmp_path / "c", known), d)
        assert not moved.equals(want)

    def test_evening_gate_inputs_always_take_the_substitutes(self, tmp_path):
        _, root, _ = self._setup(tmp_path)
        d = pd.Timestamp("2024-02-20", tz="UTC")
        hours = pd.date_range(d, periods=24, freq="1h")
        de, nb = pd.Series(1.0, index=hours), pd.Series(2.0, index=hours)
        got = nets.gate_inputs(root, d, de, nb, evening=True)  # the cache HAS day D
        assert (got["load_de"] == 1.0).all() and (got["load_nb"] == 2.0).all()
        assert got["flags"] == {"load_surrogate": True, "neighbour_surrogate": True}
        np.testing.assert_allclose(got["fuel"], nets.gate_inputs(root, d)["fuel"])
        with pytest.raises(RuntimeError, match="substitutes"):
            nets.gate_inputs(root, d, de, None, evening=True)

    def test_evening_shape_runs_on_the_interpolated_surrogate(self):
        rng = np.random.default_rng(0)
        q_idx = pd.date_range("2025-10-01", "2025-11-30 23:45", freq="15min", tz="UTC")
        h_idx = pd.date_range("2025-10-01", "2025-11-30 23:00", freq="1h", tz="UTC")
        d = pd.Timestamp("2025-11-30", tz="UTC")
        load15 = pd.Series(50000 + 5000 * np.sin(np.arange(len(q_idx)) / 20), index=q_idx)
        load15 = load15[load15.index < d]  # nothing published for D yet
        solar = pd.Series(np.clip(np.sin((h_idx.hour - 6) / 12 * np.pi), 0, None) * 2e4, h_idx)
        wind = pd.Series(1e4 + rng.normal(0, 100, len(h_idx)), index=h_idx)
        price15 = pd.Series(60 + rng.normal(0, 5, len(q_idx)), index=q_idx)
        qh = nets.QHInputs(load15, solar, wind, price15)
        hours = pd.date_range(d, periods=24, freq="1h")
        q_h = pd.DataFrame(np.tile(np.linspace(40, 80, 99), (24, 1)), index=hours, columns=Q_COLS)
        parts = pd.DataFrame(0.0, index=hours, columns=list(nets_res_cols()))
        p_hist = pd.Series(60.0, index=h_idx)
        _, shaped = nets.quarter_forecast(q_h, d, parts, p_hist, qh)
        assert not shaped  # without a surrogate: hourly repeated
        surrogate = pd.Series(np.linspace(40000, 60000, 24), index=hours)
        out, shaped = nets.quarter_forecast(q_h, d, parts, p_hist, qh, surrogate)
        assert shaped and len(out) == 96
        med = out["q50"].groupby(out.index.floor("h")).mean()
        np.testing.assert_allclose(med.to_numpy(), q_h["q50"].to_numpy(), atol=1e-9)


def test_pit_seed_keeps_the_past_window_and_lets_logs_win(tmp_path):
    from pred_el_prices.production import site

    idx = pd.date_range("2025-01-01", "2026-08-31 23:00", freq="1h", tz="UTC")
    q = pd.DataFrame(np.tile(np.linspace(0, 98, 99), (len(idx), 1)), index=idx, columns=Q_COLS)
    q["actual"] = 1.0
    q.to_parquet(tmp_path / "q.parquet")
    cal = q.copy()
    cal["q50"] = 7.0
    cal.to_parquet(tmp_path / "cal.parquet")
    day = pd.Timestamp("2026-07-01", tz="UTC")
    hours = pd.date_range(day, periods=24, freq="1h")
    hourly = pd.DataFrame(100.0, index=hours, columns=Q_COLS)
    hourly[R_COLS] = 200.0
    rows = site.log_rows(hourly, hourly.reindex(pd.date_range(day, periods=96, freq="15min"),
                                                method="ffill"), {}, "x")  # fmt: skip
    site.append_log(tmp_path / "log.parquet", rows)

    before = pd.Timestamp("2026-08-01", tz="UTC")
    seed = nets.seed_pit(
        tmp_path / "q.parquet", tmp_path / "cal.parquet", before, (tmp_path / "log.parquet",)
    )
    assert seed.index.max() == before - pd.Timedelta(hours=1)
    assert seed.index.min() == before - pd.Timedelta(days=400)
    assert (seed.loc[hours, R_COLS] == 200.0).all().all()  # the log wins
    assert (seed.loc[hours, "q50"] == 100.0).all()
    assert (seed.drop(hours)["q50"] == 7.0).all()


def test_backfill_scores_compare_lear_on_full_days_only(tmp_path):
    from pred_el_prices.production import site
    from pred_el_prices.production.backfill import scores

    hours = pd.date_range("2026-08-01", periods=48, freq="1h", tz="UTC")
    hourly = pd.DataFrame(np.tile(np.linspace(40, 60, 99), (48, 1)), index=hours,
                          columns=Q_COLS)  # fmt: skip
    hourly[R_COLS] = hourly[Q_COLS].to_numpy()
    quarters = hourly[Q_COLS].reindex(pd.date_range(hours[0], periods=192, freq="15min"),
                                      method="ffill")  # fmt: skip
    site.append_log(tmp_path / "l.parquet", site.log_rows(hourly, quarters, {}, "x"))
    log = site.read_log(tmp_path / "l.parquet")
    prices = pd.Series(53.0, index=hours)
    lear = pd.Series(50.0, index=hours[:30])  # day 2 only partly forecast by LEAR
    out = scores(log, prices, prices.reindex(quarters.index, method="ffill"), lear)
    assert out["days"] == 2
    assert out["hourly"]["mae"] == 3.0 and out["quarter"]["mae"] == 3.0
    assert out["quarter_repeated_hourly"]["pinball"] == out["quarter"]["pinball"]
    assert out["vs_lear"] == {"days": 1, "lear_mae": 3.0, "nets_mae": 3.0}


def nets_res_cols():
    from pred_el_prices.daily_forecast import RES_TARGETS

    return RES_TARGETS.keys()
