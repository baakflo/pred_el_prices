"""Learned nets+t0 mixes M1/M3 (production/mix_learned.py), synthetic only."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.production import mix, mix_learned, site
from pred_el_prices.production.t0_shadow import LEVELS, Q_NAMES

DAY = pd.Timedelta(days=1)
D = pd.Timestamp("2026-12-20", tz="UTC")  # after NETS_LIVE_FROM
OFFSETS = np.linspace(-20, 20, 99)
SHAPE = np.tile([-3.0, -1.0, 1.0, 3.0], 24)


def _history(n_days, last=D - 2 * DAY, nets_level=100.0, t0_level=60.0, y=100.0, seed=0):
    """n_days consecutive complete days ending `last` (+ D-1 and D, which a fit must not
    read): nets centred on `nets_level`, t0 on `t0_level`, prices near `y`."""
    days = pd.date_range(last - (n_days - 1) * DAY, last + 2 * DAY, freq="D")
    n = len(days)
    rng = np.random.default_rng(seed)
    qn = np.broadcast_to(nets_level + OFFSETS, (n, 24, 99)).copy()
    qt = np.broadcast_to(t0_level + OFFSETS, (n, 24, 99)).copy()
    yy = y + rng.normal(0, 5, (n, 24))
    feat = np.stack(
        [
            50000 + rng.normal(0, 3000, (n, 24)),
            20000 + rng.normal(0, 5000, (n, 24)),
            np.zeros((n, 24)),
            np.repeat(rng.normal(90, 10, (n, 1)), 24, 1),
        ],
        -1,
    )
    return mix_learned.History(days, qn, qt, yy, feat, np.zeros(n, bool), np.zeros(n, bool))


# ------------------------------------------------------------------ M1 solver


def test_chain_dp_is_the_exact_minimum():
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 1, 6)
    unary = rng.random((4, 6))
    pair = 0.7 * (grid[:, None] - grid[None, :]) ** 2
    ks = mix_learned.chain_dp(unary, pair)

    def cost(k):
        return sum(unary[h, k[h]] for h in range(4)) + sum(pair[k[h], k[h + 1]] for h in range(3))

    best = min(itertools.product(range(6), repeat=4), key=cost)
    assert cost(ks) == pytest.approx(cost(best))


def _toy(n_days=30):
    """Hours 0-11: the nets are exact (the price sits on their median), 12-23: t0 is."""
    rng = np.random.default_rng(5)
    level = rng.normal(100, 20, (n_days, 24))
    qn = level[..., None] + OFFSETS
    qt = level[..., None] + OFFSETS
    qn[:, 12:] += 30.0
    qt[:, :12] -= 30.0
    return qn, qt, level


def test_m1_unpenalised_picks_the_better_model_per_hour():
    qn, qt, y = _toy()
    w = mix_learned.fit_m1(qn, qt, y, lam=0.0, mu=0.0)
    np.testing.assert_allclose(w[:12], 1.0)
    np.testing.assert_allclose(w[12:], 0.0)


def test_m1_shrinks_toward_half_and_smooths_across_hours():
    qn, qt, y = _toy()
    np.testing.assert_allclose(mix_learned.fit_m1(qn, qt, y, lam=1e6, mu=0.0), 0.5)
    w = mix_learned.fit_m1(qn, qt, y, lam=0.0, mu=1e6)
    assert np.ptp(w) == 0.0  # one weight for all hours
    w_ref = mix_learned.fit_m1(qn, qt, y)  # lambda 1, mu 3
    assert ((w_ref >= 0) & (w_ref <= 1)).all()
    assert np.allclose(np.round(w_ref / 0.02), w_ref / 0.02)  # on the 0.02 grid
    assert w_ref[:12].mean() > 0.5 > w_ref[12:].mean()


def test_m1_minimises_the_experiments_objective():
    qn, qt, y = _toy()
    rng = np.random.default_rng(9)
    y = y + rng.normal(0, 15, y.shape)  # neither model exact
    lam, mu = mix_learned.M1_LAMBDA, mix_learned.M1_MU

    def objective(w):
        loss = np.array(
            [
                mix_learned._pinball(mix_learned.weighted(qn[:, h], qt[:, h], w[h]), y[:, h]).mean()
                for h in range(24)
            ]
        )
        return (loss + lam * (w - 0.5) ** 2).sum() + mu * (np.diff(w) ** 2).sum()

    w = mix_learned.fit_m1(qn, qt, y)
    best = objective(w)
    for h in range(24):  # no single-hour grid step improves it
        for step in (-0.02, 0.02):
            v = w.copy()
            v[h] = np.clip(v[h] + step, 0, 1)
            assert objective(v) >= best - 1e-12


# ------------------------------------------------------------------ window, fallback


@pytest.mark.parametrize("variant", ["m1", "m3"])
def test_the_fit_never_reads_d_minus_1_or_d(variant):
    hist = _history(90)
    idx = hist.window(D, need_features=variant == "m3")
    assert hist.days[idx].max() == D - 2 * DAY and len(idx) == 90
    feat_d = hist.feat[-1]
    want, info = mix_learned.learned_weights(variant, hist, D, hist.qn[-1], hist.qt[-1], feat_d)
    hist.qn[-2:] += 500.0  # D-1 and D wildly different
    hist.y[-2:] = 3000.0
    got, _ = mix_learned.learned_weights(variant, hist, D, hist.qn[-1] - 500.0, hist.qt[-1], feat_d)
    np.testing.assert_allclose(got, want)
    assert not info["fallback"] and info["fit_days"] == 90


def test_the_window_is_90_calendar_days():
    hist = _history(120)
    assert len(hist.window(D)) == 90
    hist.y[5, 3] = np.nan  # an incomplete day drops out
    hist.y[-10, 3] = np.nan
    assert len(hist.window(D)) == 89


@pytest.mark.parametrize("variant", ["m1", "m3"])
@pytest.mark.parametrize("n_days, fallback", [(59, True), (60, False)])
def test_below_60_fit_days_the_weight_is_one_half(variant, n_days, fallback):
    hist = _history(n_days)
    w, info = mix_learned.learned_weights(variant, hist, D, hist.qn[-1], hist.qt[-1], hist.feat[-1])
    assert info["fallback"] is fallback and info["fit_days"] == n_days
    if fallback:
        np.testing.assert_allclose(w, 0.5)
    else:
        assert w.mean() > 0.7  # prices sit on the nets' median, t0 is 40 below


def test_m3_needs_the_delivery_days_features():
    hist = _history(70)
    with pytest.raises(RuntimeError, match="features"):
        mix_learned.learned_weights("m3", hist, D, hist.qn[-1], hist.qt[-1], None)


def test_m3_has_16_features_and_is_one_half_at_theta_zero():
    hist = _history(3)
    x = mix_learned.m3_raw(hist.qn, hist.qt, hist.feat, hist.days)
    assert x.shape == (len(hist.days), 24, len(mix_learned.M3_FEATURES)) == (5, 24, 16)
    fit = mix_learned.M3Fit(np.zeros(17), x.reshape(-1, 16).mean(0), np.ones(16))
    np.testing.assert_allclose(fit.weights(x[0]), 0.5)


# ------------------------------------------------------------------ history: seed + live


def _nets_rows(day, level=100.0, evening=False):
    hours = pd.date_range(day, periods=24, freq="1h", tz="UTC")
    hourly = pd.DataFrame(level + OFFSETS[None, :].repeat(24, 0), hours, columns=site.Q_COLS)
    hourly[site.R_COLS] = hourly[site.Q_COLS].to_numpy() - 1.0
    quarters = pd.DataFrame(
        hourly[site.Q_COLS].to_numpy().repeat(4, 0) + SHAPE[:, None],
        pd.date_range(day, periods=96, freq="15min", tz="UTC"),
        columns=site.Q_COLS,
    )
    flags = {"trained_through": "2026-12-12", "evening": evening, "load_surrogate": evening}
    return site.log_rows(hourly, quarters, flags, f"{day:%Y-%m-%d}T09:20:00+00:00")


def _t0(day, level=60.0, generated="g"):
    hours = pd.date_range(day, periods=24, freq="1h", tz="UTC")
    q = np.tile([level - 20 + 40 * t for t in LEVELS], (24, 1))
    return pd.DataFrame(q, index=hours, columns=Q_NAMES).assign(
        median=lambda d: d["q50"], generated_utc=generated
    )


def _seed(first, last, nets_level=100.0, t0_level=60.0, features=True):
    days = pd.date_range(first, last, freq="D", tz="UTC")
    idx = pd.DatetimeIndex([h for d in days for h in pd.date_range(d, periods=24, freq="1h")])
    f = pd.DataFrame(
        np.concatenate(
            [
                np.tile(nets_level + OFFSETS, (len(idx), 1)),
                np.tile(t0_level + OFFSETS, (len(idx), 1)),
            ],
            axis=1,
        ),
        idx,
        mix_learned.N_COLS + mix_learned.T_COLS,
    ).astype("float32")
    f["y"] = 100.0
    rng = np.random.default_rng(0)
    for col, v in zip(mix_learned.FEAT_COLS, (50000.0, 20000.0, 0.0, 90.0), strict=True):
        f[col] = v + (rng.normal(0, 1000, len(idx)) if v else 0.0) if features else np.nan
    f.index.name = "t"
    return f


def _prices(first, last, value=100.0):
    idx = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    return pd.Series(value + np.random.default_rng(1).normal(0, 5, len(idx)), idx)


def _empty_cal(prices):
    return mix.T0Calibration(
        pd.DataFrame(columns=Q_NAMES, index=pd.DatetimeIndex([], tz="UTC")), prices
    )


def test_history_takes_the_live_logs_over_the_seed(tmp_path):
    _seed("2026-12-01", "2026-12-15", nets_level=100.0, t0_level=60.0).to_parquet(
        tmp_path / mix_learned.SEED_NAME
    )
    for day in pd.date_range("2026-12-12", "2026-12-18", tz="UTC"):
        site.append_log(tmp_path / site.LOG_DIR, _nets_rows(day, level=200.0))
    for day in pd.date_range("2026-12-10", "2026-12-18", tz="UTC"):
        site._write_partitions(tmp_path / "t0_log", _t0(day, level=10.0))
    prices = _prices("2026-11-01", "2026-12-20")
    hist = mix_learned.learned_history(tmp_path, D, prices, _empty_cal(prices))
    days = [f"{d:%m-%d}" for d in hist.days]
    assert days[0] == "12-01" and days[-1] == "12-18"  # nothing after D-2
    by_day = dict(zip(days, range(len(days)), strict=True))
    i = by_day["12-05"]  # seed only
    assert hist.qn[i, 0, 49] == 100.0 and hist.qt[i, 0, 49] == 60.0 and hist.seed[i]
    i = by_day["12-11"]  # live t0, seed nets
    assert hist.qn[i, 0, 49] == 100.0 and hist.qt[i, 0, 49] == pytest.approx(10.0) and hist.seed[i]
    i = by_day["12-13"]  # both live: live wins
    assert hist.qn[i, 0, 49] == 200.0 and hist.qt[i, 0, 49] == pytest.approx(10.0)
    assert not hist.seed[i]
    np.testing.assert_allclose(hist.y[i], prices.reindex(mix_learned.hours_of(hist.days[i])))
    # features: the seed's; live days without logged features and no dataset: none
    assert np.isfinite(hist.feat[by_day["12-05"]]).all()
    assert np.isnan(hist.feat[by_day["12-17"]]).all() and not hist.proxy.any()


def test_live_nets_before_the_eua_fix_are_skipped(tmp_path):
    day = mix_learned.NETS_LIVE_FROM - DAY
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(day))
    site._write_partitions(tmp_path / "t0_log", _t0(day))
    prices = _prices(f"{day - 5 * DAY:%Y-%m-%d}", f"{day + 5 * DAY:%Y-%m-%d}")
    hist = mix_learned.learned_history(tmp_path, day + 2 * DAY, prices, _empty_cal(prices))
    assert len(hist.days) == 0


def test_proxy_features_fill_live_days_from_the_dataset(tmp_path):
    day = pd.Timestamp("2026-12-17", tz="UTC")
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(day))
    site._write_partitions(tmp_path / "t0_log", _t0(day))
    idx = pd.date_range("2026-12-15", "2026-12-18 23:00", freq="1h", tz="UTC")
    dataset = pd.DataFrame(
        {"load_forecast_mw": 50000.0, **dict.fromkeys(mix_learned.RES_COLS, 5000.0)}, idx
    )
    dataset.loc["2026-12-17 22:00":"2026-12-17 23:00", "load_forecast_mw"] = np.nan
    prices = _prices("2026-12-10", "2026-12-20")
    hist = mix_learned.learned_history(tmp_path, D - DAY, prices, _empty_cal(prices), dataset)
    assert hist.proxy.tolist() == [True]
    f = hist.feat[0]
    np.testing.assert_allclose(f[:, 0], 50000.0)  # UTC 22-23 from 24 h earlier
    np.testing.assert_allclose(f[:, 1], 15000.0)
    want = prices.reindex(pd.date_range("2026-12-16", periods=22, freq="1h", tz="UTC")).mean()
    np.testing.assert_allclose(f[:, 3], want)


def test_recalibrate_days_matches_t0_calibration():
    days = pd.date_range("2026-10-01", "2026-12-20", freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    hist = pd.concat([_t0(d, level=60.0 + rng.normal(0, 5)) for d in days])[Q_NAMES]
    prices = _prices("2026-10-01", "2026-12-20", value=80.0)
    cal = mix.T0Calibration(hist, prices)
    want = {d: cal.hourly(_t0(d))[0].to_numpy() for d in days[[10, 40, -1]]}
    got = mix_learned.recalibrate_days(cal, {d: _t0(d) for d in want})
    for d, q in want.items():
        np.testing.assert_allclose(got[d], q, atol=1e-9)


# ------------------------------------------------------------------ the step and its log


def _state(tmp_path, seed_days=80, evening=False):
    if seed_days:
        _seed(D - (seed_days + 1) * DAY, D - 2 * DAY).to_parquet(tmp_path / mix_learned.SEED_NAME)
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(D, evening=evening))
    hours = mix_learned.hours_of(D)
    return pd.Series(50000.0, hours), pd.Series(20000.0, hours)


def test_learned_step_logs_both_variants_with_weights_and_flags(tmp_path, capsys):
    load_d, own_res = _state(tmp_path, evening=True)
    prices = _prices("2026-09-01", "2026-12-19")
    done = mix_learned.learned_step(
        tmp_path, D, _t0(D), "2026-12-18T21:40:00+00:00", evening=True, prices=prices,
        load_d=load_d, own_res=own_res,
    )  # fmt: skip
    assert done == {"m1": True, "m3": True}
    nets = site.read_log(tmp_path / site.LOG_DIR)
    nets_h, nets_qh = nets[nets["kind"] == "h"], nets[nets["kind"] == "qh"]
    t0_99 = mix.t0_on_grid(_t0(D)).to_numpy()
    for variant in ("m1", "m3"):
        log = site.read_partitions(tmp_path / mix_learned.LOG_DIRS[variant])
        h, qh = log[log["kind"] == "h"], log[log["kind"] == "qh"]
        assert len(h) == 24 and len(qh) == 96 and (log["variant"] == variant).all()
        assert log["evening"].all() and log["load_surrogate"].all()
        assert not log["fallback"].any() and (log["fit_days"] == 80).all()
        assert (log["seed_days"] == 80).all() and (log["components"] == "nets+t0").all()
        w = h["w"].to_numpy(dtype=float)
        assert ((w >= 0) & (w <= 1)).all() and w.mean() > 0.6  # the nets were right
        want = np.clip(t0_99 + w[:, None] * (nets_h[site.Q_COLS].to_numpy() - t0_99), -500, 4000)
        np.testing.assert_allclose(h[site.Q_COLS].to_numpy(), want, atol=1e-3)
        # quarters: the nets' quarters shifted by what the mix moved their hour
        shift = (h[site.Q_COLS].to_numpy() - nets_h[site.Q_COLS].to_numpy()).repeat(4, 0)
        np.testing.assert_allclose(
            qh[site.Q_COLS].to_numpy(), nets_qh[site.Q_COLS].to_numpy() + shift, atol=1e-3
        )
        np.testing.assert_allclose(qh["w"].to_numpy(), np.repeat(w, 4), atol=1e-6)
    m3 = site.read_partitions(tmp_path / mix_learned.LOG_DIRS["m3"])
    np.testing.assert_allclose(m3.loc[m3["kind"] == "h", "f_load"], 50000.0)
    assert (m3["proxy_days"] == 0).all()
    assert "f_load" not in site.read_partitions(tmp_path / mix_learned.LOG_DIRS["m1"])
    assert "mix learned m1: 2026-12-20 logged (80 fit days" in capsys.readouterr().out


def test_without_a_seed_the_learned_mixes_fall_back_to_the_equal_mix(tmp_path):
    load_d, own_res = _state(tmp_path, seed_days=0)
    prices = _prices("2026-09-01", "2026-12-19")
    done = mix_learned.learned_step(
        tmp_path, D, _t0(D), "x", prices=prices, load_d=load_d, own_res=own_res
    )
    assert done == {"m1": True, "m3": True}
    mix.mix_step(tmp_path, D, _t0(D), "x", prices=prices)
    equal = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    for variant in ("m1", "m3"):
        log = site.read_partitions(tmp_path / mix_learned.LOG_DIRS[variant])
        assert log["fallback"].all() and (log["w"] == 0.5).all() and (log["fit_days"] == 0).all()
        for kind in ("h", "qh"):
            np.testing.assert_allclose(
                log.loc[log["kind"] == kind, site.Q_COLS].to_numpy(),
                equal.loc[equal["kind"] == kind, site.Q_COLS].to_numpy(),
                atol=1e-3,
            )


def test_the_learned_step_leaves_the_equal_mix_log_alone(tmp_path):
    load_d, own_res = _state(tmp_path)
    prices = _prices("2026-09-01", "2026-12-19")
    mix.mix_step(tmp_path, D, _t0(D), "x", prices=prices)
    before = {p.name: p.read_bytes() for p in (tmp_path / site.MIX_LOG_DIR).glob("*")}
    mix_learned.learned_step(
        tmp_path, D, _t0(D), "x", prices=prices, load_d=load_d, own_res=own_res
    )
    assert {p.name: p.read_bytes() for p in (tmp_path / site.MIX_LOG_DIR).glob("*")} == before


def test_a_failing_variant_costs_only_its_own_rows(tmp_path, capsys):
    _state(tmp_path)
    prices = _prices("2026-09-01", "2026-12-19")
    done = mix_learned.learned_step(tmp_path, D, _t0(D), "x", prices=prices)  # no M3 inputs
    assert done == {"m1": True, "m3": False}
    assert "::warning::learned mix m3 failed" in capsys.readouterr().out
    assert not (tmp_path / mix_learned.LOG_DIRS["m3"]).exists()


@pytest.mark.parametrize("case", ["no t0", "no prices", "wrong edition", "broken t0", "no nets"])
def test_the_learned_step_never_raises(tmp_path, capsys, case):
    load_d, own_res = _state(tmp_path)
    prices = _prices("2026-09-01", "2026-12-19")
    kw = {"prices": prices, "load_d": load_d, "own_res": own_res}
    t0 = _t0(D)
    day = D
    if case == "no t0":
        t0 = None
    elif case == "no prices":
        kw["prices"] = None
    elif case == "wrong edition":
        kw["evening"] = True
    elif case == "broken t0":
        t0 = _t0(D + DAY)  # wrong hours
    else:
        day = D + DAY
    assert mix_learned.learned_step(tmp_path, day, t0, "x", **kw) == {"m1": False, "m3": False}
    out = capsys.readouterr().out
    if case in ("no prices", "broken t0"):
        assert "::warning::learned mix" in out
    assert not any(tmp_path.glob("mix_m*_log"))


# ------------------------------------------------------------------ seed


def test_seed_mix_learned_builds_the_experiments_inputs(tmp_path):
    days = pd.date_range("2026-03-01", "2026-09-24", freq="D", tz="UTC")
    nets = pd.concat([_nets_rows(d) for d in days])
    nets.to_parquet(tmp_path / "nets_log.parquet")
    t0 = pd.concat([_t0(d) for d in days])[Q_NAMES]
    t0.insert(0, "delivery_day", t0.index.normalize())
    t0.to_parquet(tmp_path / "t0_B.parquet")
    hours = pd.date_range("2026-02-27", "2026-09-26", freq="1h", tz="UTC")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pd.DataFrame(
        {"price": 90.0, "load_tso": 50000.0, "res_tso": 1.0, "holiday_share": 0.0}, hours
    ).to_parquet(inputs / "hourly.parquet")
    own = pd.DataFrame({"delivery_day": hours.normalize(), "own_res": 20000.0}, hours)
    own.to_parquet(inputs / "own_res.parquet")
    prices = _prices("2026-03-01", "2026-09-24")
    seed = mix_learned.seed_mix_learned(
        (tmp_path / "nets_log.parquet",), tmp_path / "t0_B.parquet", inputs, prices,
        pd.Timestamp("2026-09-25", tz="UTC"),
    )  # fmt: skip
    assert seed.index.min() == pd.Timestamp("2026-03-29", tz="UTC")  # 180 days
    assert seed.index.max() == pd.Timestamp("2026-09-24 23:00", tz="UTC")
    assert len(seed) == 24 * 180 and seed.index.name == "t"
    assert list(seed.columns) == [
        *mix_learned.N_COLS,
        *mix_learned.T_COLS,
        "y",
        *mix_learned.FEAT_COLS,
    ]
    np.testing.assert_allclose(seed["n50"], 100.0)
    assert (np.diff(seed[mix_learned.T_COLS].to_numpy(), axis=1) >= 0).all()
    np.testing.assert_allclose(seed["f_price_dm1"], 90.0)
    np.testing.assert_allclose(seed["y"], prices.reindex(seed.index))
