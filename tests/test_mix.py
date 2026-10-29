"""nets+t0 quantile mix: t0 on the 99-level grid, the mix, its log (synthetic only)."""

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.production import mix, site
from pred_el_prices.production.t0_shadow import LEVELS, Q_NAMES

OFFSETS = np.linspace(-20, 20, 99)
SHAPE = np.tile([-3.0, -1.0, 1.0, 3.0], 24)  # within-hour shape, sums to zero per hour


def _t0(day="2026-08-03", fn=lambda tau: 100.0 * tau):
    hours = pd.date_range(day, periods=24, freq="1h", tz="UTC")
    q = np.tile([fn(t) for t in LEVELS], (24, 1))
    return pd.DataFrame(q, index=hours, columns=Q_NAMES).assign(median=lambda d: d["q50"])


def _nets_rows(day="2026-08-03", evening=False, level=100.0):
    hours = pd.date_range(day, periods=24, freq="1h", tz="UTC")
    hourly = pd.DataFrame(level + OFFSETS[None, :].repeat(24, 0), hours, columns=site.Q_COLS)
    hourly[site.R_COLS] = hourly[site.Q_COLS].to_numpy() - 1.0
    quarters = pd.DataFrame(
        hourly[site.Q_COLS].to_numpy().repeat(4, 0) + SHAPE[:, None],
        pd.date_range(day, periods=96, freq="15min", tz="UTC"),
        columns=site.Q_COLS,
    )
    flags = {
        "trained_through": "2026-07-25",
        "recal_days": 300,
        "shaped": True,
        "evening": evening,
        "load_surrogate": evening,
        "neighbour_surrogate": evening,
        "weather_vintage": "12Z" if evening else "00Z",
    }
    return site.log_rows(hourly, quarters, flags, "2026-08-02T09:20:00+00:00")


# ------------------------------------------------------------------ t0 on the grid


def test_t0_grid_keeps_t0s_levels_and_spans_q01_to_q99():
    curved = _t0(fn=lambda tau: np.log(tau / (1 - tau)) * 30 + 80)  # heavy tails
    g = mix.t0_on_grid(curved)
    assert list(g.columns) == site.Q_COLS and g.notna().all().all()  # no extrapolation
    for name, tau in zip(Q_NAMES, LEVELS, strict=True):
        np.testing.assert_allclose(g[f"q{round(tau * 100):02d}"], curved[name])
    assert (np.diff(g.to_numpy(), axis=1) > 0).all()  # monotone


def test_t0_grid_does_not_overshoot_between_knots():
    # flat stretch then a jump: a cubic spline would ring below/above the knots
    g = mix.t0_on_grid(_t0(fn=lambda tau: 50.0 if tau <= 0.5 else 300.0))
    v = g.to_numpy()
    assert (np.diff(v, axis=1) >= -1e-9).all()
    assert v.min() >= 50.0 - 1e-9 and v.max() <= 300.0 + 1e-9
    np.testing.assert_allclose(g["q30"], 50.0)  # inside a flat stretch: flat


def test_t0_grid_on_a_linear_quantile_function_is_linear():
    g = mix.t0_on_grid(_t0())
    np.testing.assert_allclose(g.iloc[0].to_numpy(), np.arange(1, 100), atol=1e-9)


def test_t0_grid_sorts_crossed_levels():
    t0 = _t0()
    t0[["q40", "q45"]] = t0[["q45", "q40"]].to_numpy()
    assert (np.diff(mix.t0_on_grid(t0).to_numpy(), axis=1) >= 0).all()


# ------------------------------------------------------------------ the mix


def test_hourly_mix_is_the_equal_weight_average_per_level():
    rows = _nets_rows()
    h = rows[rows["kind"] == "h"]
    t0 = mix.t0_on_grid(_t0())
    m = mix.mix_hourly(h, t0)
    np.testing.assert_allclose(m.to_numpy(), 0.5 * (h[site.Q_COLS].to_numpy() + t0.to_numpy()))
    assert (np.diff(m.to_numpy(), axis=1) >= 0).all()


def test_hourly_mix_needs_t0_on_the_same_hours():
    rows = _nets_rows()
    with pytest.raises(RuntimeError, match="cover"):
        mix.mix_hourly(rows[rows["kind"] == "h"], mix.t0_on_grid(_t0("2026-08-04")))


def test_quarters_keep_the_nets_shape_on_the_mixed_hours():
    rows = _nets_rows()
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    m = mix.mix_hourly(h, mix.t0_on_grid(_t0()))
    q = mix.mix_quarters(qh, h, m)
    np.testing.assert_allclose(q.to_numpy(), m.to_numpy().repeat(4, 0) + SHAPE[:, None], atol=1e-4)
    # the hourly mean of the four quarters is the mixed hour, at every level
    np.testing.assert_allclose(
        q.groupby(q.index.floor("h")).mean().to_numpy(), m.to_numpy(), atol=1e-4
    )
    assert (np.diff(q.to_numpy(), axis=1) >= 0).all()


def test_quarters_are_monotone_after_clipping():
    rows = _nets_rows(level=-490.0)  # the lower levels sit on the -500 floor in the nets
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    qh = qh.copy()
    qh[site.Q_COLS] = np.clip(qh[site.Q_COLS].to_numpy(), -500.0, None)
    m = mix.mix_hourly(h, mix.t0_on_grid(_t0(fn=lambda tau: -600.0 + 50 * tau)))
    q = mix.mix_quarters(qh, h, m).to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and q.min() >= -500.0


# ------------------------------------------------------------------ rows and the log


def test_mix_rows_carry_the_nets_flags_and_components():
    rows = mix.mix_rows(_nets_rows(evening=True), _t0(), "2026-08-01T21:50:00+00:00")
    assert len(rows) == 120 and (rows["kind"] == "h").sum() == 24
    assert (rows["components"] == "nets+t0").all()
    assert (rows["nets_generated_utc"] == "2026-08-02T09:20:00+00:00").all()
    assert rows["evening"].all() and rows["neighbour_surrogate"].all()
    assert (rows["trained_through"] == "2026-07-25").all()
    assert not any(c.startswith("r") and c[1:].isdigit() for c in rows.columns)  # no raw


def test_without_t0_the_mix_is_the_nets_flagged():
    nets_rows = _nets_rows()
    rows = mix.mix_rows(nets_rows, None, "x")
    assert (rows["components"] == "nets").all()
    np.testing.assert_allclose(
        rows[site.Q_COLS].to_numpy(), nets_rows[site.Q_COLS].to_numpy(), atol=1e-4
    )


def test_mix_step_logs_partitions_like_the_nets_log(tmp_path, capsys):
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows())
    day = pd.Timestamp("2026-08-03", tz="UTC")
    assert mix.mix_step(tmp_path, day, _t0(), "2026-08-02T09:20:00+00:00") is True
    assert sorted(p.name for p in (tmp_path / site.MIX_LOG_DIR).glob("*")) == ["2026-08.parquet"]
    log = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    assert len(log) == 120 and mix.mix_logged(tmp_path, day)
    assert "mix: 2026-08-03 logged (nets+t0)" in capsys.readouterr().out
    # a t0-less rerun stands (last row wins) and no longer counts as a mix with t0
    mix.mix_step(tmp_path, day, None, "2026-08-02T09:50:00+00:00")
    assert not mix.mix_logged(tmp_path, day)


def test_mix_step_never_mixes_editions(tmp_path, capsys):
    # the morning nets failed: the evening nets stand, the morning t0 must not join them
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(evening=True))
    day = pd.Timestamp("2026-08-03", tz="UTC")
    assert mix.mix_step(tmp_path, day, _t0(), "x", evening=False) is False
    assert "no morning nets" in capsys.readouterr().out
    assert not (tmp_path / site.MIX_LOG_DIR).exists()
    assert mix.mix_step(tmp_path, day, _t0(), "x", evening=True) is True
    assert not mix.mix_logged(tmp_path, day)  # an evening mix is no morning mix


def test_mix_step_failure_is_contained(tmp_path, capsys):
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows())
    broken = _t0("2026-08-04")  # wrong hours
    assert mix.mix_step(tmp_path, pd.Timestamp("2026-08-03", tz="UTC"), broken, "x") is False
    assert "::warning::mix step failed" in capsys.readouterr().out
    assert not (tmp_path / site.MIX_LOG_DIR).exists()
    assert mix.mix_step(tmp_path, pd.Timestamp("2026-08-05", tz="UTC"), None, "x") is False


def test_mix_log_ignores_a_legacy_nets_file_next_to_it(tmp_path):
    _nets_rows().to_parquet(tmp_path / site.LOG_NAME)  # single-file nets log, legacy state
    assert site.read_partitions(tmp_path / site.MIX_LOG_DIR) is None
    assert not (tmp_path / site.MIX_LOG_DIR).exists()


# ------------------------------------------------------------------ site blocks


def _t0_logged(day="2026-08-03", generated="2026-08-02T09:20:00+00:00", fn=None, **flags):
    t0 = _t0(day) if fn is None else _t0(day, fn)
    return t0.assign(generated_utc=generated, **flags)


def test_t0_site_quarters_are_t0s_hours_plus_the_nets_shape():
    rows = mix.t0_site_rows(_t0_logged(), _nets_rows())
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    np.testing.assert_allclose(h[site.Q_COLS].iloc[0], np.arange(1, 100), atol=1e-4)
    want = h[site.Q_COLS].to_numpy().repeat(4, 0) + SHAPE[:, None]
    np.testing.assert_allclose(qh[site.Q_COLS].to_numpy(), want, atol=1e-4)
    assert not rows["flat_quarters"].any() and not rows["evening"].any()
    assert (rows["generated_utc"] == "2026-08-02T09:20:00+00:00").all()


def test_t0_site_quarters_without_nets_repeat_the_hour():
    rows = mix.t0_site_rows(_t0_logged(evening=True), None)
    h, qh = rows[rows["kind"] == "h"], rows[rows["kind"] == "qh"]
    assert len(qh) == 96 and rows["flat_quarters"].all() and rows["evening"].all()
    np.testing.assert_allclose(qh[site.Q_COLS].to_numpy(), h[site.Q_COLS].to_numpy().repeat(4, 0))


def test_site_mix_rows_equal_the_logged_mix_and_flag_mixed_editions():
    nets = _nets_rows(evening=True)
    t0 = _t0_logged(generated="2026-08-02T09:50:00+00:00", evening=False)
    rows = mix.site_mix_rows(nets, t0)
    logged = mix.mix_rows(nets, t0, "x")
    np.testing.assert_allclose(rows[site.Q_COLS].to_numpy(), logged[site.Q_COLS].to_numpy())
    assert rows["mixed_editions"].all() and rows["evening"].all()
    assert (rows["generated_utc"] == "2026-08-02T09:50:00+00:00").all()  # the later run
    assert (rows["t0_generated_utc"] == "2026-08-02T09:50:00+00:00").all()
    assert (rows["nets_generated_utc"] == "2026-08-02T09:20:00+00:00").all()


def test_site_days_skip_incomplete_t0_days(tmp_path, capsys):
    site._write_partitions(tmp_path / "t0_log", _t0_logged().iloc[:20])
    nets = {pd.Timestamp("2026-08-03", tz="UTC"): _nets_rows()}
    assert mix.site_days(tmp_path, nets) == {"t0": {}, "mix": {}}
    assert "incomplete (20 hours)" in capsys.readouterr().out
    assert mix.site_days(tmp_path / "none", nets, ("t0",)) == {"t0": {}}


# ------------------------------------------------------------------ t0 recalibration

D = pd.Timestamp("2026-08-03", tz="UTC")


def _t0_history(first="2026-06-01", last="2026-08-03", fn=lambda tau: 100.0 * tau):
    """Logged t0 rows for every day first..last (the delivery day D among them)."""
    days = pd.date_range(first, last, freq="D", tz="UTC")
    return pd.concat([_t0_logged(f"{d:%Y-%m-%d}", generated="g", fn=fn) for d in days])


def _actuals(first="2026-06-01", last="2026-08-04", value=95.0):
    """Clearing prices: always near t0's 95th percentile (t0's tails far too low)."""
    idx = pd.date_range(first, pd.Timestamp(last) + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.Series(value + rng.normal(0, 2, len(idx)), idx)


def test_t0_recalibration_moves_the_lopsided_tails():
    cal = mix.T0Calibration(_t0_history(), _actuals())
    q, n = cal.hourly(_t0_logged())
    assert n == len(pd.date_range("2026-06-01", "2026-08-01"))  # days <= D-2 only
    raw = mix.t0_on_grid(_t0_logged())
    assert (q["q50"] > raw["q50"] + 30).all()  # the actuals sat near t0's 95th percentile
    assert (np.diff(q.to_numpy(), axis=1) >= 0).all()


def test_t0_recalibration_never_reads_d_minus_1_or_d():
    hist, prices = _t0_history(), _actuals()
    want, n = mix.T0Calibration(hist[hist.index < D - pd.Timedelta(days=1)], prices).hourly(_t0())
    # D-1's and D's forecasts and prices wildly different: nothing may change
    late = hist.index >= D - pd.Timedelta(days=1)
    hist.loc[late, Q_NAMES] = hist.loc[late, Q_NAMES] * 10.0 - 500.0
    prices[prices.index >= D - pd.Timedelta(days=1)] = 3000.0
    got, m = mix.T0Calibration(hist, prices).hourly(_t0())
    assert m == n
    pd.testing.assert_frame_equal(got, want)


@pytest.mark.parametrize("n_days, recalibrated", [(27, False), (28, True)])
def test_t0_below_28_pit_days_stays_raw(n_days, recalibrated):
    first = D - pd.Timedelta(days=n_days + 1)  # n_days days up to D-2, plus D-1 and D
    hist = _t0_history(first=f"{first:%Y-%m-%d}")
    q, n = mix.T0Calibration(hist, _actuals()).hourly(_t0())
    raw = mix.t0_on_grid(_t0())
    assert n == (n_days if recalibrated else 0)
    assert np.allclose(q.to_numpy(), raw.to_numpy()) is not recalibrated


def test_the_mix_uses_the_recalibrated_t0():
    cal = mix.T0Calibration(_t0_history(), _actuals())
    nets_rows = _nets_rows()
    rows = mix.mix_rows(nets_rows, _t0(), "x", cal)
    h = rows[rows["kind"] == "h"]
    t0_99, n = cal.hourly(_t0())
    want = 0.5 * (
        nets_rows.loc[nets_rows["kind"] == "h", site.Q_COLS].to_numpy() + t0_99.to_numpy()
    )
    np.testing.assert_allclose(h[site.Q_COLS].to_numpy(), want, atol=1e-3)
    assert (rows["t0_recal_days"] == n).all() and n > 0
    assert "t0_recal_days" not in mix.mix_rows(nets_rows, None, "x", cal)  # nets only


def test_t0_history_is_the_seed_then_the_log(tmp_path):
    seed = _t0_history("2026-06-01", "2026-07-31", fn=lambda tau: 50.0 * tau)[Q_NAMES]
    seed.to_parquet(tmp_path / site.T0_SEED_NAME)
    site._write_partitions(tmp_path / "t0_log", _t0_history("2026-07-30", "2026-08-03"))
    hist = mix.t0_history(tmp_path)
    assert hist.index.is_unique and len(hist) == 24 * len(pd.date_range("2026-06-01", "2026-08-03"))
    assert hist.loc["2026-07-29", "q99"].eq(49.5).all()  # seed
    assert hist.loc["2026-07-30", "q99"].eq(99.0).all()  # the log wins where both exist


def test_seed_t0_pit_keeps_the_400_days_before_the_first_log_day(tmp_path):
    backtest = _t0_history("2025-08-01", "2026-09-24")
    backtest.insert(0, "delivery_day", backtest.index.normalize())
    backtest.to_parquet(tmp_path / "t0_B.parquet")
    seed = mix.seed_t0_pit(tmp_path / "t0_B.parquet", pd.Timestamp("2026-09-30", tz="UTC"))
    assert list(seed.columns) == Q_NAMES and seed.index.name == "t"
    assert seed.index.min() == pd.Timestamp("2025-08-26", tz="UTC")  # 365 + 35 days
    assert seed.index.max() == pd.Timestamp("2026-09-24 23:00", tz="UTC")


def test_mix_step_recalibrates_and_leaves_the_t0_log_raw(tmp_path):
    site._write_partitions(tmp_path / "t0_log", _t0_history())
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows())
    before = {p.name: p.read_bytes() for p in (tmp_path / "t0_log").glob("*")}
    assert mix.mix_step(tmp_path, D, _t0(), "x", prices=_actuals())
    log = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    assert (log["t0_recal_days"] == 62).all()
    days = mix.site_days(tmp_path, {D: _nets_rows()}, prices=_actuals())
    assert {p.name: p.read_bytes() for p in (tmp_path / "t0_log").glob("*")} == before
    t0_rows = days["t0"][D]
    want, _ = mix.T0Calibration(_t0_history(), _actuals()).hourly(_t0())
    np.testing.assert_allclose(
        t0_rows.loc[t0_rows["kind"] == "h", site.Q_COLS].to_numpy(), want.to_numpy(), atol=1e-3
    )
    assert (t0_rows["t0_recal_days"] == 62).all()
    # the logged mix (recalibrated) stands; a mix logged on raw t0 is mixed anew
    np.testing.assert_allclose(
        days["mix"][D][site.Q_COLS].to_numpy(), log[site.Q_COLS].to_numpy(), atol=1e-3
    )
    raw_mix = mix.mix_rows(_nets_rows(), _t0(), "x")
    assert not mix._mix_stands(raw_mix, _nets_rows(), recal=True)
    assert mix._mix_stands(raw_mix, _nets_rows(), recal=False)


@pytest.mark.parametrize("day", ["2026-10-25", "2027-03-28"])  # DST end (25 h), start (23 h)
def test_dst_days_mix_24_hours_and_96_quarters(tmp_path, day):
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(day))
    assert mix.mix_step(tmp_path, pd.Timestamp(day, tz="UTC"), _t0(day), "x")
    log = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    h, qh = log[log["kind"] == "h"], log[log["kind"] == "qh"]
    assert (h.index == pd.date_range(day, periods=24, freq="1h", tz="UTC")).all()
    assert (qh.index == pd.date_range(day, periods=96, freq="15min", tz="UTC")).all()
