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


@pytest.mark.parametrize("day", ["2026-10-25", "2027-03-28"])  # DST end (25 h), start (23 h)
def test_dst_days_mix_24_hours_and_96_quarters(tmp_path, day):
    site.append_log(tmp_path / site.LOG_DIR, _nets_rows(day))
    assert mix.mix_step(tmp_path, pd.Timestamp(day, tz="UTC"), _t0(day), "x")
    log = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    h, qh = log[log["kind"] == "h"], log[log["kind"] == "qh"]
    assert (h.index == pd.date_range(day, periods=24, freq="1h", tz="UTC")).all()
    assert (qh.index == pd.date_range(day, periods=96, freq="15min", tz="UTC")).all()
