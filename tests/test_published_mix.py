"""The published nets + t0-all mix inside the site's nets block (--primary mix, nets alone).

Synthetic logs only. The live site reads latest.json / history.json / days/*.json in the
shape they had before the mix (2026-11-11); these tests pin that shape.
"""

import json

import numpy as np
import pandas as pd
import pytest

from pred_el_prices import daily_forecast
from pred_el_prices.daily_forecast import published_mix_step, write_site_json
from pred_el_prices.production import mix, site
from pred_el_prices.production import t0_all_shadow as T
from pred_el_prices.production.t0_shadow import LEVELS, Q_NAMES
from test_site_v2 import _lear_log, _nets_log, _prices

DAYS = ("2026-08-01", "2026-08-02", "2026-08-03")
NETS_GEN = "2026-07-31T09:10:00+00:00"  # test_site_v2._nets_log's run

# the key structure of the live latest.json (website origin/main, delivery 2026-11-11)
LIVE_TOP = {"generated_utc", "delivery_day", "model", "pre_gate", "weather_vintage", "note",
            "hours", "schema", "levels", "nets"}  # fmt: skip
LIVE_NETS = {"model", "trained_through", "generated_utc", "hours", "quarters"}
LIVE_POINT = {"t", "q", "actual"}


def _t0(day, shift=10.0):
    """24 t0-all rows: its 21 levels on the nets' spread around their median (LEAR's
    curve) + `shift`, so the equal-weight mix is the nets + shift / 2 at every level."""
    idx = pd.date_range(day, periods=24, freq="1h", tz="UTC")
    f = _lear_log().loc[idx, "forecast"].to_numpy()
    offs = np.interp(LEVELS, np.arange(1, 100) / 100, np.linspace(-20, 20, 99))
    return pd.DataFrame(f[:, None] + offs + shift, index=idx, columns=Q_NAMES)


def _setup(tmp_path, evening=False):
    lear = _lear_log()
    if evening:
        lear["evening"] = lear.index.normalize() == pd.Timestamp(DAYS[2], tz="UTC")
        lear["load_surrogate"] = lear["evening"]
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    nets = _nets_log(lear, DAYS)
    if evening:
        last = nets.index.normalize() == pd.Timestamp(DAYS[2], tz="UTC")
        for col in ("evening", "load_surrogate", "neighbour_surrogate"):
            nets[col] = last
    site.append_log(tmp_path / site.LOG_DIR, nets)
    return lear


def _site(tmp_path, lear, primary="mix", mix_from=DAYS[1]):
    prices, prices_qh = _prices(lear)
    write_site_json(
        tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh,
        primary=primary, mix_from=mix_from if primary != "nets" else None,
    )  # fmt: skip
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    return latest, {d["day"]: d for d in history["days"]}


def _day_file(tmp_path, day):
    return json.loads((tmp_path / site.DAYS_DIR / f"{day}.json").read_text(encoding="utf-8"))


def _log_mix(tmp_path, day, t0, evening=False):
    d = pd.Timestamp(day, tz="UTC")
    return mix.mix_step(tmp_path, d, t0, "2026-08-02T09:20:00+00:00", evening,
                        label=mix.WITH_T0_ALL)  # fmt: skip


def test_the_mix_is_published_inside_the_nets_block_in_the_live_shape(tmp_path):
    lear = _setup(tmp_path)
    for day in DAYS:
        assert _log_mix(tmp_path, day, _t0(day))
    latest, history = _site(tmp_path, lear)
    assert set(latest) == LIVE_TOP  # no primary key, no sibling blocks
    nets = latest["nets"]
    assert set(nets) == LIVE_NETS
    assert set(nets["hours"][0]) == LIVE_POINT and set(nets["quarters"][0]) == LIVE_POINT
    assert len(nets["hours"]) == 24 and len(nets["quarters"]) == 96
    assert nets["model"] == site.PUBLISHED_MIX_LABEL
    assert nets["trained_through"] == "2026-07-25"  # the nets run's
    # mix = nets + 5 at every level (t0-all = nets + 10, equal weights)
    nets_only = latest["hours"][0]["forecast"] + np.linspace(-20, 20, 99)[49]
    assert nets["hours"][0]["q"][4] == pytest.approx(nets_only + 5.0, abs=0.02)
    assert nets["quarters"][0]["q"][4] == pytest.approx(nets_only + 5.0, abs=0.02)
    # history and day files: same keys as a nets day, scores of the mix
    assert set(history[DAYS[1]]["nets"]) == set(history[DAYS[0]]["nets"])
    assert history[DAYS[1]]["nets"]["mae"] == 0.0  # actual = LEAR + 5 = the mix median
    assert history[DAYS[0]]["nets"]["mae"] == 5.0  # before the cut-over: the nets
    f0, f1 = _day_file(tmp_path, DAYS[0]), _day_file(tmp_path, DAYS[1])
    assert set(f1) == set(f0) and set(f1["nets"]) == set(f0["nets"])
    assert f0["nets"]["model"] == site.NETS_MODEL_LABEL
    assert f1["nets"]["model"] == site.PUBLISHED_MIX_LABEL


def test_days_before_the_cut_over_keep_their_published_nets_blocks(tmp_path):
    lear = _setup(tmp_path)
    _, before = _site(tmp_path, lear, primary="nets")
    for day in DAYS:
        _log_mix(tmp_path, day, _t0(day))
    latest, after = _site(tmp_path, lear, mix_from=DAYS[2])
    assert after[DAYS[0]] == before[DAYS[0]] and after[DAYS[1]] == before[DAYS[1]]
    assert latest["nets"]["model"] == site.PUBLISHED_MIX_LABEL  # DAYS[2] = the cut-over


def test_the_evening_edition_publishes_the_mix_with_the_nets_flags(tmp_path):
    lear = _setup(tmp_path, evening=True)
    assert _log_mix(tmp_path, DAYS[2], _t0(DAYS[2]), evening=False) is False  # editions
    assert _log_mix(tmp_path, DAYS[2], _t0(DAYS[2]), evening=True)
    latest, _ = _site(tmp_path, lear)
    nets = latest["nets"]
    assert nets["model"] == site.PUBLISHED_MIX_LABEL
    flags = {k for k in nets if k not in LIVE_NETS}
    assert flags == {"evening", "load_surrogate", "neighbour_surrogate"}  # as the nets'
    assert latest["evening"] is True


def test_t0_all_failure_publishes_the_nets_exactly_as_before(tmp_path, capsys):
    lear = _setup(tmp_path)
    nets_latest, nets_history = _site(tmp_path, lear, primary="nets")
    d = pd.Timestamp(DAYS[2], tz="UTC")
    ok = published_mix_step(
        tmp_path, tmp_path, None, None, d, None, pd.Series(dtype=float), pd.DataFrame(),
        "2026-08-02T09:20:00+00:00", "00Z", False, False, tmp_path / "absent.pt",
    )  # fmt: skip
    assert ok is False
    assert "PUBLISHED FORECAST FALLBACK" in capsys.readouterr().out
    logged = site.read_partitions(tmp_path / site.MIX_LOG_DIR)
    assert (logged["components"] == "nets").all() and logged["t0_missing"].all()
    latest, history = _site(tmp_path, lear, mix_from=DAYS[0])
    assert latest["nets"] == nets_latest["nets"] and history == nets_history


def test_a_mix_on_an_older_nets_run_falls_back_to_the_nets(tmp_path):
    lear = _setup(tmp_path)
    _log_mix(tmp_path, DAYS[2], _t0(DAYS[2]))
    rerun = _nets_log(lear, [DAYS[2]]).assign(generated_utc="2026-08-02T09:55:00+00:00")
    site.append_log(tmp_path / site.LOG_DIR, rerun)
    latest, _ = _site(tmp_path, lear)
    assert latest["nets"]["model"] == site.NETS_MODEL_LABEL


def test_a_zero_shot_mix_is_never_published_in_the_nets_block(tmp_path):
    lear = _setup(tmp_path)
    mix.mix_step(tmp_path, pd.Timestamp(DAYS[2], tz="UTC"), _t0(DAYS[2]), "x")  # nets+t0
    latest, _ = _site(tmp_path, lear)
    assert latest["nets"]["model"] == site.NETS_MODEL_LABEL


def test_the_mix_stays_inside_the_sdac_limits(tmp_path):
    lear = _setup(tmp_path)
    _log_mix(tmp_path, DAYS[2], _t0(DAYS[2], shift=9000.0))
    latest, _ = _site(tmp_path, lear)
    top = max(max(h["q"]) for h in latest["nets"]["hours"] + latest["nets"]["quarters"])
    assert top <= mix.PRICE_BOUNDS[1]


def test_primary_mix_alone_needs_its_cut_over_and_no_sibling_blocks_mean_no_primary(tmp_path):
    lear = _setup(tmp_path)
    prices, prices_qh = _prices(lear)
    with pytest.raises(ValueError, match="mix_from"):
        write_site_json(tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh,
                        primary="mix")  # fmt: skip
    with pytest.raises(ValueError, match="not among"):
        write_site_json(tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh,
                        site_models=("nets", "t0"), primary="mix", mix_from=DAYS[0])  # fmt: skip


@pytest.mark.parametrize("evening", [False, True])
def test_run_daily_publishes_the_t0_all_mix_in_both_editions(tmp_path, monkeypatch, evening):
    """run_daily's wiring: t0-all runs in both editions (evening flag passed on), then the
    published mix step; the zero-shot t0 stays a shadow."""
    from pred_el_prices.production import t0_shadow
    from test_t0_all_shadow import _run_daily_stubs

    calls = _run_daily_stubs(tmp_path, monkeypatch)
    monkeypatch.setattr(daily_forecast, "nets_step", lambda *a, **k: calls.append(("nets", k)))
    monkeypatch.setattr(t0_shadow, "t0_step", lambda *a: None)
    monkeypatch.setattr(mix, "mix_step", lambda *a, **k: calls.append(("mix", a, k)) or True)
    delivery = "2026-07-15"
    daily_forecast.run_daily(
        tmp_path / "cache", tmp_path, tmp_path / "features.parquet", tmp_path / "site",
        delivery_day=delivery, skip_fetch=True, t0_shadow=True, evening=evening,
        nets_bundle=tmp_path / "nets.pt", t0_all_checkpoint=tmp_path / "t0_all.pt",
        primary="mix", mix_from=delivery,
    )  # fmt: skip
    kinds = [c[0] for c in calls]
    assert kinds == ["nets", "t0_all", "mix", "site"]
    t0_all = calls[1]
    assert t0_all[2]["evening"] is evening and t0_all[2]["checkpoint"] == tmp_path / "t0_all.pt"
    _, args, kw = calls[2]
    assert args[2] == "T0ALL" and kw["label"] == mix.WITH_T0_ALL and args[4] is evening
    assert calls[3][2] == {"site_models": ("nets",), "primary": "mix", "mix_from": delivery}


def test_substitute_horizon_reads_the_utc_day_only():
    day = pd.Timestamp("2026-01-15", tz="UTC")  # CET: origin 23Z of D-1, 26 h horizon
    hz = pd.date_range(day - pd.Timedelta(hours=1), periods=26, freq="1h")
    sub = pd.Series(np.arange(24.0), index=pd.date_range(day, periods=24, freq="1h"))
    v = T.substitute_horizon(sub, hz, day)
    assert v[0] == 23.0  # UTC 23 of D-1 = local D 00: the substitute's 23 UTC of D
    np.testing.assert_allclose(v[1:25], np.arange(24.0))
    assert v[25] == 0.0  # UTC 00 of D+1 = local D+1: 24 h earlier
