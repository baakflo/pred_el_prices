"""Site data contract v2: the nets block next to the v1 (LEAR) fields (synthetic only)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.daily_forecast import nets_step, write_site_json
from pred_el_prices.production import site

OFFSETS = np.linspace(-20, 20, 99)


def _lear_log(days=3, start="2026-08-01", generated="2026-08-02T09:00:00+00:00"):
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    df = pd.DataFrame({"forecast": np.linspace(50, 150, len(idx))}, index=idx)
    df["generated_utc"] = generated
    return df


def _nets_log(lear: pd.DataFrame, days, backfill: bool = False) -> pd.DataFrame:
    parts = []
    for day in days:
        f = lear.loc[lear.index.normalize() == pd.Timestamp(day, tz="UTC"), "forecast"]
        hourly = pd.DataFrame(f.to_numpy()[:, None] + OFFSETS, index=f.index, columns=site.Q_COLS)
        hourly[site.R_COLS] = hourly[site.Q_COLS].to_numpy() - 1.0
        quarters = hourly[site.Q_COLS].reindex(
            pd.date_range(f.index[0], periods=96, freq="15min"), method="ffill"
        )
        flags = {"trained_through": "2026-07-25", "recal_days": 300, "shaped": True}
        if backfill:
            flags["backfill"] = True
        parts.append(site.log_rows(hourly, quarters, flags, "2026-07-31T09:10:00+00:00"))
    return pd.concat(parts)


def _prices(lear, n_hours=48):
    prices = (lear["forecast"] + 5.0).iloc[:n_hours]
    prices_qh = prices.reindex(
        pd.date_range(prices.index[0], periods=len(prices) * 4, freq="15min"), method="ffill"
    )
    return prices, prices_qh


def _write(tmp_path, nets_days=("2026-08-01", "2026-08-02", "2026-08-03"), backfill=False):
    lear = _lear_log()
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    if nets_days:
        site.append_log(tmp_path / site.LOG_DIR, _nets_log(lear, nets_days, backfill))
    prices = (lear["forecast"] + 5.0).iloc[:48]  # the third day is tomorrow
    prices_qh = prices.reindex(
        pd.date_range(prices.index[0], periods=len(prices) * 4, freq="15min"), method="ffill"
    )
    write_site_json(tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    return latest, history


def test_latest_carries_the_nets_block_next_to_v1(tmp_path):
    latest, _ = _write(tmp_path)
    assert latest["schema"] == 2
    assert latest["levels"] == [1, 5, 10, 25, 50, 75, 90, 95, 99]
    assert latest["delivery_day"] == "2026-08-03"
    assert len(latest["hours"]) == 24 and "forecast" in latest["hours"][0]  # v1 = LEAR
    nets = latest["nets"]
    assert nets["model"].startswith("24 networks")
    assert nets["trained_through"] == "2026-07-25"
    assert len(nets["hours"]) == 24 and len(nets["quarters"]) == 96
    first = nets["hours"][0]
    assert first["t"] == "2026-08-03T00:00:00+00:00"
    assert first["actual"] is None  # not auctioned yet
    lear_first = latest["hours"][0]["forecast"]
    want = [round(lear_first + OFFSETS[lv - 1], 2) for lv in latest["levels"]]
    np.testing.assert_allclose(first["q"], want, atol=0.01)
    assert nets["quarters"][1]["t"] == "2026-08-03T00:15:00+00:00"


def test_history_days_carry_nets_scores(tmp_path):
    _, history = _write(tmp_path)
    assert [d["day"] for d in history["days"]] == ["2026-08-01", "2026-08-02"]
    for d in history["days"]:
        assert d["mae"] == 5.0  # v1: LEAR
        n = d["nets"]
        # q50 = LEAR's curve, actual = +5: MAE 5 hourly and on quarters
        assert n["mae"] == 5.0 and n["mae_qh"] == 5.0
        diff = 5.0 - OFFSETS
        taus = np.arange(1, 100) / 100
        want = np.maximum(taus * diff, (taus - 1) * diff).mean()
        assert n["pinball"] == round(want, 2) and n["pinball_qh"] == round(want, 2)
        for k in ("cov80", "cov90", "cov98", "cov80_qh", "cov90_qh", "cov98_qh"):
            assert n[k] == 1.0
        assert len(n["hours"]) == 24
        h = n["hours"][0]
        assert set(h) == {"t", "q1", "q5", "q10", "q50", "q90", "q95", "q99", "actual"}
        assert round(h["actual"] - h["q50"], 2) == 5.0
        assert round(h["q99"] - h["q50"], 2) == 20.0 and round(h["q1"] - h["q50"], 2) == -20.0


def test_coverage_bands_count_the_tails_separately():
    idx = pd.date_range("2026-08-01", periods=4, freq="1h", tz="UTC")
    rows = pd.DataFrame(np.tile(OFFSETS, (4, 1)), index=idx, columns=site.Q_COLS)
    # 17 is outside [q10, q90] = +-16.3, inside [q05, q95] = +-18.4; 19 only inside +-20
    a = pd.Series([0.0, 17.0, 19.0, 25.0], index=idx)
    s = site.day_scores(rows, rows.iloc[:0], a, pd.Series(dtype=float))
    assert (s["cov80"], s["cov90"], s["cov98"]) == (0.25, 0.5, 0.75)
    assert s["cov98_qh"] is None and s["mae_qh"] is None


def test_days_before_go_live_carry_no_nets_key(tmp_path):
    latest, history = _write(tmp_path, nets_days=("2026-08-02",))
    assert "nets" not in latest
    by_day = {d["day"]: d for d in history["days"]}
    assert "nets" not in by_day["2026-08-01"]
    assert "nets" in by_day["2026-08-02"]


def test_without_a_nets_log_v1_is_unchanged(tmp_path):
    latest, history = _write(tmp_path, nets_days=())
    assert "nets" not in latest
    assert all("nets" not in d for d in history["days"])
    assert latest["schema"] == 2  # additive keys only


def test_published_nets_survive_a_nets_log_loss(tmp_path):
    _write(tmp_path)
    (tmp_path / site.LOG_DIR).rename(tmp_path / "gone")
    _, history = _write(tmp_path, nets_days=())
    assert all(d["nets"]["mae"] == 5.0 for d in history["days"])


def test_nets_step_failure_is_contained(tmp_path, capsys):
    ok = nets_step(
        tmp_path / "missing.pt", tmp_path, pd.DataFrame(), pd.DataFrame(),
        pd.Timestamp("2026-08-03", tz="UTC"), tmp_path, pd.Series(dtype=float),
        "2026-08-02T09:00:00+00:00", "00Z",
    )  # fmt: skip
    assert ok is False
    assert "nets step failed" in capsys.readouterr().out
    assert not site.log_exists(tmp_path / site.LOG_DIR)


def test_log_keeps_the_last_run_per_period(tmp_path):
    lear = _lear_log(1)
    rows = _nets_log(lear, ["2026-08-01"])
    later = rows.copy()
    later[site.Q_COLS] += 1.0
    later["generated_utc"] = "2026-07-31T09:50:00+00:00"
    for path in (tmp_path / "l.parquet", tmp_path / site.LOG_DIR):
        site.append_log(path, rows)
        site.append_log(path, later)
        log = site.read_log(path)
        assert len(log) == 24 + 96
        assert (log["generated_utc"] == "2026-07-31T09:50:00+00:00").all()


# ------------------------------------------------------------------ partitioned log


def test_log_is_partitioned_by_month(tmp_path):
    lear = _lear_log(days=3, start="2026-08-30")
    site.append_log(tmp_path / site.LOG_DIR, _nets_log(lear, ["2026-08-30", "2026-08-31"]))
    site.append_log(tmp_path / site.LOG_DIR, _nets_log(lear, ["2026-09-01"]))
    parts = sorted(p.name for p in (tmp_path / site.LOG_DIR).glob("*.parquet"))
    assert parts == ["2026-08.parquet", "2026-09.parquet"]
    assert len(pd.read_parquet(tmp_path / site.LOG_DIR / "2026-09.parquet")) == 120
    assert len(site.read_log(tmp_path / site.LOG_DIR)) == 3 * 120


def test_a_single_file_log_is_migrated_once_and_left_in_place(tmp_path, capsys):
    lear = _lear_log(days=2, start="2026-08-31")
    old = _nets_log(lear, ["2026-08-31", "2026-09-01"])
    old.to_parquet(tmp_path / site.LOG_NAME)
    log_dir = tmp_path / site.LOG_DIR
    assert site.log_exists(log_dir)
    assert len(site.read_log(log_dir)) == 240
    assert "NOTICE" in capsys.readouterr().out
    assert len(list(log_dir.glob("*.parquet"))) == 2
    # new runs go to the partitions only; the old file stays as it was
    site.append_log(log_dir, _nets_log(_lear_log(1, "2026-09-02"), ["2026-09-02"]))
    assert len(site.read_log(log_dir)) == 360
    assert len(pd.read_parquet(tmp_path / site.LOG_NAME)) == 240
    assert "NOTICE" not in capsys.readouterr().out


# ------------------------------------------------------------------ replay flag


def test_replayed_rows_are_flagged_everywhere(tmp_path):
    latest, history = _write(tmp_path, backfill=True)
    assert latest["nets"]["replay"] is True
    assert all(d["nets"]["replay"] is True for d in history["days"])
    day = json.loads((tmp_path / "days" / "2026-08-01.json").read_text(encoding="utf-8"))
    assert day["nets"]["replay"] is True


def test_evening_substitutes_are_flagged_in_the_nets_blocks(tmp_path):
    from pred_el_prices.daily_forecast import nets_logged

    lear = _lear_log()
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    rows = _nets_log(lear, ["2026-08-01", "2026-08-02", "2026-08-03"])
    evening = rows.index.normalize() == pd.Timestamp("2026-08-03", tz="UTC")
    rows["evening"] = evening
    rows["load_surrogate"] = evening
    rows["neighbour_surrogate"] = evening
    site.append_log(tmp_path / site.LOG_DIR, rows)
    prices, prices_qh = _prices(lear)
    write_site_json(tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    n = latest["nets"]
    assert n["evening"] and n["load_surrogate"] and n["neighbour_surrogate"]
    day = json.loads((tmp_path / "days" / "2026-08-03.json").read_text(encoding="utf-8"))
    assert day["nets"]["neighbour_surrogate"] is True
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert all("neighbour_surrogate" not in d["nets"] for d in history["days"])
    # a standing evening nets forecast does not stop the morning retry
    assert not nets_logged(tmp_path, pd.Timestamp("2026-08-03", tz="UTC"))
    assert nets_logged(tmp_path, pd.Timestamp("2026-08-02", tz="UTC"))


def test_live_rows_carry_no_replay_flag(tmp_path):
    latest, history = _write(tmp_path)
    assert "replay" not in latest["nets"]
    assert all("replay" not in d["nets"] for d in history["days"])


# ------------------------------------------------------------------ day files


def test_day_files_mirror_latest_with_scores(tmp_path):
    latest, history = _write(tmp_path)
    files = sorted(p.name for p in (tmp_path / "days").glob("*.json"))
    assert files == ["2026-08-01.json", "2026-08-02.json", "2026-08-03.json"]
    today = json.loads((tmp_path / "days" / "2026-08-03.json").read_text(encoding="utf-8"))
    # the latest day's file is latest.json plus (still empty) scores
    for key in ("delivery_day", "hours", "levels", "schema", "pre_gate", "weather_vintage"):
        assert today[key] == latest[key]
    assert today["nets"]["quarters"] == latest["nets"]["quarters"]
    assert today["mae"] is None and today["nets"]["mae"] is None
    past = json.loads((tmp_path / "days" / "2026-08-01.json").read_text(encoding="utf-8"))
    entry = history["days"][0]
    assert past["mae"] == entry["mae"] == 5.0
    for k in ("mae", "pinball", "cov80", "mae_qh", "pinball_qh"):
        assert past["nets"][k] == entry["nets"][k]
    assert len(past["nets"]["hours"]) == 24 and len(past["nets"]["quarters"]) == 96
    assert past["nets"]["hours"][0]["actual"] is not None


def test_day_files_are_rewritten_only_when_they_change(tmp_path):
    _write(tmp_path)
    days = tmp_path / "days"
    stamps = {p.name: p.stat().st_mtime_ns for p in days.glob("*.json")}
    before = (days / "2026-08-03.json").read_text(encoding="utf-8")
    lear = pd.read_parquet(tmp_path / "forecast_log.parquet")
    prices, prices_qh = _prices(lear, 72)  # the auction result for 08-03 is in
    for p in days.glob("*.json"):  # backdate, so any rewrite shows in the mtime
        os.utime(p, ns=(stamps[p.name] - 10**9, stamps[p.name] - 10**9))
    write_site_json(tmp_path, tmp_path / "forecast_log.parquet", prices, prices_qh)
    after = {p.name: p.stat().st_mtime_ns for p in days.glob("*.json")}
    assert after["2026-08-01.json"] == stamps["2026-08-01.json"] - 10**9  # untouched
    assert (days / "2026-08-03.json").read_text(encoding="utf-8") != before
    today = json.loads((days / "2026-08-03.json").read_text(encoding="utf-8"))
    assert today["mae"] == 5.0 and today["nets"]["mae_qh"] == 5.0


def test_day_files_cover_history_only_days_and_are_never_deleted(tmp_path):
    (tmp_path / "days").mkdir()
    (tmp_path / "days" / "2026-01-01.json").write_text("{}", encoding="utf-8")
    curve = [{"t": "2026-07-28T00:00:00+00:00", "forecast": 1.0, "actual": 2.0}]
    entry = {"day": "2026-07-28", "mae": 1.0, "post_gate": True, "hours": curve}
    (tmp_path / "history.json").write_text(json.dumps({"days": [entry]}), encoding="utf-8")
    _write(tmp_path)
    old = json.loads((tmp_path / "days" / "2026-07-28.json").read_text(encoding="utf-8"))
    assert old["hours"] == curve and old["mae"] == 1.0 and old["pre_gate"] is False
    assert "nets" not in old
    assert (tmp_path / "days" / "2026-01-01.json").exists()


# ------------------------------------------------------------------ t0 and mix blocks

THREE = ("nets", "t0", "mix")
DAYS = ("2026-08-01", "2026-08-02", "2026-08-03")
RAW_T0 = {"t0_recal_days", "t0_uncalibrated"}  # t0/mix blocks, too little PIT history


def _t0_rows(lear: pd.DataFrame, day: str, generated="2026-07-31T09:12:00+00:00", **flags):
    """t0 for one day, as the t0 log holds it: median 10 above LEAR (the nets' median is
    LEAR itself), so the equal-weight mix's median sits 5 above LEAR, exactly on the
    synthetic actuals. Linear in the level: PCHIP puts it on the 99-level grid exactly."""
    from pred_el_prices.production.t0_shadow import LEVELS, Q_NAMES

    f = lear.loc[lear.index.normalize() == pd.Timestamp(day, tz="UTC"), "forecast"]
    taus = np.asarray(LEVELS)
    rows = pd.DataFrame(f.to_numpy()[:, None] + 10.0 + 40.0 * (taus - 0.5), f.index, Q_NAMES)
    rows.insert(0, "generated_utc", generated)
    for k, v in flags.items():
        rows[k] = v
    rows.index.name = "t"
    return rows


def _logs(tmp_path, t0_days=DAYS, mix_days=DAYS, nets_days=DAYS):
    """The _write fixture's logs plus a t0 log (`t0_days`) and a mix log (`mix_days`:
    with t0 where the day has t0, nets-only elsewhere). Returns the LEAR log."""
    from pred_el_prices.production import mix

    lear = _lear_log()
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    nets_rows = _nets_log(lear, nets_days)
    site.append_log(tmp_path / site.LOG_DIR, nets_rows)
    for day in t0_days:
        site._write_partitions(tmp_path / "t0_log", _t0_rows(lear, day))
    cal = mix.T0Calibration.from_dir(tmp_path, _prices(lear)[0])  # as mix_step logs it
    for day in mix_days:
        rows = nets_rows[nets_rows.index.normalize() == pd.Timestamp(day, tz="UTC")]
        t0 = _t0_rows(lear, day) if day in t0_days else None
        logged = mix.mix_rows(rows, t0, "2026-07-31T09:12Z", cal)
        site.append_log(tmp_path / site.MIX_LOG_DIR, logged)
    return lear


def _site(tmp_path, lear, **kw) -> dict:
    write_site_json(tmp_path, tmp_path / "forecast_log.parquet", *_prices(lear), **kw)
    return _read_site(tmp_path)


def _three(tmp_path, primary="mix", mix_from="2026-08-01", **logs) -> dict:
    lear = _logs(tmp_path, **logs)
    return _site(tmp_path, lear, site_models=THREE, primary=primary, mix_from=mix_from)


def _read_site(tmp_path) -> dict:
    """Every published file, as bytes."""
    files = [tmp_path / "latest.json", tmp_path / "history.json"]
    files += sorted((tmp_path / "days").glob("*.json"))
    return {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in files}


def _json(out: dict, name: str) -> dict:
    return json.loads(out[name])


@pytest.mark.parametrize("explicit", [True, False])
def test_default_site_models_ignore_the_t0_and_mix_logs_byte_for_byte(tmp_path, explicit):
    plain, more = tmp_path / "plain", tmp_path / "more"
    plain.mkdir()
    more.mkdir()
    _write(plain)
    lear = _logs(more)  # the same logs, plus a t0 log, a mix log and a t0 PIT seed
    seed = _t0_rows(lear, "2026-08-01").drop(columns="generated_utc")
    seed.set_axis(seed.index - pd.Timedelta(days=30)).to_parquet(more / site.T0_SEED_NAME)
    kw = {"site_models": ("nets",), "primary": "nets", "mix_from": None} if explicit else {}
    published = _site(more, lear, **kw)
    assert published == _read_site(plain)
    blob = b"".join(published.values())
    assert len(published) == 5
    assert b'"primary"' not in blob and b'"t0"' not in blob and b'"mix"' not in blob


def test_three_models_publish_sibling_blocks_of_the_nets_shape(tmp_path):
    plain, three = tmp_path / "plain", tmp_path / "three"
    plain.mkdir()
    three.mkdir()
    _write(plain)
    out = _three(three)
    latest = _json(out, "latest.json")
    assert latest["nets"] == _json(_read_site(plain), "latest.json")["nets"]  # as on main
    assert latest["primary"] == "mix"
    t0, mix = latest["t0"], latest["mix"]
    assert t0["model"] == site.T0_MODEL_LABEL and "trained_through" not in t0
    assert mix["model"] == site.MIX_MODEL_LABEL and mix["trained_through"] == "2026-07-25"
    for block in (t0, mix):
        assert len(block["hours"]) == 24 and len(block["quarters"]) == 96
    i50 = latest["levels"].index(50)
    lear0 = latest["hours"][0]["forecast"]
    assert round(t0["hours"][0]["q"][i50] - lear0, 2) == 10.0
    assert round(mix["hours"][0]["q"][i50] - lear0, 2) == 5.0  # (nets 0 + t0 10) / 2
    # the nets' quarters repeat their hour here, so t0's quarters repeat t0's hour
    assert t0["quarters"][1]["q"] == t0["hours"][0]["q"] and "flat_quarters" not in t0
    # the logged mix stands on the nets run: its run made the t0 too
    assert mix["nets_generated_utc"] == latest["nets"]["generated_utc"]
    assert mix["t0_generated_utc"] == mix["generated_utc"] == "2026-07-31T09:12Z"
    for d in _json(out, "history.json")["days"]:
        assert d["primary"] == "mix" and d["mae"] == 5.0  # v1 (LEAR) untouched
        assert d["nets"]["mae"] == 5.0 and d["t0"]["mae"] == 5.0 and d["mix"]["mae"] == 0.0
        assert d["t0"]["mae_qh"] == 5.0 and d["mix"]["mae_qh"] == 0.0
        assert set(d["t0"]) == set(d["mix"]) == set(d["nets"]) | RAW_T0
        assert d["t0"]["t0_recal_days"] == d["mix"]["t0_recal_days"] == 0  # 3 days of history
    day = _json(out, "days/2026-08-01.json")
    assert day["primary"] == "mix" and day["mix"]["mae"] == 0.0 and day["t0"]["pinball"]
    assert set(day["t0"]) == set(day["nets"]) - {"trained_through"} | RAW_T0
    assert set(day["mix"]) == set(day["nets"]) | {"nets_generated_utc", "t0_generated_utc"} | RAW_T0
    assert latest["t0"]["t0_uncalibrated"] is True and "t0_recal_days" not in latest["nets"]


def test_the_mix_is_mixed_at_site_build_time_without_a_mix_log(tmp_path):
    logged, built = tmp_path / "logged", tmp_path / "built"
    logged.mkdir()
    built.mkdir()
    a = _three(logged)
    b = _three(built, mix_days=())  # t0 log only: history back to its start
    for name in ("latest.json", "days/2026-08-01.json"):
        ma, mb = _json(a, name)["mix"], _json(b, name)["mix"]
        assert mb["hours"] == ma["hours"] and mb["quarters"] == ma["quarters"]
        assert mb["t0_generated_utc"] == "2026-07-31T09:12:00+00:00"
        assert mb["nets_generated_utc"] == "2026-07-31T09:10:00+00:00"
        assert mb["generated_utc"] == mb["t0_generated_utc"]  # the later of the two runs
        assert "mixed_editions" not in mb
    assert [d["mix"]["mae"] for d in _json(b, "history.json")["days"]] == [0.0, 0.0]


def test_a_day_without_t0_has_no_t0_or_mix_and_falls_back_to_the_nets(tmp_path):
    out = _three(tmp_path, t0_days=("2026-08-01",))
    latest = _json(out, "latest.json")
    assert "t0" not in latest and "mix" not in latest
    assert latest["primary"] == "nets" and latest["primary_fallback"] == "mix"
    by_day = {d["day"]: d for d in _json(out, "history.json")["days"]}
    first = by_day["2026-08-01"]
    assert first["primary"] == "mix" and "primary_fallback" not in first
    assert by_day["2026-08-02"]["primary"] == "nets" and "mix" not in by_day["2026-08-02"]
    day = _json(out, "days/2026-08-02.json")
    assert day["primary"] == "nets" and day["primary_fallback"] == "mix"


def test_days_before_the_cut_over_stay_nets(tmp_path):
    out = _three(tmp_path, mix_from="2026-08-02")
    by_day = {d["day"]: d for d in _json(out, "history.json")["days"]}
    assert by_day["2026-08-01"]["primary"] == "nets" and "mix" in by_day["2026-08-01"]
    assert "primary_fallback" not in by_day["2026-08-01"]
    assert by_day["2026-08-02"]["primary"] == "mix"
    assert _json(out, "days/2026-08-01.json")["primary"] == "nets"


def test_primary_t0(tmp_path):
    out = _three(tmp_path, primary="t0", t0_days=("2026-08-01", "2026-08-03"))
    assert _json(out, "latest.json")["primary"] == "t0"
    by_day = {d["day"]: d for d in _json(out, "history.json")["days"]}
    assert by_day["2026-08-02"]["primary_fallback"] == "t0"


def test_a_mix_across_editions_is_flagged(tmp_path):
    """2026-10-25: the morning nets failed; the evening nets stood next to the morning t0."""
    lear = _logs(tmp_path, mix_days=())
    rows = _nets_log(lear, ["2026-08-03"])
    rows["evening"] = True
    rows["generated_utc"] = "2026-07-31T21:51:00+00:00"
    site.append_log(tmp_path / site.LOG_DIR, rows)
    site._write_partitions(tmp_path / "t0_log", _t0_rows(lear, "2026-08-03", evening=False))
    out = _site(tmp_path, lear, site_models=THREE, primary="mix", mix_from="2026-08-01")
    mix = _json(out, "latest.json")["mix"]
    assert mix["mixed_editions"] is True and mix["evening"] is True
    assert mix["nets_generated_utc"] == "2026-07-31T21:51:00+00:00"
    assert mix["generated_utc"] == mix["nets_generated_utc"]  # the later run
    assert "evening" not in _json(out, "latest.json")["t0"]


def test_a_mix_logged_on_an_older_nets_run_is_mixed_anew(tmp_path):
    lear = _logs(tmp_path)
    rerun = _nets_log(lear, ["2026-08-03"])
    rerun["generated_utc"] = "2026-08-02T09:50:00+00:00"  # the mix step missed this run
    site.append_log(tmp_path / site.LOG_DIR, rerun)
    out = _site(tmp_path, lear, site_models=THREE, primary="mix", mix_from="2026-08-01")
    mix = _json(out, "latest.json")["mix"]
    assert mix["nets_generated_utc"] == "2026-08-02T09:50:00+00:00"
    assert mix["t0_generated_utc"] == "2026-07-31T09:12:00+00:00"


def test_t0_without_nets_has_flat_quarters_and_no_mix(tmp_path):
    out = _three(tmp_path, nets_days=("2026-08-01", "2026-08-02"), mix_days=())
    latest = _json(out, "latest.json")
    assert "nets" not in latest and "mix" not in latest
    assert latest["t0"]["flat_quarters"] is True
    assert latest["t0"]["quarters"][3]["q"] == latest["t0"]["hours"][0]["q"]
    assert "primary" not in latest  # no nets to fall back to, nor the model wanted


def test_published_t0_and_mix_blocks_survive_a_log_loss(tmp_path):
    _three(tmp_path)
    (tmp_path / "t0_log").rename(tmp_path / "gone")
    (tmp_path / site.MIX_LOG_DIR).rename(tmp_path / "gone_too")
    lear = pd.read_parquet(tmp_path / "forecast_log.parquet")
    out = _site(tmp_path, lear, site_models=THREE, primary="mix", mix_from="2026-08-01")
    for d in _json(out, "history.json")["days"]:
        assert d["mix"]["mae"] == 0.0 and d["t0"]["mae"] == 5.0 and d["primary"] == "mix"
    # back on nets alone, the site is the nets site again
    out = _site(tmp_path, lear)
    days = _json(out, "history.json")["days"]
    assert all(set(d) & {"t0", "mix", "primary"} == set() for d in days)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"site_models": THREE, "primary": "mix"}, "mix_from"),
        # mix with the nets block alone is the published mix (test_published_mix.py)
        (
            {"site_models": ("nets", "t0"), "primary": "mix", "mix_from": "2026-08-01"},
            "site models",
        ),
        ({"primary": "t0", "mix_from": "2026-08-01"}, "site models"),
        ({"site_models": ("t0",)}, "site_models"),
        ({"site_models": ("nets", "lear")}, "site_models"),
    ],
)
def test_inconsistent_site_settings_are_refused(tmp_path, kw, match):
    lear = _lear_log()
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    with pytest.raises(ValueError, match=match):
        write_site_json(tmp_path, tmp_path / "forecast_log.parquet", *_prices(lear), **kw)


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        ([], (("nets",), "nets", None)),
        (
            ["--site-models", "mix, nets,t0", "--primary", "mix", "--mix-from", "2026-11-05"],
            (THREE, "mix", "2026-11-05"),
        ),
        (["--site-models", "nets,t0"], (("nets", "t0"), "nets", None)),
        # the workflow default: the nets + t0-all mix inside the nets block
        (["--primary", "mix", "--mix-from", "2026-11-12"], (("nets",), "mix", "2026-11-12")),
    ],
)
def test_cli_passes_the_site_settings_through(monkeypatch, argv, want):
    import sys

    from pred_el_prices import cli, daily_forecast

    seen = {}
    monkeypatch.setattr(daily_forecast, "run_daily", lambda **k: seen.update(k))
    monkeypatch.setattr(sys, "argv", ["pep", "forecast", *argv])
    cli.main()
    assert (seen["site_models"], seen["primary"], seen["mix_from"]) == want


@pytest.mark.parametrize(
    "extra",
    [
        ["--site-models", "nets,t0,mix", "--primary", "mix"],
        ["--site-models", "nets,t0,mix", "--primary", "mix", "--mix-from", "next week"],
        ["--site-models", "nets,t0", "--primary", "mix", "--mix-from", "2026-11-05"],
        ["--primary", "t0", "--mix-from", "2026-11-05"],  # t0 not among the site models
        ["--primary", "mix"],  # the published mix needs its cut-over day
        ["--site-models", "t0,mix"],
        ["--site-models", "nets,lear"],
    ],
)
def test_cli_refuses_inconsistent_site_settings(monkeypatch, extra):
    import sys

    from pred_el_prices import cli, daily_forecast

    monkeypatch.setattr(daily_forecast, "run_daily", lambda **k: pytest.fail("ran"))
    monkeypatch.setattr(sys, "argv", ["pep", "forecast", *extra])
    with pytest.raises(SystemExit):
        cli.main()


def test_build_site_rebuilds_from_existing_logs(tmp_path):
    pytest.importorskip("torch")  # the backfill module loads the network code
    from pred_el_prices.pipeline import cache
    from pred_el_prices.production.backfill import build_site

    lear = _lear_log()
    src = tmp_path / "src"
    src.mkdir()
    lear.to_parquet(src / "forecast_log.parquet")
    _nets_log(lear, ["2026-08-01", "2026-08-02", "2026-08-03"], True).to_parquet(
        src / site.LOG_NAME
    )
    prices, _ = _prices(lear, 72)
    cache.upsert(tmp_path / "cache", "entsoe/day_ahead_prices", prices.to_frame("price_eur_mwh"))
    out = tmp_path / "site"
    later = {"day": "2026-08-03", "mae": 9.0, "hours": [{"t": "x", "forecast": 1, "actual": 2}]}
    (src / "history.json").write_text(json.dumps({"days": [later]}), encoding="utf-8")
    build_site(src / site.LOG_NAME, src / "forecast_log.parquet", tmp_path / "cache", out,
               src / "history.json", end="2026-08-02")  # fmt: skip
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert "2026-08-03" not in [d["day"] for d in history["days"]]  # nothing after `end`
    latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
    assert latest["delivery_day"] == "2026-08-02" and latest["nets"]["replay"] is True
    names = sorted(p.name for p in (out / "days").glob("*.json"))
    assert names == ["2026-08-01.json", "2026-08-02.json"]
    assert len(list((out / site.LOG_DIR).glob("*.parquet"))) == 1
    build_site(src / site.LOG_NAME, src / "forecast_log.parquet", tmp_path / "cache", out,
               end="2026-08-02")  # rerun: partitions replaced, not doubled  # fmt: skip
    assert len(pd.read_parquet(out / site.LOG_DIR / "2026-08.parquet")) == 2 * 120
