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
