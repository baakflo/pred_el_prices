"""Site data contract v2: the nets block next to the v1 (LEAR) fields (synthetic only)."""

import json

import numpy as np
import pandas as pd

from pred_el_prices.daily_forecast import nets_step, write_site_json
from pred_el_prices.production import site

OFFSETS = np.linspace(-20, 20, 99)


def _lear_log(days=3, start="2026-08-01", generated="2026-08-02T09:00:00+00:00"):
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    df = pd.DataFrame({"forecast": np.linspace(50, 150, len(idx))}, index=idx)
    df["generated_utc"] = generated
    return df


def _nets_log(lear: pd.DataFrame, days) -> pd.DataFrame:
    parts = []
    for day in days:
        f = lear.loc[lear.index.normalize() == pd.Timestamp(day, tz="UTC"), "forecast"]
        hourly = pd.DataFrame(f.to_numpy()[:, None] + OFFSETS, index=f.index, columns=site.Q_COLS)
        hourly[site.R_COLS] = hourly[site.Q_COLS].to_numpy() - 1.0
        quarters = hourly[site.Q_COLS].reindex(
            pd.date_range(f.index[0], periods=96, freq="15min"), method="ffill"
        )
        flags = {"trained_through": "2026-07-25", "recal_days": 300, "shaped": True}
        parts.append(site.log_rows(hourly, quarters, flags, "2026-07-31T09:10:00+00:00"))
    return pd.concat(parts)


def _write(tmp_path, nets_days=("2026-08-01", "2026-08-02", "2026-08-03")):
    lear = _lear_log()
    lear.to_parquet(tmp_path / "forecast_log.parquet")
    if nets_days:
        site.append_log(tmp_path / site.LOG_NAME, _nets_log(lear, nets_days))
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
        assert n["cov80"] == 1.0
        assert len(n["hours"]) == 24
        h = n["hours"][0]
        assert set(h) == {"t", "q10", "q50", "q90", "actual"}
        assert round(h["actual"] - h["q50"], 2) == 5.0


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
    (tmp_path / site.LOG_NAME).rename(tmp_path / "gone.parquet")
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
    assert not (tmp_path / site.LOG_NAME).exists()


def test_log_keeps_the_last_run_per_period(tmp_path):
    lear = _lear_log(1)
    rows = _nets_log(lear, ["2026-08-01"])
    later = rows.copy()
    later[site.Q_COLS] += 1.0
    later["generated_utc"] = "2026-07-31T09:50:00+00:00"
    site.append_log(tmp_path / "l.parquet", rows)
    site.append_log(tmp_path / "l.parquet", later)
    log = site.read_log(tmp_path / "l.parquet")
    assert len(log) == 24 + 96
    assert (log["generated_utc"] == "2026-07-31T09:50:00+00:00").all()
