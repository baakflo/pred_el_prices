"""Offline tests for the DE_LU control-area step-in (synthetic data, no network)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pred_el_prices.features.dataset import build_dataset
from pred_el_prices.features.tso_areas import (
    CONTROL_AREAS,
    entsoe_de_series,
    fill_de_forecast,
    fill_from_areas,
)
from pred_el_prices.pipeline import cache

SHARES = {"DE_50HZ": 0.4, "DE_AMPRION": 0.3, "DE_TENNET": 0.2, "DE_TRANSNET": 0.08}  # LU 2 %
N_DAYS = 40


def _index(n_days=N_DAYS, freq="1h"):
    return pd.date_range("2026-09-01", periods=n_days * 24 * (4 if freq == "15min" else 1),
                         freq=freq, tz="UTC")  # fmt: skip


def _wind(idx):
    return pd.Series(20000 + 8000 * np.sin(np.arange(len(idx)) / 17.0), index=idx)


def _solar(idx):
    h = idx.hour + idx.minute / 60
    return pd.Series(np.clip(30000 * np.sin((h - 5) / 14 * np.pi), 0, None), index=idx)


def _areas(de):
    return pd.DataFrame({a: de * s for a, s in SHARES.items()})


def _last_day(idx):
    return idx >= idx[-1].normalize()


def test_fills_only_missing_hours_with_a_reporting_area():
    idx = _index()
    de = _wind(idx)
    areas = _areas(de)
    target = de.copy()
    last = _last_day(idx)
    target[last] = np.nan
    areas.loc[last, "DE_TRANSNET"] = np.nan  # the 2026-10-24 pattern
    no_area = idx[last][5]
    areas.loc[no_area] = np.nan  # nobody reported this hour: no fill
    filled, info = fill_from_areas(target, areas, per_hour=False)
    rest = last & (idx != no_area)
    np.testing.assert_allclose(filled[rest], de[rest], rtol=1e-9)
    assert np.isnan(filled[no_area])
    assert info["filled"] == 23
    assert info["missing_sets"] == {"DE_TRANSNET": 23}
    pd.testing.assert_series_equal(filled[~last], target[~last])


def test_never_overwrites_a_ladder_value():
    idx = _index()
    de = _wind(idx)
    areas = _areas(de) * 1.3  # areas disagree with the aggregate
    target = de.copy()
    target.iloc[-3:] = np.nan
    filled, info = fill_from_areas(target, areas, per_hour=False)
    pd.testing.assert_series_equal(filled.iloc[:-3], target.iloc[:-3])
    assert info["filled"] == 3


@pytest.mark.parametrize("per_hour", [True, False])
def test_every_subset_of_missing_areas_is_reconstructed(per_hour):
    idx = _index()
    de = _solar(idx) + 1000.0
    last = _last_day(idx)
    for k in (1, 2, 3):
        for missing in itertools.combinations(CONTROL_AREAS, k):
            areas = _areas(de)
            areas.loc[last, list(missing)] = np.nan
            target = de.where(~last)
            filled, info = fill_from_areas(target, areas, per_hour=per_hour)
            np.testing.assert_allclose(filled[last], de[last], rtol=1e-9)
            assert info["missing_sets"] == {"+".join(missing): 24}


def test_guard_too_few_complete_days_leaves_nan():
    idx = _index(n_days=12)
    de = _wind(idx)
    areas = _areas(de)
    target = de.where(~_last_day(idx))
    filled, info = fill_from_areas(target, areas, per_hour=False)
    assert filled[_last_day(idx)].isna().all()
    assert info == {"filled": 0, "guarded": 24, "capped": 0, "missing_sets": {}}


def test_cap_refuses_a_fill_from_too_small_a_share():
    idx = _index()
    de = _wind(idx)
    last = _last_day(idx)
    areas = _areas(de)
    areas.loc[last, ["DE_50HZ", "DE_AMPRION"]] = np.nan  # reporting share 0.28
    target = de.where(~last)
    filled, info = fill_from_areas(target, areas, per_hour=False, min_reported_share=0.4)
    assert filled[last].isna().all()
    assert info["capped"] == 24 and info["filled"] == 0
    areas.loc[last, "DE_AMPRION"] = de[last] * SHARES["DE_AMPRION"]  # share 0.58: fills
    filled, info = fill_from_areas(target, areas, per_hour=False, min_reported_share=0.4)
    np.testing.assert_allclose(filled[last], de[last], rtol=1e-9)
    assert info["capped"] == 0


def test_incomplete_history_days_do_not_count():
    idx = _index()
    de = _wind(idx)
    areas = _areas(de)
    # TransnetBW lacks one hour on every day: no complete day for a set containing it
    areas.loc[idx.hour == 3, "DE_TRANSNET"] = np.nan
    target = de.where(~_last_day(idx))
    _, info = fill_from_areas(target, areas, per_hour=False)
    assert info["guarded"] == 23  # hours where TransnetBW reported: R incl. it, no history
    assert info["filled"] == 1  # hour 3: R = the other three, complete days exist


def test_night_solar_is_zero_only_when_reporters_say_zero():
    idx = _index()
    de = _solar(idx)
    areas = _areas(de)
    last = _last_day(idx)
    areas.loc[last, "DE_TRANSNET"] = np.nan
    target = de.where(~last)
    night = last & (idx.hour == 2)
    areas.loc[night, "DE_TENNET"] = 50.0  # a reporter says non-zero at night
    filled, _ = fill_from_areas(target, areas, per_hour=True)
    dark = last & (de == 0) & ~night
    assert (filled[dark] == 0.0).all()
    assert filled[night].isna().all()  # 0/0 share and a non-zero report: no guess
    day = last & (de > 0)
    np.testing.assert_allclose(filled[day], de[day], rtol=1e-9)


def test_absent_rows_count_as_missing_and_quarter_hours_work(tmp_path):
    idx = _index(freq="15min")
    de = _wind(idx)
    for a, s in SHARES.items():
        cache.upsert(tmp_path, f"entsoe/{a}/load_forecast", (de * s).to_frame("Forecasted Load"))
    agg = de.to_frame("Forecasted Load")
    cache.upsert(tmp_path, "entsoe/load_forecast", agg[idx < idx[-1].normalize()])
    s = entsoe_de_series(tmp_path, "load_forecast_mw", "15min")
    last = _last_day(idx)
    np.testing.assert_allclose(s.reindex(idx[last]), de[last], rtol=1e-9)
    _, info = fill_de_forecast(tmp_path, "load_forecast_mw", de[~last].resample("1h").mean(), "1h")
    assert info["filled"] == 24


def test_control_area_refresh_is_best_effort(tmp_path, monkeypatch):
    import requests

    from pred_el_prices.daily_forecast import AREA_SEED_DAYS, fetch_control_areas
    from pred_el_prices.pipeline import entsoe as entsoe_pipeline

    idx = pd.date_range("2026-09-01", periods=24, freq="1h", tz="UTC")
    cache.upsert(
        tmp_path, "entsoe/DE_50HZ/load_forecast", pd.DataFrame({"Forecasted Load": 1.0}, idx)
    )
    calls = []

    def fake_backfill(client, datasets, start, end, cache_root, sleep_s=0.5, zone=None):
        calls.append((zone, datasets[0], start))
        if zone == "DE_AMPRION":
            raise ValueError("entsoe-py parse error")  # skips just this area
        if zone == "DE_TRANSNET":
            raise requests.ConnectionError("platform down")  # ends the sweep

    monkeypatch.setattr(entsoe_pipeline, "backfill", fake_backfill)
    end = pd.Timestamp("2026-11-03", tz="UTC")
    done = fetch_control_areas(object(), tmp_path, end)  # must not raise
    assert done == ["DE_50HZ", "DE_TENNET"]
    # a seeded cache resumes from its tail; an unseeded one seeds only recent days
    assert calls[0] == ("DE_50HZ", "load_forecast", pd.Timestamp("2015-01-01", tz="UTC"))
    assert calls[1] == ("DE_50HZ", "wind_solar_forecast", end - pd.Timedelta(days=AREA_SEED_DAYS))
    assert [c[0] for c in calls] == ["DE_50HZ", "DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TENNET",
                                     "DE_TRANSNET"]  # fmt: skip


def test_build_dataset_counts_area_fills(tmp_path):
    idx = _index()
    last = _last_day(idx)
    cache.upsert(
        tmp_path, "entsoe/day_ahead_prices", pd.DataFrame({"price_eur_mwh": 80.0}, index=idx)
    )
    cache.upsert(tmp_path, "entsoe/load_forecast", pd.DataFrame({"Forecasted Load": 5e4}, idx))
    solar, onshore = _solar(idx), _wind(idx)
    ws = pd.DataFrame({"Solar": solar, "Wind Onshore": onshore, "Wind Offshore": 3000.0})
    ws.loc[last, ["Solar", "Wind Onshore"]] = np.nan  # the incident: rows present, NaN
    cache.upsert(tmp_path, "entsoe/wind_solar_forecast", ws)
    sm = pd.DataFrame(
        {
            "wind_onshore_forecast_mw": ws["Wind Onshore"],
            "wind_offshore_forecast_mw": 3000.0,
            "solar_forecast_mw": ws["Solar"],
        }
    )
    cache.upsert(tmp_path, "smard_wind_solar", sm)  # SMARD mirrors the hole
    cache.upsert(tmp_path, "smard_load", pd.DataFrame({"load_forecast_mw": 5e4}, idx))
    for a, s in SHARES.items():
        area = pd.DataFrame({"Solar": solar * s, "Wind Onshore": onshore * s})
        if a == "DE_TRANSNET":
            area = area[~last]
        cache.upsert(tmp_path, f"entsoe/{a}/wind_solar_forecast", area)
    df, summary = build_dataset(tmp_path)
    p = summary["forecast_patches"]
    assert p["solar_forecast_mw"]["filled_from_areas"] == 24
    assert p["solar_forecast_mw"]["areas_missing"] == {"DE_TRANSNET": 24}
    assert p["wind_onshore_forecast_mw"]["filled_from_areas"] == 24
    assert p["wind_onshore_forecast_mw"]["still_missing"] == 0
    # the areas report no offshore here, and the load had no gap: untouched
    assert p["wind_offshore_forecast_mw"]["filled_from_areas"] == 0
    assert p["load_forecast_mw"]["filled_from_areas"] == 0
    np.testing.assert_allclose(df.loc[last, "wind_onshore_forecast_mw"], onshore[last])
    np.testing.assert_allclose(df.loc[last, "solar_forecast_mw"], solar[last], atol=1e-6)
