from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pred_el_prices.pipeline import dwd_det


def test_single_level_url_icon_d2() -> None:
    url = dwd_det.single_level_url("icon-d2", date(2026, 9, 24), 0, 21, "u_10m")
    assert url == (
        "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/u_10m/"
        "icon-d2_germany_regular-lat-lon_single-level_2026092400_021_2d_u_10m.grib2.bz2"
    )


def test_single_level_url_icon_eu() -> None:
    url = dwd_det.single_level_url("icon-eu", date(2026, 9, 24), 3, 18, "aswdir_s")
    assert url == (
        "https://opendata.dwd.de/weather/nwp/icon-eu/grib/03/aswdir_s/"
        "icon-eu_europe_regular-lat-lon_single-level_2026092403_018_ASWDIR_S.grib2.bz2"
    )


def test_model_level_url_icon_d2() -> None:
    url = dwd_det.model_level_url("icon-d2", date(2026, 9, 24), 0, 21, "u", 64)
    assert url == (
        "https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/u/"
        "icon-d2_germany_regular-lat-lon_model-level_2026092400_021_64_u.grib2.bz2"
    )


def test_model_level_url_icon_eu() -> None:
    url = dwd_det.model_level_url("icon-eu", date(2026, 9, 24), 0, 21, "v", 72)
    assert url == (
        "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/v/"
        "icon-eu_europe_regular-lat-lon_model-level_2026092400_021_72_V.grib2.bz2"
    )


def test_aggregate_field_cell_means_and_drops_row() -> None:
    # two points in cell (50, 10), one in (48, 8), one outside the box
    lat = np.array([50.2, 50.8, 48.5, 40.0])
    lon = np.array([10.1, 10.9, 8.2, 10.0])
    values = np.array([1.0, 3.0, 5.0, 99.0])

    df = dwd_det._aggregate_field(lat, lon, values, dwd_det.MODELS["icon-d2"].bbox)

    assert list(df.columns) == ["cell_lat", "cell_lon", "value"]
    assert len(df) == 2  # 2 cells; outside point dropped
    cell_50_10 = df.query("cell_lat == 50 and cell_lon == 10")["value"]
    assert cell_50_10.item() == 2.0  # mean of 1.0 and 3.0


def test_eu_bbox_is_superset_of_germany_bbox() -> None:
    from pred_el_prices.pipeline.dwd import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

    # A single aggregation pass over the ICON-EU box must reproduce the same
    # Germany cells as the EPS/ICON-D2 box, so the two are not double-defined.
    lat_min, lat_max, lon_min, lon_max = dwd_det.MODELS["icon-eu"].bbox
    assert lat_min <= LAT_MIN and lat_max >= LAT_MAX
    assert lon_min <= LON_MIN and lon_max >= LON_MAX


def test_archive_run_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dwd_det, "SINGLE_LEVEL_VARS", ["t_2m"])
    monkeypatch.setattr(dwd_det, "STEPS", {("icon-d2", 0): [21, 22]})
    monkeypatch.setattr(
        dwd_det,
        "MODELS",
        {
            "icon-d2": dwd_det.ModelSpec(
                domain="germany",
                upper_var=False,
                single_level_tag="2d",
                bbox=(47.0, 56.0, 5.0, 16.0),
                wind_levels={64: 37},
            )
        },
    )
    calls: list[str] = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        return b""

    def fake_read_grid_field(raw: bytes, tmp_dir: Path):
        lat = np.array([50.2, 50.8])
        lon = np.array([10.1, 10.9])
        values = np.array([1.0, 3.0])
        return lat, lon, values

    monkeypatch.setattr(dwd_det, "_download", fake_download)
    monkeypatch.setattr(dwd_det, "_read_grid_field", fake_read_grid_field)

    out = dwd_det.archive_run("icon-d2", date(2026, 9, 24), 0, tmp_path)
    assert out == tmp_path / "icon-d2" / "2026" / "icon-d2_2026092400.parquet"

    df = pd.read_parquet(out)
    # t_2m: 2 steps x 1 cell; u/v @ level 64: 2 steps x 2 vars x 1 cell
    assert len(df) == 6
    assert set(df.variable) == {"t_2m", "u", "v"}
    assert df.value.dtype == np.float32

    wind_rows = df[df.variable == "u"]
    assert (wind_rows.level == 64).all()
    assert (wind_rows.height_m == 37).all()
    surface_rows = df[df.variable == "t_2m"]
    assert surface_rows.level.isna().all()
    assert surface_rows.height_m.isna().all()

    n_calls = len(calls)
    out2 = dwd_det.archive_run("icon-d2", date(2026, 9, 24), 0, tmp_path)
    assert out2 == out
    assert len(calls) == n_calls  # second run: already archived, no downloads


def test_archive_today_skips_failed_combo_and_continues(tmp_path, monkeypatch) -> None:
    def fake_archive_run(model, run_date, run_hour, archive_dir):
        if model == "icon-eu":
            raise RuntimeError("run not published yet")
        return archive_dir / f"{model}.parquet"

    monkeypatch.setattr(dwd_det, "archive_run", fake_archive_run)

    written = dwd_det.archive_today(tmp_path, date(2026, 9, 24), runs=(0,))

    assert len(written) == 1
    assert written[0].name == "icon-d2.parquet"
