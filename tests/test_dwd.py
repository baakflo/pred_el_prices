from datetime import date

import numpy as np

from pred_el_prices.pipeline import dwd


def test_single_level_url() -> None:
    url = dwd.single_level_url(date(2026, 7, 30), 24, "u_10m")
    assert url == (
        "https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib/00/u_10m/"
        "icon-eu-eps_europe_icosahedral_single-level_2026073000_024_u_10m.grib2.bz2"
    )


def test_as_degrees_converts_radians() -> None:
    rad = np.array([0.9, -0.5])
    np.testing.assert_allclose(dwd._as_degrees(rad), np.degrees(rad))
    deg = np.array([51.0, 48.5])
    np.testing.assert_allclose(dwd._as_degrees(deg), deg)


def test_aggregate_members_cell_means() -> None:
    # two points in cell (50, 10), one in (48, 8), one outside Germany
    lat = np.array([50.2, 50.8, 48.5, 40.0])
    lon = np.array([10.1, 10.9, 8.2, 10.0])
    fields = np.array(
        [
            [1.0, 3.0, 5.0, 99.0],  # member 0
            [2.0, 4.0, 6.0, 99.0],  # member 1
        ]
    )
    out = dwd.aggregate_members(lat, lon, fields)

    assert len(out) == 4  # 2 cells x 2 members; outside point dropped
    m0_cell_50_10 = out.query("cell_lat == 50 and cell_lon == 10 and member == 0")["value"]
    assert m0_cell_50_10.item() == 2.0  # mean of 1.0 and 3.0
    m1_cell_48_8 = out.query("cell_lat == 48 and cell_lon == 8 and member == 1")["value"]
    assert m1_cell_48_8.item() == 6.0
