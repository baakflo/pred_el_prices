"""Daily archiver for DWD ICON-EU-EPS ensemble forecasts over Germany.

DWD's open-data server keeps only ~24 h of files, so this must run every day;
a missed day is unrecoverable. The 00 UTC run is the freshest ensemble
reliably published before the 12:00 CET day-ahead auction on D-1. Each run's
~1.4 GB of GRIB is distilled to per-member means over 1-degree cells covering
Germany (~2 MB Parquet/day), keeping enough spatial structure to
capacity-weight wind/solar regions later.
"""

from __future__ import annotations

import bz2
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-eu-eps/grib/00"

# Wind (10 m; no hub-height wind in the EPS products), direct+diffuse solar
# radiation, and 2 m temperature.
VARIABLES = ["u_10m", "v_10m", "aswdir_s", "aswdifd_s", "t_2m"]

# Steps +21h..+48h from the 00Z run of D-1 cover the delivery day D fully in
# UTC and in CET/CEST.
STEPS = range(21, 49)

# 1-degree cells covering Germany; cells are labeled by their lower-left
# corner. The box overlaps neighbours slightly - downstream weighting decides
# what counts.
LAT_MIN, LAT_MAX = 47.0, 56.0
LON_MIN, LON_MAX = 5.0, 16.0


def single_level_url(run_date: date, step: int, var: str) -> str:
    stamp = f"{run_date:%Y%m%d}00"
    return (
        f"{BASE_URL}/{var}/"
        f"icon-eu-eps_europe_icosahedral_single-level_{stamp}_{step:03d}_{var}.grib2.bz2"
    )


def invariant_url(run_date: date, var: str) -> str:
    stamp = f"{run_date:%Y%m%d}00"
    return (
        f"{BASE_URL}/{var}/"
        f"icon-eu-eps_europe_icosahedral_time-invariant_{stamp}_{var}.grib2.bz2"
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, max=60), reraise=True)
def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return bz2.decompress(resp.content)


def _read_field(raw_grib: bytes, tmp_dir: Path) -> np.ndarray:
    """Return the single data variable of a GRIB message as a numpy array."""
    path = tmp_dir / "current.grib2"
    path.write_bytes(raw_grib)
    # indexpath="" stops cfgrib littering .idx sidecar files
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        (name,) = ds.data_vars
        return ds[name].values


def _as_degrees(coords: np.ndarray) -> np.ndarray:
    """DWD grid-coordinate files may encode radians; normalize to degrees."""
    return np.degrees(coords) if np.abs(coords).max() <= np.pi else coords


def aggregate_members(lat: np.ndarray, lon: np.ndarray, fields: np.ndarray) -> pd.DataFrame:
    """Mean per ensemble member over 1-degree cells covering Germany.

    lat/lon: (points,) coordinates of the icosahedral grid cells (degrees).
    fields: (members, points) values on that grid.
    Returns columns: cell_lat, cell_lon, member, value.
    """
    mask = (lat >= LAT_MIN) & (lat < LAT_MAX) & (lon >= LON_MIN) & (lon < LON_MAX)
    cell_lat = np.floor(lat[mask])
    cell_lon = np.floor(lon[mask])
    n_members, n_points = fields.shape[0], int(mask.sum())
    df = pd.DataFrame(
        {
            "cell_lat": np.tile(cell_lat, n_members),
            "cell_lon": np.tile(cell_lon, n_members),
            "member": np.repeat(np.arange(n_members), n_points),
            "value": fields[:, mask].ravel(),
        }
    )
    return df.groupby(["cell_lat", "cell_lon", "member"], as_index=False)["value"].mean()


def archive_run(run_date: date, archive_dir: Path) -> Path:
    """Download and aggregate one 00Z ICON-EU-EPS run. Idempotent."""
    out = (
        archive_dir
        / "icon-eu-eps"
        / f"{run_date:%Y}"
        / f"icon-eu-eps_{run_date:%Y%m%d}00.parquet"
    )
    if out.exists():
        print(f"already archived: {out}")
        return out

    run_time = datetime(run_date.year, run_date.month, run_date.day, tzinfo=UTC)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        lat = _as_degrees(_read_field(_download(invariant_url(run_date, "clat")), tmp_dir))
        lon = _as_degrees(_read_field(_download(invariant_url(run_date, "clon")), tmp_dir))
        for var in VARIABLES:
            for step in STEPS:
                fields = _read_field(_download(single_level_url(run_date, step, var)), tmp_dir)
                df = aggregate_members(lat, lon, fields)
                df["variable"] = var
                df["valid_time"] = run_time + timedelta(hours=step)
                frames.append(df)
            print(f"{var}: {len(STEPS)} steps done")

    result = pd.concat(frames, ignore_index=True)
    result["run_time"] = run_time
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"archived {len(result)} rows -> {out}")
    return out
