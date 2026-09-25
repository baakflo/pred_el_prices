"""Daily archiver for ICON-D2 and ICON-EU deterministic NWP runs (DWD open data).

Alongside the ICON-EU-EPS ensemble (dwd.py), these two higher-resolution
deterministic runs add: wind near hub height (there is no clean "100 m wind"
product on either model - both resolve the boundary layer with several thin
levels instead, so the two lowest model levels above the 10 m diagnostic
level are archived, with their approximate AGL height recorded per row -
see WIND_LEVELS), direct+diffuse solar radiation, 2 m temperature and total
cloud cover. Same ~24 h DWD retention as the EPS archiver: a missed day is
unrecoverable.

Both models publish every 3 h; only the 00Z and 03Z runs are archived here
(the 00Z run is the EPS-equivalent primary; 03Z is a fresher same-morning
backup). ICON-D2 (~2.2 km, German domain) is fully hourly out to +48 h on
both runs. ICON-EU (~7 km, European domain) is hourly to +48 h on the 00Z
run, but its off-hour runs (03/09/15/21Z) coarsen to 6-hourly past +30 h
(missing +33/+39/+45) - STEPS below reflects that by using +48 (the nearest
available step) instead of the nominal +45, so ICON-EU 03Z still covers the
full delivery-day window, just with 3 h of extra look-ahead at the top.

Radiation (aswdir_s, aswdifd_s) is stored as published: ICON's value at step t
is the mean flux since run start (W/m2), not an instantaneous one. Hourly flux
for hour (t-1, t] is t * value(t) - (t-1) * value(t-1); the 03Z ICON-EU steps
past +30 h are 6-hourly, so use their actual spacing there. ICON-D2's
radiation files each hold four 15-minute messages (+0/15/30/45 min past the
file's hour); every one is archived with its own valid_time.

Both models publish regular-lat-lon grib2 files directly (unlike the EPS
icosahedral grid), so lat/lon come from each file's own coordinates - no
separate invariant coordinate file is needed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pred_el_prices.pipeline.dwd import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    _download,
    aggregate_cells,
)

BASE_URL = "https://opendata.dwd.de/weather/nwp"

# Neighbour-country box for ICON-EU - a superset of the Germany box above, so
# one aggregation pass over it yields both: 1-degree cells share the same
# floor() edges regardless of which box they were computed from.
EU_LAT_MIN, EU_LAT_MAX = 45.0, 57.0
EU_LON_MIN, EU_LON_MAX = 2.0, 20.0

SINGLE_LEVEL_VARS = ["u_10m", "v_10m", "aswdir_s", "aswdifd_s", "t_2m", "clct"]

# +21h..+48h from 00Z covers the delivery day D fully (as in dwd.py); 03Z is
# the same window shifted 3h earlier, except ICON-EU's off-hour coarsening
# (see module docstring) which is why its 03Z list is hand-built.
STEPS = {
    ("icon-d2", 0): list(range(21, 49)),
    ("icon-d2", 3): list(range(18, 46)),
    ("icon-eu", 0): list(range(21, 49)),
    ("icon-eu", 3): list(range(18, 31)) + [36, 42, 48],
}


@dataclass(frozen=True)
class ModelSpec:
    domain: str  # DWD's domain word in the filename: "germany" / "europe"
    upper_var: bool  # ICON-EU spells variables upper-case in filenames
    single_level_tag: str | None  # ICON-D2 inserts "2d" before the var name
    bbox: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    # Lowest model levels above the 10 m diagnostic level -> approx AGL
    # metres (mean over the domain, from HHL/HSURF; terrain-following, so it
    # varies a little by cell, but not enough to matter for this feature).
    wind_levels: dict[int, int]


MODELS = {
    "icon-d2": ModelSpec(
        domain="germany",
        upper_var=False,
        single_level_tag="2d",
        bbox=(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX),
        wind_levels={64: 37, 63: 76},
    ),
    "icon-eu": ModelSpec(
        domain="europe",
        upper_var=True,
        single_level_tag=None,
        bbox=(EU_LAT_MIN, EU_LAT_MAX, EU_LON_MIN, EU_LON_MAX),
        wind_levels={73: 41, 72: 93},
    ),
}


def _var_token(spec: ModelSpec, var: str) -> str:
    return var.upper() if spec.upper_var else var.lower()


def single_level_url(model: str, run_date: date, run_hour: int, step: int, var: str) -> str:
    spec = MODELS[model]
    stamp = f"{run_date:%Y%m%d}{run_hour:02d}"
    tag = f"_{spec.single_level_tag}" if spec.single_level_tag else ""
    return (
        f"{BASE_URL}/{model}/grib/{run_hour:02d}/{var}/"
        f"{model}_{spec.domain}_regular-lat-lon_single-level_{stamp}_{step:03d}"
        f"{tag}_{_var_token(spec, var)}.grib2.bz2"
    )


def model_level_url(
    model: str, run_date: date, run_hour: int, step: int, var: str, level: int
) -> str:
    spec = MODELS[model]
    stamp = f"{run_date:%Y%m%d}{run_hour:02d}"
    return (
        f"{BASE_URL}/{model}/grib/{run_hour:02d}/{var}/"
        f"{model}_{spec.domain}_regular-lat-lon_model-level_{stamp}_{step:03d}"
        f"_{level}_{_var_token(spec, var)}.grib2.bz2"
    )


def _read_grid_field(
    raw_grib: bytes, tmp_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[tuple[timedelta | None, np.ndarray]]]:
    """Return flattened lat/lon plus one (lead time, values) pair per GRIB message.

    Most files hold a single message (lead time None: use the file's step).
    ICON-D2 radiation files hold four 15-minute messages (+0/15/30/45 min
    past the file's hour), each returned with its own lead time.
    """
    path = tmp_dir / "current.grib2"
    path.write_bytes(raw_grib)
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        (name,) = ds.data_vars
        da = ds[name]
        lat2d, lon2d = np.meshgrid(da.latitude.values, da.longitude.values, indexing="ij")
        if "step" in da.dims:
            fields = [
                (pd.Timedelta(s).to_pytimedelta(), da.isel(step=i).values.ravel())
                for i, s in enumerate(da.step.values)
            ]
        else:
            fields = [(None, da.values.ravel())]
        return lat2d.ravel(), lon2d.ravel(), fields


def _aggregate_field(lat: np.ndarray, lon: np.ndarray, values: np.ndarray, bbox) -> pd.DataFrame:
    df = aggregate_cells(lat, lon, values[np.newaxis, :], *bbox)
    return df.drop(columns="row")


def archive_run(model: str, run_date: date, run_hour: int, archive_dir: Path) -> Path:
    """Download and aggregate one ICON-D2/ICON-EU deterministic run. Idempotent."""
    spec = MODELS[model]
    out = (
        archive_dir / model / f"{run_date:%Y}" / f"{model}_{run_date:%Y%m%d}{run_hour:02d}.parquet"
    )
    if out.exists():
        print(f"already archived: {out}")
        return out

    steps = STEPS[(model, run_hour)]
    run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour, tzinfo=UTC)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for var in SINGLE_LEVEL_VARS:
            for step in steps:
                url = single_level_url(model, run_date, run_hour, step, var)
                lat, lon, fields = _read_grid_field(_download(url), tmp_dir)
                for lead, values in fields:
                    df = _aggregate_field(lat, lon, values, spec.bbox)
                    df["variable"] = var
                    df["level"] = np.nan
                    df["height_m"] = np.nan
                    df["valid_time"] = run_time + (lead or timedelta(hours=step))
                    frames.append(df)
            print(f"{var}: {len(steps)} steps done", flush=True)

        for level, height_m in spec.wind_levels.items():
            for var in ("u", "v"):
                for step in steps:
                    url = model_level_url(model, run_date, run_hour, step, var, level)
                    lat, lon, fields = _read_grid_field(_download(url), tmp_dir)
                    for lead, values in fields:
                        df = _aggregate_field(lat, lon, values, spec.bbox)
                        df["variable"] = var
                        df["level"] = level
                        df["height_m"] = height_m
                        df["valid_time"] = run_time + (lead or timedelta(hours=step))
                        frames.append(df)
                print(f"{var}@level {level} (~{height_m} m): {len(steps)} steps done", flush=True)

    result = pd.concat(frames, ignore_index=True)
    result["run_time"] = run_time
    for col in ("value", "level", "height_m"):
        result[col] = result[col].astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"archived {len(result)} rows -> {out}")
    return out


def archive_today(archive_dir: Path, run_date: date, runs: tuple[int, ...] = (0, 3)) -> list[Path]:
    """Archive every (model, run) combination not yet archived; skip-and-log failures.

    A run that isn't fully published yet on DWD's server (e.g. this slot ran
    before the 03Z run finished) must not block the other combinations - the
    workflow retries at its next slot regardless.
    """
    written = []
    for run_hour in runs:
        for model in MODELS:
            try:
                written.append(archive_run(model, run_date, run_hour, archive_dir))
            except Exception as e:  # noqa: BLE001 - genuinely any failure must not block siblings
                print(f"SKIP {model} {run_hour:02d}Z {run_date}: {e}", flush=True)
    return written
