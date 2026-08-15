"""Hourly RES-forecast features from the archived ECMWF ENS per-date Parquets.

Each archive file holds one 00Z run of D-1 as (cell_lat, cell_lon, member,
variable, valid_time) rows: per-member means over 1-degree cells covering
Germany, 3-hourly steps +21h..+48h. This module distills a run into hourly
ensemble-mean features for the UTC delivery day D:

- wind speed is computed per member/cell BEFORE averaging (the mean wind
  vector underestimates speed when directions disagree);
- ssrd arrives accumulated since forecast start; consecutive steps are
  differenced into 3-hour mean W/m^2 assigned to the interval midpoint;
- 3-hourly anchors are linearly interpolated to the 24 delivery hours (the
  downstream model gets hour-of-day features to fix residual diurnal shape).

Cell groups are deliberately coarse (national / north / south / sea): the
fleet-weighting refinement belongs to a later iteration, not v1.

Requires the 6-variable era (100 m wind + ssrd), i.e. runs from 2024-03-19.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# cell groups by lower-left cell latitude
NORTH_MIN = 53.0  # coastal/northern Germany, where most onshore wind sits
SOUTH_MAX = 51.0  # southern Germany, where the solar fleet skews
SEA_MIN = 54.0  # cells >= 54N are dominated by the North/Baltic Sea

WIND_VARS = {"u_100m", "v_100m", "u_10m", "v_10m"}


def _speed(df: pd.DataFrame, u_var: str, v_var: str) -> pd.DataFrame:
    """Per (valid_time, cell, member) wind speed from u/v components."""
    keys = ["valid_time", "cell_lat", "cell_lon", "member"]
    wide = (
        df[df["variable"].isin([u_var, v_var])]
        .pivot_table(index=keys, columns="variable", values="value")
        .dropna()
    )
    return np.hypot(wide[u_var], wide[v_var]).rename("value").reset_index()


def _group_means(values: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Ensemble-and-cell mean per valid_time for each named cell mask."""
    out = {}
    for name, mask in groups.items():
        out[name] = values[mask].groupby("valid_time")["value"].mean()
    return pd.DataFrame(out)


def _deaccumulate_ssrd(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulated J/m^2 -> 3-hour mean W/m^2 at the interval midpoint."""
    ssrd = df[df["variable"] == "ssrd"]
    keys = ["cell_lat", "cell_lon", "member"]
    wide = ssrd.pivot_table(index=keys, columns="valid_time", values="value").sort_index(axis=1)
    step_s = 3 * 3600
    rates = wide.diff(axis=1).iloc[:, 1:] / step_s
    rates.columns = rates.columns - pd.Timedelta(hours=1.5)
    long = rates.clip(lower=0.0).stack().rename("value").reset_index()
    return long.rename(columns={"level_3": "valid_time"})


def run_features(archive_path: Path) -> pd.DataFrame:
    """Hourly feature rows for the UTC delivery day covered by one archived run."""
    raw = pd.read_parquet(archive_path)
    run_time = raw["run_time"].iloc[0]
    delivery = (run_time + pd.Timedelta(days=1)).normalize()
    missing = WIND_VARS.union({"ssrd"}) - set(raw["variable"].unique())
    if missing:
        raise ValueError(
            f"{archive_path.name}: missing variables {sorted(missing)} (pre-6-var era?)"
        )

    def groups_for(values: pd.DataFrame) -> dict[str, pd.Series]:
        lat = values["cell_lat"]
        return {
            "nat": pd.Series(True, index=values.index),
            "north": lat >= NORTH_MIN,
            "south": lat < SOUTH_MAX,
            "sea": lat >= SEA_MIN,
        }

    ws100 = _speed(raw, "u_100m", "v_100m")
    ws10 = _speed(raw, "u_10m", "v_10m")
    t2m = raw[raw["variable"] == "t_2m"][["valid_time", "cell_lat", "cell_lon", "member", "value"]]
    ssrd = _deaccumulate_ssrd(raw)

    anchors = pd.concat(
        [
            _group_means(ws100, groups_for(ws100)).add_prefix("ws100_"),
            _group_means(ws10, {"nat": pd.Series(True, index=ws10.index)}).add_prefix("ws10_"),
            _group_means(t2m, {"nat": pd.Series(True, index=t2m.index)}).add_prefix("t2m_"),
            _group_means(
                ssrd, {k: v for k, v in groups_for(ssrd).items() if k in ("nat", "south")}
            ).add_prefix("ssrd_"),
        ],
        axis=1,
    ).sort_index()

    hours = pd.date_range(delivery, periods=24, freq="1h", tz="UTC")
    combined = anchors.reindex(anchors.index.union(hours)).interpolate(method="time")
    features = combined.reindex(hours)
    features[[c for c in features.columns if c.startswith("ssrd_")]] = features[
        [c for c in features.columns if c.startswith("ssrd_")]
    ].clip(lower=0.0)
    features["run_date"] = pd.Timestamp(run_time).date()
    features.index.name = "time_utc"
    return features


def build_features(archive_dir: Path, start: date, end: date) -> pd.DataFrame:
    """Concatenated hourly features for all archived runs in [start, end].

    Missing archive dates (the documented upstream holes) are skipped with a
    notice; the resulting frame is NOT guaranteed to be gap-free.
    """
    frames = []
    day = start
    while day <= end:
        path = archive_dir / "ecmwf-ens" / f"{day:%Y}" / f"ecmwf-ens_{day:%Y%m%d}00.parquet"
        if path.exists():
            frames.append(run_features(path))
        else:
            print(f"no archive for {day}, skipping")
        day += timedelta(days=1)
    if not frames:
        raise ValueError(f"no archived runs in [{start}, {end}] under {archive_dir}")
    return pd.concat(frames)
