"""Hourly RES-forecast features from the archived ECMWF ENS per-date Parquets.

Each archive file holds one run as (cell_lat, cell_lon, member, variable,
valid_time) rows: per-member means over 1-degree cells covering Germany,
3-hourly steps spanning the delivery day (00Z run of D-1, steps +21h..+48h;
or the 12Z fallback run of D-2, steps +33h..+60h). This module distills a
run into hourly ensemble-mean features for the UTC delivery day D:

- wind speed is computed per member/cell BEFORE averaging (the mean wind
  vector underestimates speed when directions disagree);
- ssrd arrives accumulated since forecast start; consecutive steps are
  differenced into 3-hour mean W/m^2 assigned to the interval midpoint;
- 3-hourly anchors are linearly interpolated to the 24 delivery hours (the
  downstream model gets hour-of-day features to fix residual diurnal shape).

v2 cell groups (registered 2026-08-15 after the v1 swap miss): wind gets a
north/center/south belt split plus separate North Sea and Baltic sea groups;
ssrd gets an east/west split for morning/evening cloud asymmetry; ensemble
q10/q90 across members for the headline wind/ssrd groups. True
capacity-weighting of cells stays a later refinement.

Requires the 6-variable era (100 m wind + ssrd), i.e. runs from 2024-03-19.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# cell groups by lower-left cell label (1-degree cells, Germany box 47-56N/5-16E)
NORTH_MIN = 53.0  # coastal/northern Germany, where most onshore wind sits
CENTER_MIN = 51.0  # central belt between the wind north and the solar south
SOUTH_MAX = 51.0  # southern Germany, where the solar fleet skews
SEA_MIN = 54.0  # cells >= 54N are dominated by the North/Baltic Sea
NORTHSEA_LON_MAX = 8.0  # German Bight clusters sit west of ~9E
BALTIC_LON_MIN = 10.0  # Baltic clusters east of ~11E; the 9E cell is mostly land
EAST_LON_MIN = 10.0  # ssrd east/west split for morning/evening cloud asymmetry
WEST_LON_MAX = 8.0

WIND_VARS = {"u_100m", "v_100m", "u_10m", "v_10m"}

QUANTILES = (0.1, 0.9)


def _speed(df: pd.DataFrame, u_var: str, v_var: str) -> pd.DataFrame:
    """Per (valid_time, cell, member) wind speed from u/v components."""
    keys = ["valid_time", "cell_lat", "cell_lon", "member"]
    wide = (
        df[df["variable"].isin([u_var, v_var])]
        .pivot_table(index=keys, columns="variable", values="value")
        .dropna()
    )
    return np.hypot(wide[u_var], wide[v_var]).rename("value").reset_index()


def _group_stats(
    values: pd.DataFrame,
    groups: dict[str, pd.Series],
    quantile_names: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Cell-group means per valid_time; ensemble q10/q90 for selected groups.

    Cells are averaged within each member first, so quantiles are proper
    across-member statistics (every member has the same cell count, so the
    mean of member-means equals the plain mean).
    """
    out = {}
    for name, mask in groups.items():
        member_means = values[mask].groupby(["valid_time", "member"])["value"].mean()
        by_time = member_means.groupby("valid_time")
        out[name] = by_time.mean()
        if name in quantile_names:
            for q in QUANTILES:
                out[f"{name}_q{int(q * 100)}"] = by_time.quantile(q)
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
    # +36h lands mid-delivery-day for both vintages: 00Z of D-1 and the
    # evening-before fallback 12Z of D-2 (steps +33..+60) both target day D.
    delivery = (run_time + pd.Timedelta(hours=36)).normalize()
    missing = WIND_VARS.union({"ssrd"}) - set(raw["variable"].unique())
    if missing:
        raise ValueError(
            f"{archive_path.name}: missing variables {sorted(missing)} (pre-6-var era?)"
        )

    def wind_groups(values: pd.DataFrame) -> dict[str, pd.Series]:
        lat, lon = values["cell_lat"], values["cell_lon"]
        return {
            "nat": pd.Series(True, index=values.index),
            "north": lat >= NORTH_MIN,
            "center": (lat >= CENTER_MIN) & (lat < NORTH_MIN),
            "south": lat < SOUTH_MAX,
            "northsea": (lat >= SEA_MIN) & (lon <= NORTHSEA_LON_MAX),
            "baltic": (lat >= SEA_MIN) & (lon >= BALTIC_LON_MIN),
        }

    def ssrd_groups(values: pd.DataFrame) -> dict[str, pd.Series]:
        lat, lon = values["cell_lat"], values["cell_lon"]
        return {
            "nat": pd.Series(True, index=values.index),
            "south": lat < SOUTH_MAX,
            "east": lon >= EAST_LON_MIN,
            "west": lon <= WEST_LON_MAX,
        }

    ws100 = _speed(raw, "u_100m", "v_100m")
    ws10 = _speed(raw, "u_10m", "v_10m")
    t2m = raw[raw["variable"] == "t_2m"][["valid_time", "cell_lat", "cell_lon", "member", "value"]]
    ssrd = _deaccumulate_ssrd(raw)

    anchors = pd.concat(
        [
            _group_stats(
                ws100, wind_groups(ws100), frozenset({"nat", "northsea", "baltic"})
            ).add_prefix("ws100_"),
            _group_stats(ws10, {"nat": pd.Series(True, index=ws10.index)}).add_prefix("ws10_"),
            _group_stats(
                t2m,
                {
                    "nat": pd.Series(True, index=t2m.index),
                    "south": t2m["cell_lat"] < SOUTH_MAX,
                },
            ).add_prefix("t2m_"),
            _group_stats(ssrd, ssrd_groups(ssrd), frozenset({"nat"})).add_prefix("ssrd_"),
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


def build_features(archive_dir: Path, start: date, end: date, run_hour: int = 0) -> pd.DataFrame:
    """Concatenated hourly features for all archived runs in [start, end].

    Missing archive dates (the documented upstream holes) are skipped with a
    notice; the resulting frame is NOT guaranteed to be gap-free.
    """
    frames = []
    day = start
    while day <= end:
        path = (
            archive_dir / "ecmwf-ens" / f"{day:%Y}" / f"ecmwf-ens_{day:%Y%m%d}{run_hour:02d}.parquet"
        )
        if path.exists():
            frames.append(run_features(path))
        else:
            print(f"no archive for {day}, skipping")
        day += timedelta(days=1)
    if not frames:
        raise ValueError(f"no archived runs in [{start}, {end}] under {archive_dir}")
    return pd.concat(frames)
