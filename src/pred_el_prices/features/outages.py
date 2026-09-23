"""Generation-unit unavailability known before the gate (ENTSO-E A80, DE_LU and FR).

The cutoff for delivery day D is c = D-1 10:00 UTC (the 12:00 CET/CEST gate
minus a margin). ENTSO-E re-stamped every outage message dated before its
2025-10-05..07 migration, so the version current at c is recoverable only from
2025-10 on (`exact_profile`, validation only). The model features use the
registered approximation from the latest version of each message
(`approx_profile`):
- planned messages: their latest schedule; cancelled/withdrawn ones drop out
  (the leak: postponements and cancellations published after c);
- forced messages: counted only if the outage started by c - 1 h (REMIT
  requires publication within one hour), with the MW unavailable at c held
  flat over all of D (persistence), so later revisions of the end never enter.
Unavailable MW of a message point = nominal_power - avail_qty over [start, end).
"""

from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    "thermal": [
        "Nuclear",
        "Fossil Brown coal/Lignite",
        "Fossil Hard coal",
        "Fossil Gas",
        "Fossil Coal-derived gas",
        "Fossil Oil",
    ],
    "nuclear": ["Nuclear"],
}
DISPATCHABLE = [
    *GROUPS["thermal"],
    "Biomass",
    "Waste",
    "Hydro Pumped Storage",
    "Hydro Water Reservoir",
]
INACTIVE = {"Cancelled", "Withdrawn"}
PLANNED = "Planned maintenance"
CUTOFF_HOUR_D_MINUS_1 = 10


def load_latest(cache_root: Path, zone: str) -> pd.DataFrame:
    """Latest version of every message, one row per availability point."""
    files = sorted((Path(cache_root) / "outages" / zone / "latest").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # a message spanning months appears in several monthly files, possibly at
    # different revisions: keep the newest revision seen
    newest = df.groupby("mrid")["revision"].transform("max")
    df = df[df["revision"] == newest]
    return df.drop_duplicates(["mrid", "revision", "start", "end"]).reset_index(drop=True)


def load_revisions(cache_root: Path, zone: str) -> pd.DataFrame:
    files = sorted((Path(cache_root) / "outages" / zone / "revisions").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["mrid", "revision", "start", "end"]).reset_index(drop=True)


def _mw(df: pd.DataFrame) -> np.ndarray:
    return (df["nominal_power"] - df["avail_qty"]).clip(lower=0).to_numpy()


def _hourly_sum(rows: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    """Hour-averaged MW of all rows' [start, end) intervals on an hourly UTC index."""
    minutes = np.zeros(len(index) * 60 + 1)
    t0 = index[0]
    s = ((rows["start"] - t0).dt.total_seconds() // 60).to_numpy()
    e = ((rows["end"] - t0).dt.total_seconds() // 60).to_numpy()
    s, e = np.clip(s, 0, len(minutes) - 1), np.clip(e, 0, len(minutes) - 1)
    mw = _mw(rows)
    np.add.at(minutes, s.astype(int), mw)
    np.add.at(minutes, e.astype(int), -mw)
    return np.cumsum(minutes)[:-1].reshape(len(index), 60).mean(axis=1)


def approx_profile(latest: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Unavailable MW known at each hour's cutoff, registered approximation."""
    active = latest[~latest["docstatus"].isin(INACTIVE)]
    planned = active[active["businesstype"] == PLANNED]
    out = pd.Series(_hourly_sum(planned, index), index=index)

    forced = active[active["businesstype"] != PLANNED]
    doc_start = forced.groupby("mrid")["start"].transform("min")
    days = index.normalize().unique()
    cutoffs = days - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR_D_MINUS_1)
    fs, fe = forced["start"].to_numpy(), forced["end"].to_numpy()
    ds, mw = doc_start.to_numpy(), _mw(forced)
    forced_at_cutoff = np.array(
        [
            mw[(fs <= c) & (fe > c) & (ds <= c - np.timedelta64(1, "h"))].sum()
            for c in cutoffs.to_numpy()
        ]
    )
    per_day = pd.Series(forced_at_cutoff, index=days)
    return out + per_day.reindex(index.normalize()).to_numpy()


def persistence_profile(latest: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Unavailable MW at each day's cutoff, held flat over the day (rule B).

    Any message (planned or forced) counts only if its outage began by c - 1 h
    and the latest version still has it running at c. Messages published ahead
    of time are ignored: the timestamps that would say which were public at c
    are gone for pre-2025-10 data.
    """
    active = latest[~latest["docstatus"].isin(INACTIVE)]
    doc_start = active.groupby("mrid")["start"].transform("min").to_numpy()
    fs, fe, mw = active["start"].to_numpy(), active["end"].to_numpy(), _mw(active)
    days = index.normalize().unique()
    cutoffs = days - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR_D_MINUS_1)
    at_cutoff = np.array(
        [
            mw[(fs <= c) & (fe > c) & (doc_start <= c - np.timedelta64(1, "h"))].sum()
            for c in cutoffs.to_numpy()
        ]
    )
    return pd.Series(at_cutoff, index=days).reindex(index.normalize()).set_axis(index)


def exact_profile(
    latest: pd.DataFrame, revisions: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.Series:
    """Unavailable MW from the version current at each day's cutoff (post-2025-10 only)."""
    allv = pd.concat([revisions, latest], ignore_index=True).drop_duplicates(
        ["mrid", "revision", "start", "end"]
    )
    out = pd.Series(0.0, index=index)
    for day in index.normalize().unique():
        c = day - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR_D_MINUS_1)
        known = allv[allv["created_doc_time"] <= c]
        if known.empty:
            continue
        current = known.groupby("mrid")["revision"].transform("max")
        known = known[(known["revision"] == current) & ~known["docstatus"].isin(INACTIVE)]
        hours = index[index.normalize() == day]
        out.loc[hours] = _hourly_sum(known, hours)
    return out


def installed(cache_root: Path, zone: str, types: list[str], index: pd.DatetimeIndex) -> pd.Series:
    """Installed MW of `types`, the latest yearly publication at or before each hour."""
    cap = pd.read_parquet(Path(cache_root) / "outages" / zone / "installed_capacity.parquet")
    total = cap.reindex(columns=types).fillna(0.0).sum(axis=1).sort_index()
    return total.reindex(index, method="ffill")


def outage_features(cache_root: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly pre-gate unavailability features for the dataset (registered approximation)."""
    out = pd.DataFrame(index=index)
    de = load_latest(cache_root, "DE_LU")
    fr = load_latest(cache_root, "FR")
    out["unavail_de_thermal_mw"] = approx_profile(de[de["plant_type"].isin(GROUPS["thermal"])], index)
    out["unavail_de_dispatchable_mw"] = approx_profile(de[de["plant_type"].isin(DISPATCHABLE)], index)
    out["unavail_fr_nuclear_mw"] = approx_profile(fr[fr["plant_type"].isin(GROUPS["nuclear"])], index)
    out["unavail_fr_total_mw"] = approx_profile(fr[fr["plant_type"].isin(DISPATCHABLE)], index)
    out["installed_de_dispatchable_mw"] = installed(cache_root, "DE_LU", DISPATCHABLE, index)
    # no outage history before the first cached month: unknown, not zero
    first = min(de["start"].min(), fr["start"].min()).normalize() + pd.offsets.MonthBegin(1)
    return out.where(out.index >= first)
